"""LLM General Expert Reranker for the Dual-Expert architecture.

The LLM receives the top-K L2 candidates from the Domain Expert and
chooses exactly one — it is a reranker, not a free classifier.

Key design constraints:
- The LLM must select from the candidate list only (no hallucinated categories).
- JSON output is required.
- Fallback to Domain Expert top-1 if parsing or validation fails.
- Supports providers: 'local' (HTTP endpoint), 'mistral_api',
  'openai_compatible', 'hf_local' (in-process HuggingFace model),
  and 'none' (dry-run for testing).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LLMRerankerConfig:
    """Configuration for the LLM General Expert Reranker."""

    enabled: bool = False
    provider: str = "none"              # none | local | mistral_api | openai_compatible
    model_name: str = "mixtral-8x7b-instruct"
    temperature: float = 0.0
    max_new_tokens: int = 512
    top_k: int = 10
    prompt_path: str = "prompts/llm_reranker_fr.txt"
    require_json: bool = True
    fallback_to_domain_expert: bool = True
    timeout_seconds: float = 15.0
    retry_count: int = 1
    # For local / openai_compatible providers
    api_base_url: str = "http://localhost:8080/v1"
    api_key: str = "none"
    # For hf_local provider
    hf_load_in_4bit: bool = False   # quantize to 4-bit via bitsandbytes


@dataclass
class RerankResult:
    """Output of the LLM reranking step."""

    selected_l2_id: int
    selected_path: str = ""
    confidence: str = "medium"          # high | medium | low
    reason: str = ""
    raw_response: str = ""
    valid_json: bool = True
    fallback_used: bool = False
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# JSON parsing and validation
# ---------------------------------------------------------------------------


def parse_llm_json(response_text: str) -> dict:
    """Robustly parse a JSON object from LLM response text.

    Tries exact parse first; falls back to regex extraction if the LLM
    wraps the JSON in markdown fences or extra text.

    Args:
        response_text: Raw LLM output string.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If no valid JSON object is found.
    """
    text = response_text.strip()
    # Remove common markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass  # malformed JSON — fall through to field extraction

    # Fallback: extract fields individually via regex (handles truncated or malformed JSON)
    result = {}
    for field in ("definition_fr", "confidence"):
        m = re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"', text, flags=re.DOTALL)
        if m:
            result[field] = m.group(1).replace('\\"', '"').replace("\\n", " ").strip()
    for field in ("included_products", "excluded_products", "positive_signals", "negative_signals", "ambiguities"):
        m = re.search(rf'"{field}"\s*:\s*(\[.*?\])', text, flags=re.DOTALL)
        if m:
            try:
                result[field] = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    if result.get("definition_fr"):
        return result

    raise ValueError(f"No JSON object found in LLM response: {text[:200]!r}")


def validate_llm_output(
    output: dict,
    candidate_l2_ids: list[int],
) -> dict:
    """Validate that the LLM's selected L2 is in the candidate list.

    Args:
        output: Parsed LLM output dict.
        candidate_l2_ids: Valid L2 ids from the candidate list.

    Returns:
        Validated (possibly corrected) output dict.

    Raises:
        ValueError: If 'selected_l2_id' is missing or not in candidate list.
    """
    if "selected_l2_id" not in output:
        raise ValueError("LLM output missing 'selected_l2_id'")

    selected = int(output["selected_l2_id"])
    if selected not in candidate_l2_ids:
        raise ValueError(
            f"LLM selected l2_id={selected} not in candidates: {candidate_l2_ids}"
        )

    if output.get("confidence") not in ("high", "medium", "low"):
        output["confidence"] = "medium"

    output["selected_l2_id"] = selected
    return output


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


class PromptBuilder:
    """Build French-language reranking prompts for the LLM.

    Args:
        prompt_template_path: Path to the prompt template file.
    """

    _DEFAULT_TEMPLATE = """Tu es un expert en classification de produits e-commerce.

Ta tâche est de choisir exactement UNE catégorie L2 parmi les catégories candidates.
Tu ne dois jamais proposer une catégorie absente de la liste.
Tu dois classer selon la fonction principale et l'usage prévu du produit.
La désignation est prioritaire si la description est bruitée ou marketing.
Ne choisis pas une catégorie seulement parce qu'un mot est présent : compare le sens global.

Produit :
- Désignation : {designation}
- Description : {description}

Catégories candidates :
{candidate_blocks}

Pour choisir :
1. Identifie les indices importants dans la désignation et la description.
2. Compare les catégories proches.
3. Applique les exclusions indiquées dans les définitions.
4. Sélectionne uniquement une catégorie candidate.

Réponds uniquement en JSON valide :
{
  "selected_l2_id": <int>,
  "selected_path": "<L1 > L2>",
  "confidence": "high|medium|low",
  "reason": "<raison courte en français>"
}"""

    def __init__(self, prompt_template_path: Optional[str | Path] = None) -> None:
        self._template = self._DEFAULT_TEMPLATE
        if prompt_template_path is not None:
            p = Path(prompt_template_path)
            if p.exists():
                self._template = p.read_text(encoding="utf-8")
            else:
                print(f"[PromptBuilder] Template not found: {p} — using default.")

    def build(
        self,
        product: dict,
        candidates: list[dict],
    ) -> str:
        """Build the reranking prompt.

        Args:
            product: Dict with 'designation' and 'description'.
            candidates: List of candidate dicts from TopKCandidateGenerator,
                        optionally enriched with 'definition_fr',
                        'positive_signals', 'negative_signals'.

        Returns:
            Full prompt string.
        """
        candidate_blocks = "\n\n".join(
            self._format_candidate(c, rank=i + 1) for i, c in enumerate(candidates)
        )
        return (
            self._template
            .replace("{designation}", product.get("designation", ""))
            .replace("{description}", product.get("description", ""))
            .replace("{candidate_blocks}", candidate_blocks)
        )

    def _format_candidate(self, candidate: dict, rank: int) -> str:
        l2_id = candidate.get("l2_id", "?")
        l2_label = candidate.get("l2_label", str(l2_id))
        l1_label = candidate.get("parent_l1_label", candidate.get("parent_l1_id", "?"))
        path_text = f"{l1_label} > {l2_label}"
        definition = candidate.get("definition_fr", "")
        positive = ", ".join(candidate.get("positive_signals", []))
        negative = ", ".join(candidate.get("negative_signals", []))
        hierarchy_score = candidate.get("hierarchy_score", "?")
        prob_l2 = candidate.get("prob_l2", "?")

        lines = [f"[{rank}] ID L2: {l2_id}"]
        lines.append(f"Chemin: {path_text}")
        if definition:
            lines.append(f"Définition: {definition}")
        if positive:
            lines.append(f"Signaux positifs: {positive}")
        if negative:
            lines.append(f"Exclusions / signaux négatifs: {negative}")
        # if isinstance(prob_l2, float):
        #     lines.append(f"P(L2|texte): {prob_l2:.4f}")
        # if isinstance(hierarchy_score, float):
        #     lines.append(f"Score hiérarchique: {hierarchy_score:.4f}")
        # else:
        #     lines.append(f"Score hiérarchique: {hierarchy_score}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------


class _NoneClient:
    """Dry-run client that always returns a fallback response."""

    def generate(self, prompt: str, config: LLMRerankerConfig) -> str:
        return json.dumps({
            "selected_l2_id": -1,
            "selected_path": "dry-run",
            "confidence": "low",
            "reason": "dry-run mode — no LLM configured",
        })


class _OpenAICompatibleClient:
    """Client for OpenAI-compatible API endpoints (local or cloud).

    Works with llama.cpp server, vLLM, Ollama, etc. that expose /chat/completions.
    """

    def __init__(self, config: LLMRerankerConfig) -> None:
        try:
            import requests
            self._requests = requests
        except ImportError as e:
            raise ImportError("requests is required for LLM API calls: pip install requests") from e
        self._config = config

    def generate(self, prompt: str, config: LLMRerankerConfig) -> str:
        payload = {
            "model": config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            "max_tokens": config.max_new_tokens,
        }
        headers = {"Authorization": f"Bearer {config.api_key}"}
        url = f"{config.api_base_url.rstrip('/')}/chat/completions"

        resp = self._requests.post(
            url, json=payload, headers=headers, timeout=config.timeout_seconds
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class _HFLocalClient:
    """In-process HuggingFace client — loads the model once, runs inference in-memory.

    No HTTP server needed. The model is loaded on the first call and kept alive
    for the lifetime of the reranker.

    provider = "hf_local"
    model_name = any HuggingFace model id or local path
                 e.g. "mistralai/Mistral-7B-Instruct-v0.2"
                      "/scratch/models/mixtral-8x7b"

    Requires: transformers, torch (and optionally bitsandbytes for 4-bit).
    Set load_in_4bit=True via hf_load_in_4bit config field to quantize.
    """

    def __init__(self, config: LLMRerankerConfig) -> None:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for hf_local provider"
            ) from e

        print(f"[HFLocalClient] Loading model: {config.model_name} ...")
        load_kwargs: dict = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        if getattr(config, "hf_load_in_4bit", False):
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        import os
        _local = os.path.isdir(config.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True,
            local_files_only=_local,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            config.model_name, local_files_only=_local, **load_kwargs
        )
        self._model.eval()
        self._config = config
        print(f"[HFLocalClient] Model ready.")

    def generate(self, prompt: str, config: LLMRerankerConfig) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        # Use apply_chat_template if available (instruct models), else raw prompt
        if hasattr(self._tokenizer, "apply_chat_template"):
            input_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = prompt

        # Measure prompt length before truncation to detect silent cuts
        raw_ids = self._tokenizer(input_text, return_tensors="pt", truncation=False)
        n_tokens = raw_ids["input_ids"].shape[1]
        if n_tokens > 4096:
            print(
                f"[HFLocalClient] WARNING: prompt is {n_tokens} tokens — "
                f"truncated to 4096. Designation/description or candidates may be cut."
            )
        else:
            print(f"[HFLocalClient] Prompt: {n_tokens} tokens (limit: 4096)")

        inputs = self._tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=4096
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=max(config.temperature, 1e-6),
                do_sample=config.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True)


def _make_client(config: LLMRerankerConfig):
    """Factory that creates the appropriate LLM client."""
    if config.provider == "none":
        return _NoneClient()
    if config.provider == "hf_local":
        return _HFLocalClient(config)
    if config.provider in ("local", "openai_compatible", "mistral_api"):
        return _OpenAICompatibleClient(config)
    raise ValueError(f"Unknown LLM provider: '{config.provider}'")


# ---------------------------------------------------------------------------
# Main reranker
# ---------------------------------------------------------------------------


class LLMGeneralExpertReranker:
    """Reranks top-K L2 candidates using an LLM.

    The LLM must select exactly one L2 from the provided candidate list.
    Fallback to Domain Expert top-1 on any error if configured.

    Args:
        config: Reranker configuration.
        category_definition_store: Optional CategoryDefinitionStore for
            enriching candidates with offline-generated definitions.
    """

    def __init__(
        self,
        config: Optional[LLMRerankerConfig] = None,
        category_definition_store=None,
    ) -> None:
        self.config = config or LLMRerankerConfig()
        self.definition_store = category_definition_store
        self.prompt_builder = PromptBuilder(self.config.prompt_path)
        self._client = _make_client(self.config) if self.config.enabled else _NoneClient()

    def rerank(
        self,
        product: dict,
        candidates: list[dict],
    ) -> RerankResult:
        """Rerank top-K L2 candidates for a single product.

        Args:
            product: Dict with 'designation' and 'description'.
            candidates: List of candidate dicts (from TopKCandidateGenerator).
                        Each dict must contain 'l2_id'.

        Returns:
            RerankResult with selected L2 and metadata.
        """
        if not candidates:
            return RerankResult(
                selected_l2_id=-1,
                fallback_used=True,
                reason="empty candidate list",
                confidence="low",
            )

        # Enrich candidates with definitions if store is available
        enriched = self._enrich_candidates(candidates)
        candidate_l2_ids = [int(c["l2_id"]) for c in enriched]

        prompt = self.prompt_builder.build(product, enriched)

        t0 = time.monotonic()
        for attempt in range(max(1, self.config.retry_count)):
            try:
                raw_response = self._client.generate(prompt, self.config)
                parsed = parse_llm_json(raw_response)
                validated = validate_llm_output(parsed, candidate_l2_ids)
                latency_ms = (time.monotonic() - t0) * 1000
                return RerankResult(
                    selected_l2_id=validated["selected_l2_id"],
                    selected_path=validated.get("selected_path", ""),
                    confidence=validated.get("confidence", "medium"),
                    reason=validated.get("reason", ""),
                    raw_response=raw_response,
                    valid_json=True,
                    fallback_used=False,
                    latency_ms=latency_ms,
                )
            except Exception as e:
                if attempt == self.config.retry_count - 1:
                    latency_ms = (time.monotonic() - t0) * 1000
                    if self.config.fallback_to_domain_expert:
                        fallback_l2 = candidate_l2_ids[0]
                        return RerankResult(
                            selected_l2_id=fallback_l2,
                            confidence="low",
                            reason=f"llm_error: {e}",
                            raw_response=getattr(e, "response", ""),
                            valid_json=False,
                            fallback_used=True,
                            latency_ms=latency_ms,
                        )
                    raise

        # Should never reach here
        return RerankResult(
            selected_l2_id=candidate_l2_ids[0],
            fallback_used=True,
            reason="retry_exhausted",
        )

    def _enrich_candidates(self, candidates: list[dict]) -> list[dict]:
        """Merge category definitions into candidate dicts if store is set."""
        if self.definition_store is None:
            return candidates
        enriched = []
        for c in candidates:
            c_copy = dict(c)
            defn = self.definition_store.get(int(c["l2_id"]))
            c_copy["definition_fr"] = defn.get("definition_fr", "")
            c_copy["positive_signals"] = defn.get("positive_signals", [])
            c_copy["negative_signals"] = defn.get("negative_signals", [])
            enriched.append(c_copy)
        return enriched
