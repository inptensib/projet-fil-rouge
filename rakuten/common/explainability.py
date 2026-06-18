"""Token-level explainability for HYDRA models via Gradient × Input attribution.

Generates HTML reports with highlighted tokens showing which words contributed
most to L1 and L2 predictions. Designed for MLflow artifact logging.
"""

from __future__ import annotations

import logging
from html import escape
from typing import Optional

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F


def compute_token_attributions(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    device: torch.device,
    parent_child_matrix: Optional[torch.Tensor] = None,
) -> dict:
    """Compute token-level attributions for L1 and L2 predictions.

    Uses Gradient × Input on the word embeddings to determine which tokens
    contributed most to each prediction.

    Args:
        model: Unwrapped HYDRA model (not DataParallel).
        tokenizer: HuggingFace tokenizer.
        text: Input text string.
        device: Torch device.
        parent_child_matrix: Optional hierarchy matrix (num_l1, num_l2) for
            masking L2 logits based on predicted L1.

    Returns:
        Dictionary with keys: tokens, attributions_l1, attributions_l2,
        pred_l1, pred_l2, confidence_l1, confidence_l2.
    """
    model.eval()

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=256, padding="max_length",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    seq_len = attention_mask.sum().item()

    # Get word embeddings with gradient tracking
    word_embeddings = model.encoder.embeddings.word_embeddings
    embeds = word_embeddings(input_ids)
    embeds.requires_grad_(True)
    embeds.retain_grad()

    # Forward pass through encoder using embeddings directly
    encoder_output = model.encoder(
        inputs_embeds=embeds,
        attention_mask=attention_mask,
    )
    hidden_states = encoder_output.last_hidden_state
    pooled = model._pool(hidden_states, attention_mask)

    if model.project_embedding:
        pooled = model.embedding_projector(pooled)

    logits_l1 = model.level_1_head(pooled)
    logits_l2 = model.level_2_head(pooled)

    # Masquer L2 invalides si la matrice hiérarchique est fournie
    if parent_child_matrix is not None:
        pred_l1_for_mask = torch.argmax(logits_l1, dim=-1)
        valid_l2_mask = parent_child_matrix[pred_l1_for_mask]
        logits_l2 = logits_l2.clone()
        logits_l2[valid_l2_mask == 0] = float("-inf")

    probs_l1 = F.softmax(logits_l1, dim=-1)
    probs_l2 = F.softmax(logits_l2, dim=-1)
    conf_l1, pred_l1 = probs_l1.max(dim=-1)
    conf_l2, pred_l2 = probs_l2.max(dim=-1)

    # Attribution for L1: backprop predicted class logit
    model.zero_grad()
    logits_l1[0, pred_l1].backward(retain_graph=True)
    grad_l1 = embeds.grad.clone()

    # Attribution for L2: backprop predicted class logit
    embeds.grad = None
    logits_l2[0, pred_l2].backward()
    grad_l2 = embeds.grad.clone()

    # Gradient × Input → sum over embedding dim → per-token score
    attr_l1 = (grad_l1 * embeds.detach())[0, :seq_len].sum(dim=-1).abs()
    attr_l2 = (grad_l2 * embeds.detach())[0, :seq_len].sum(dim=-1).abs()

    # Normalize to [0, 1]
    if attr_l1.max() > 0:
        attr_l1 = attr_l1 / attr_l1.max()
    if attr_l2.max() > 0:
        attr_l2 = attr_l2 / attr_l2.max()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0, :seq_len].cpu())

    return {
        "tokens": tokens,
        "attributions_l1": attr_l1.cpu().tolist(),
        "attributions_l2": attr_l2.cpu().tolist(),
        "pred_l1": pred_l1.item(),
        "pred_l2": pred_l2.item(),
        "confidence_l1": conf_l1.item(),
        "confidence_l2": conf_l2.item(),
    }


def _score_to_color(score: float, correct: bool) -> str:
    """Convert attribution score to background color.

    Green tones for correct predictions, red tones for errors.
    """
    intensity = min(score, 1.0)
    if correct:
        return f"rgba(39, 174, 96, {intensity * 0.8:.2f})"
    return f"rgba(231, 76, 60, {intensity * 0.8:.2f})"


def generate_highlight_html(
    tokens: list[str],
    attributions_l1: list[float],
    attributions_l2: list[float],
    pred_l1: str,
    true_l1: str,
    pred_l2: str,
    true_l2: str,
    text: str,
    confidence_l1: float,
    confidence_l2: float,
) -> str:
    """Generate HTML snippet for one sample with highlighted tokens.

    Args:
        tokens: List of token strings.
        attributions_l1: Attribution scores for L1 (0-1 per token).
        attributions_l2: Attribution scores for L2 (0-1 per token).
        pred_l1, true_l1: Predicted and true L1 class names.
        pred_l2, true_l2: Predicted and true L2 class names.
        text: Original text.
        confidence_l1, confidence_l2: Prediction confidence scores.

    Returns:
        HTML string.
    """
    correct_l1 = pred_l1 == true_l1
    correct_l2 = pred_l2 == true_l2
    icon_l1 = "&#10004;" if correct_l1 else "&#10008;"
    icon_l2 = "&#10004;" if correct_l2 else "&#10008;"

    # Build highlighted token spans
    def _build_spans(attributions, correct):
        spans = []
        for tok, score in zip(tokens, attributions):
            display = tok.replace("▁", " ").replace("<s>", "").replace("</s>", "")
            if not display.strip():
                continue
            color = _score_to_color(score, correct)
            spans.append(
                f'<span style="background:{color};padding:2px 1px;'
                f'border-radius:3px;margin:0 1px"'
                f' title="score: {score:.3f}">{escape(display)}</span>'
            )
        return " ".join(spans)

    spans_l1 = _build_spans(attributions_l1, correct_l1)
    spans_l2 = _build_spans(attributions_l2, correct_l2)

    return f"""
    <div style="border:1px solid #ddd;border-radius:8px;padding:16px;margin:12px 0;
                background:#fafafa;font-family:sans-serif;font-size:14px">
      <p style="color:#555;margin:0 0 8px"><b>Texte :</b> {escape(text[:200])}{'...' if len(text) > 200 else ''}</p>
      <p style="margin:4px 0"><b>L1</b> &mdash;
        Vrai: <code>{escape(true_l1)}</code> |
        Pr&eacute;dit: <code>{escape(pred_l1)}</code>
        (conf: {confidence_l1:.1%}) {icon_l1}</p>
      <p style="line-height:2">{spans_l1}</p>
      <p style="margin:4px 0"><b>L2</b> &mdash;
        Vrai: <code>{escape(true_l2)}</code> |
        Pr&eacute;dit: <code>{escape(pred_l2)}</code>
        (conf: {confidence_l2:.1%}) {icon_l2}</p>
      <p style="line-height:2">{spans_l2}</p>
    </div>"""


def generate_explainability_report(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    labels_l1: list[int],
    labels_l2: list[int],
    preds_l1: list[int],
    preds_l2: list[int],
    l1_encoder,
    l2_encoder,
    device: torch.device,
    confs_l2: Optional[list[float]] = None,
    parent_child_matrix: Optional[torch.Tensor] = None,
    n_errors: int = 10,
    n_correct: int = 10,
) -> str:
    """Generate a full HTML explainability report.

    Selects error cases and correct predictions, computes attributions,
    and generates highlighted HTML for each.

    Errors are sorted by highest confidence (model sure but wrong).
    Correct predictions are sorted by lowest confidence (model unsure but right).

    Args:
        model: Unwrapped HYDRA model.
        tokenizer: HuggingFace tokenizer.
        texts: List of input texts.
        labels_l1, labels_l2: Ground truth encoded labels.
        preds_l1, preds_l2: Predicted encoded labels.
        l1_encoder, l2_encoder: Fitted LabelEncoders.
        device: Torch device.
        confs_l2: Optional L2 confidence scores for ranking.
        n_errors: Number of error samples to include.
        n_correct: Number of correct samples to include.

    Returns:
        Complete HTML string for MLflow artifact.
    """
    # Split into errors and correct predictions
    errors_idx = []
    correct_idx = []
    for i in range(len(texts)):
        if preds_l1[i] != labels_l1[i] or preds_l2[i] != labels_l2[i]:
            errors_idx.append(i)
        else:
            correct_idx.append(i)

    if confs_l2:
        # Errors: highest confidence first (model sure but wrong = hardest)
        errors_idx.sort(key=lambda i: confs_l2[i], reverse=True)
        # Correct: lowest confidence first (model unsure but right)
        correct_idx.sort(key=lambda i: confs_l2[i])

    selected_errors = errors_idx[:n_errors]
    selected_correct = correct_idx[:n_correct]

    html_parts = ["""
    <!DOCTYPE html>
    <html><head><meta charset="utf-8">
    <title>HYDRA - Rapport d'Explicabilit&eacute;</title>
    <style>
      body { font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }
      h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
      h2 { color: #e74c3c; margin-top: 30px; }
      h2.correct { color: #27ae60; }
      .legend { background: #f0f0f0; padding: 12px; border-radius: 6px; margin: 16px 0; font-size: 13px; }
      code { background: #eef; padding: 2px 6px; border-radius: 3px; }
    </style>
    </head><body>
    <h1>&#128269; Rapport d'Explicabilit&eacute; HYDRA</h1>
    <div class="legend">
      <b>L&eacute;gende :</b> Plus la couleur est intense, plus le token a contribu&eacute; &agrave; la pr&eacute;diction.<br>
      <span style="background:rgba(231,76,60,0.6);padding:2px 6px;border-radius:3px">Rouge</span> = erreur &nbsp;
      <span style="background:rgba(39,174,96,0.6);padding:2px 6px;border-radius:3px">Vert</span> = correct &nbsp;
      Survolez un token pour voir son score d'attribution.
    </div>
    """]

    if selected_errors:
        html_parts.append(f"<h2>&#10008; Erreurs ({len(selected_errors)} exemples)</h2>")
        for i in selected_errors:
            try:
                attr = compute_token_attributions(model, tokenizer, texts[i], device, parent_child_matrix)
                html_parts.append(generate_highlight_html(
                    tokens=attr["tokens"],
                    attributions_l1=attr["attributions_l1"],
                    attributions_l2=attr["attributions_l2"],
                    pred_l1=l1_encoder.inverse_transform([attr["pred_l1"]])[0],
                    true_l1=l1_encoder.inverse_transform([labels_l1[i]])[0],
                    pred_l2=l2_encoder.inverse_transform([attr["pred_l2"]])[0],
                    true_l2=l2_encoder.inverse_transform([labels_l2[i]])[0],
                    text=texts[i],
                    confidence_l1=attr["confidence_l1"],
                    confidence_l2=attr["confidence_l2"],
                ))
            except Exception as e:
                logger.warning("Attribution failed for sample %d: %s", i, e)
                continue

    if selected_correct:
        html_parts.append(f'<h2 class="correct">&#10004; Pr&eacute;dictions correctes ({len(selected_correct)} exemples)</h2>')
        for i in selected_correct:
            try:
                attr = compute_token_attributions(model, tokenizer, texts[i], device, parent_child_matrix)
                html_parts.append(generate_highlight_html(
                    tokens=attr["tokens"],
                    attributions_l1=attr["attributions_l1"],
                    attributions_l2=attr["attributions_l2"],
                    pred_l1=l1_encoder.inverse_transform([attr["pred_l1"]])[0],
                    true_l1=l1_encoder.inverse_transform([labels_l1[i]])[0],
                    pred_l2=l2_encoder.inverse_transform([attr["pred_l2"]])[0],
                    true_l2=l2_encoder.inverse_transform([labels_l2[i]])[0],
                    text=texts[i],
                    confidence_l1=attr["confidence_l1"],
                    confidence_l2=attr["confidence_l2"],
                ))
            except Exception as e:
                logger.warning("Attribution failed for sample %d: %s", i, e)
                continue

    html_parts.append("</body></html>")
    return "\n".join(html_parts)
