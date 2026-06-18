"""Text normalization for the Rakuten corpus.

Repairs HTML entities and tags, mojibake (UTF-8 read as latin-1/cp1252),
all-uppercase titles and whitespace. Applied identically to the three
splits at load time; the JSONL files are never modified.
"""

from __future__ import annotations

import html
import re

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

# Markers typical of UTF-8 text decoded as latin-1/cp1252.
_MOJIBAKE_MARKER_RE = re.compile("Ã|Â|â€|ï¿½|ã[\u0080-¿]|â[\u0080-\u009f]")

# Frequent corpus sequences that the latin-1 round-trip cannot repair
# (mojibake mixed with clean text, title-cased mojibake, lost bytes).
# Order matters: longest and most specific entries first.
_MOJIBAKE_TABLE: tuple[tuple[str, str], ...] = (
    # UTF-8 punctuation seen through latin-1 (\x80-\x9f kept as C1 controls)
    ("â\u0080\u0099", "'"),
    ("â\u0080\u0098", "'"),
    ("â\u0080\u009c", '"'),
    ("â\u0080\u009d", '"'),
    ("â\u0080\u0093", "–"),
    ("â\u0080\u0094", "—"),
    ("â\u0080¦", "…"),
    ("â\u0080¢", "•"),
    # Same punctuation seen through cp1252
    ("â€™", "'"),
    ("â€˜", "'"),
    ("â€œ", '"'),
    ("â€\u009d", '"'),
    ("â€“", "–"),
    ("â€”", "—"),
    ("â€¦", "…"),
    ("â€¢", "•"),
    # CJK brackets and ideographic space (mojibake of scraped listings)
    ("ã\u0080\u0090", "【"),
    ("ã\u0080\u0091", "】"),
    ("ã\u0080\u0080", " "),
    # oe ligature, both latin-1 and cp1252 views
    ("Å\u0093", "œ"),
    ("Å“", "œ"),
    ("Å\u0092", "Œ"),
    ("Å’", "Œ"),
    # e-grave and o-circumflex both collapsed to y-diaeresis upstream
    # (matiÃ¿re, contrÃ¿le); e-grave dominates, contrôl* is the main
    # exception worth special-casing.
    ("ontrÃ¿l", "ontrôl"),
    ("ontrã¿l", "ontrôl"),
    ("ontrÿl", "ontrôl"),
    ("Ã¿", "è"),
    ("ã¿", "è"),
    # a-grave whose non-breaking space became a plain space
    ("Ã ", "à "),
    ("ã ", "à "),
    ("Â ", " "),
    # Apostrophe whose last byte was lost (dâ€ingrédients)
    ("â€", "'"),
)

# Two-byte UTF-8 sequence seen through latin-1: Ã (or ã after
# title-casing) followed by a continuation byte maps back to
# U+00C0-U+00FF (Ã© -> é, Ã\x89 -> É).
_LATIN1_PAIR_RE = re.compile("[Ãã]([\u0080-¿])")
# Â before a symbol in U+00A0-U+00BF is a spurious lead byte (Â° -> °);
# Â before a letter (Âge) is legitimate and left untouched.
_SPURIOUS_LEAD_RE = re.compile("Â([\u00a0-¿])")
# y-diaeresis between letters is the round-tripped form of the upstream
# e-grave corruption (matiÿre -> matière); legitimate French ÿ is
# virtually absent from the corpus.
_BROKEN_YDIA_RE = re.compile("(?<=[A-Za-z])ÿ(?=[A-Za-z])")

_SENTENCE_SPLIT_RE = re.compile(r"([.!?]+\s*)")

_MIN_ALPHA_FOR_CASE_FIX = 20
_UPPERCASE_RATIO_THRESHOLD = 0.9
# Each repair pass shrinks the text or maps to clean characters; nested
# encodings (Ã\x82 -> Â -> removed) converge in 2 passes, 4 is a margin.
_MAX_REPAIR_PASSES = 4


def _apply_repairs(text: str) -> str:
    """Apply one pass of table and pattern repairs."""
    for bad, good in _MOJIBAKE_TABLE:
        if bad in text:
            text = text.replace(bad, good)
    text = _BROKEN_YDIA_RE.sub("è", text)
    text = _LATIN1_PAIR_RE.sub(
        lambda m: chr(0xC0 + ord(m.group(1)) - 0x80), text
    )
    return _SPURIOUS_LEAD_RE.sub(r"\1", text)


def _fix_mojibake(text: str) -> str:
    """Repair UTF-8 text that was decoded as latin-1/cp1252.

    Tries a strict latin-1 round-trip first, kept only if it reduces the
    number of mojibake markers, then applies corpus-calibrated repairs
    until fixpoint. U+FFFD is irreparable and left as is.
    """
    markers = len(_MOJIBAKE_MARKER_RE.findall(text))
    if markers == 0:
        return text
    try:
        candidate = text.encode("latin-1", errors="strict").decode(
            "utf-8", errors="strict"
        )
        if len(_MOJIBAKE_MARKER_RE.findall(candidate)) < markers:
            text = candidate
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    for _ in range(_MAX_REPAIR_PASSES):
        repaired = _apply_repairs(text)
        if repaired == text:
            break
        text = repaired
    return text


def _is_mostly_uppercase(text: str) -> bool:
    """Return True for all-uppercase titles worth re-casing."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < _MIN_ALPHA_FOR_CASE_FIX:
        return False
    upper = sum(c.isupper() for c in letters)
    return upper / len(letters) >= _UPPERCASE_RATIO_THRESHOLD


def _capitalize_sentences(text: str) -> str:
    """Capitalize each sentence (first letter upper, rest lower)."""
    parts = _SENTENCE_SPLIT_RE.split(text)
    return "".join(
        part if i % 2 else part.capitalize() for i, part in enumerate(parts)
    )


def normalize_text(text: str | None) -> str:
    """Normalize a raw corpus field (designation or description).

    Pure and idempotent. Pipeline: HTML entities (unescaped twice for
    double-encoded scrapes), HTML tags, mojibake repair, all-uppercase
    titles, whitespace collapse.

    Args:
        text: Raw text, possibly None.

    Returns:
        Normalized text, "" for None or non-string input.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(html.unescape(text))
    text = _HTML_TAG_RE.sub(" ", text)
    text = _fix_mojibake(text)
    if _is_mostly_uppercase(text):
        text = _capitalize_sentences(text)
    return _WHITESPACE_RE.sub(" ", text).strip()
