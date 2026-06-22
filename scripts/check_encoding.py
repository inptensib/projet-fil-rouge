"""
Check encoding integrity of the taxonomy CSV and JSONL data splits.
Detects: question-mark substitutions, mojibake (latin-1 mis-read as UTF-8),
and stray control characters.

Usage:
    python scripts/check_encoding.py                         # taxonomy only
    python scripts/check_encoding.py --data-dir /path/to/splits
"""
import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

# Mojibake patterns: latin-1 characters read as UTF-8 (e.g. é → Ã©)
_MOJIBAKE = re.compile(r"Ã[©èêëàâùûüîïôç€™‰]|Ã‰|Ã€|â€[™\"œ<>]")
# Question mark replacing an accented char (e.g. "Câble" → "C?ble")
_QMARK = re.compile(r"[a-zA-Z]\?|[a-zA-Z]\?\w|\?\w")


def _check_string(text: str) -> str | None:
    if _QMARK.search(text):
        return "question-mark substitution"
    if _MOJIBAKE.search(text):
        return "mojibake (latin-1 / UTF-8 mismatch)"
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("C") and ch not in ("\n", "\r", "\t"):
            return f"control character U+{ord(ch):04X} ({cat})"
    return None


def check_csv(path: Path) -> list[dict]:
    issues = []
    with open(path, encoding="utf-8") as f:
        for row_num, row in enumerate(csv.DictReader(f), 1):
            for col, val in row.items():
                reason = _check_string(val)
                if reason:
                    issues.append({"file": str(path), "row": row_num, "col": col, "value": val, "reason": reason})
    return issues


def check_jsonl(path: Path, text_fields: tuple[str, ...] = ("designation", "description", "typelabel", "producttype")) -> list[dict]:
    issues = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                issues.append({"file": str(path), "row": line_num, "col": "JSON", "value": line[:80], "reason": str(e)})
                continue
            for field in text_fields:
                val = obj.get(field)
                if not isinstance(val, str):
                    continue
                reason = _check_string(val)
                if reason:
                    issues.append({"file": str(path), "row": line_num, "col": field, "value": val[:120], "reason": reason})
    return issues


def main():
    parser = argparse.ArgumentParser(description="Check encoding integrity of taxonomy and dataset files.")
    parser.add_argument("--taxonomy", default="product_classification_taxonomy.csv")
    parser.add_argument("--data-dir", default=None, help="Directory containing JSONL splits (train/val/test)")
    args = parser.parse_args()

    all_issues: list[dict] = []

    taxonomy_path = Path(args.taxonomy)
    if taxonomy_path.exists():
        print(f"Checking taxonomy: {taxonomy_path}")
        issues = check_csv(taxonomy_path)
        all_issues.extend(issues)
        print(f"  → {len(issues)} issue(s) found")
    else:
        print(f"Taxonomy not found: {taxonomy_path}", file=sys.stderr)

    if args.data_dir:
        data_dir = Path(args.data_dir)
        jsonl_files = sorted(data_dir.glob("*.jsonl"))
        if not jsonl_files:
            print(f"No .jsonl files found in {data_dir}", file=sys.stderr)
        for jsonl_path in jsonl_files:
            print(f"Checking {jsonl_path.name}...")
            issues = check_jsonl(jsonl_path)
            all_issues.extend(issues)
            print(f"  → {len(issues)} issue(s) found")

    if not all_issues:
        print("\nAll files clean — no encoding issues detected.")
        sys.exit(0)

    print(f"\n{len(all_issues)} total issue(s):\n")
    for issue in all_issues[:50]:
        print(f"  [{issue['file']}] row {issue['row']}, col '{issue['col']}': {issue['reason']}")
        print(f"    value: {repr(issue['value'])}")

    if len(all_issues) > 50:
        print(f"  ... and {len(all_issues) - 50} more.")

    sys.exit(1)


if __name__ == "__main__":
    main()
