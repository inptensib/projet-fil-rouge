#!/usr/bin/env bash
# End-to-end run of the LLM-native synthetic-data pipeline.
#
#   export ANTHROPIC_API_KEY=sk-ant-...
#   export RAKUTEN_DS=/abs/path/to/datasets_finaux   # holds the 3 splits
#   export RAKUTEN_OUT=./out
#   bash run_all.sh
#
# Steps 00/03/04/05 are pure Python. Steps 01/02 call the LLM (cached on disk,
# so re-running is free and resumable).
set -euo pipefail
cd "$(dirname "$0")"

python3 00_build_reservoirs.py   # scan splits -> leak sets, counts, reservoirs
python3 01_llm_keywords.py       # LLM: per-class keyword vocab
python3 02_llm_templates.py      # LLM: per-class designation/description templates
python3 03_fill_templates.py     # Python: fill templates -> SYNTH-GEN / SYNTH-ORPH
python3 04_oversample.py         # Python: duplication  -> SYNTH-OVS  / SYNTH-OVL
python3 05_assemble.py           # Python: compose + validate -> synthetic.jsonl

echo
echo "Done. Final artifact: \$RAKUTEN_OUT/synthetic.jsonl"
echo "Append it to train_split.jsonl (val/test stay frozen)."
