#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:-qwen2.5-coder:7b}"
OUT="${2:-results/full_matrix}"
shift || true
shift || true
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=.venv/bin/python
fi
"$PYTHON_BIN" -m cag_bench.run \
  --model "$MODEL" \
  --out "$OUT" \
  --max-tasks 30 \
  --repeat-runs 3 \
  --suite all \
  "$@"
