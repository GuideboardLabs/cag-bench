#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=.venv/bin/python
fi
"$PYTHON_BIN" -m cag_bench.benchmark --embedder tfidf --dry-run --out-dir results/smoke --max-tasks 3 --repeat-runs 2
"$PYTHON_BIN" -m cag_bench.plot --summary results/smoke/summary.csv --out-dir results/smoke --raw-trials
