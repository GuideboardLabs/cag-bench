#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=.venv/bin/python
fi
"$PYTHON_BIN" -m cag_bench.run --dry-run --out results/dry_run --max-tasks 30 --repeat-runs 3 --raw-trials "$@"
