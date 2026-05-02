#!/usr/bin/env bash
# Build a tarball of raw outputs (raw.jsonl + regen artifacts) for distribution as a GitHub Release artifact.
# These files are .gitignored to keep the repo lean; this script bundles them for upload.
#
# Usage: ./scripts/build_release_artifact.sh [output_path]
#   default output: cag_bench_raw_outputs_<git_short_sha>.tar.gz
#
# Upload the resulting tarball to a GitHub Release. Reference it from the paper.

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d results ]]; then
  echo "ERROR: no results/ directory found. Run benchmarks first." >&2
  exit 1
fi

SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "nosha")
OUT="${1:-cag_bench_raw_outputs_${SHA}.tar.gz}"

# Collect the files we deliberately exclude from git
mapfile -t FILES < <(find results -type f \
  \( -name 'raw.jsonl' -o -name 'summary_regen.csv' -o -name 'summary_plot_ready.csv' \) \
  -not -path 'results/dry_run*' \
  -not -path 'results/smoke*' \
  -not -path 'results/script_smoke*' \
  -not -path 'results/verify_*')

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "ERROR: no raw.jsonl or regen artifacts found under results/." >&2
  exit 1
fi

echo "Bundling ${#FILES[@]} files into $OUT"
tar -czf "$OUT" "${FILES[@]}"
ls -lh "$OUT"
echo
echo "Upload to: https://github.com/GuideboardLabs/cag-bench/releases"
echo "Tag suggestion: paper-v1-raw-outputs"
