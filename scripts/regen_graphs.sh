#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <run_dir> [--trial-limit N] [--raw-trials|--no-raw-trials] [--theme light|dark]"
  echo "   or: $0 <folder> <subfolder> [--trial-limit N] [--raw-trials|--no-raw-trials] [--theme light|dark]"
  echo "Examples:"
  echo "  $0 results/qwen3b_v4/20260430_112436"
  echo "  $0 results qwen3b_v4/20260430_112436"
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

OUT_DIR=""
if [[ -d "$1" ]]; then
  OUT_DIR="$1"
  shift
elif [[ $# -ge 2 ]] && [[ -d "$1/$2" ]]; then
  OUT_DIR="$1/$2"
  shift 2
else
  echo "ERROR: could not resolve run directory from args." >&2
  usage
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=.venv/bin/python
fi

"$PYTHON_BIN" - "$OUT_DIR" "$@" <<'PY'
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cag_bench.plot import make_plots
from cag_bench.scoring import BASE_WEIGHTS, composite_score as v4_weighted_score
from cag_bench.eval import composite_score as continuity_score


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _safe_norm_cost(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    x = (value - low) / (high - low)
    return max(0.0, min(1.0, x))


def _read_jsonl_df(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


def _is_probability_series(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors='coerce').dropna()
    if numeric.empty:
        return False
    return float(numeric.max()) <= 1.0


def _resolve_source_dataframe(out_dir: Path) -> tuple[pd.DataFrame, str]:
    raw_path = out_dir / 'raw.jsonl'
    mode_paths = [out_dir / f'{m}_raw.jsonl' for m in ('rag', 'dag', 'cag')]
    summary_path = out_dir / 'summary.csv'

    if raw_path.exists():
        return _read_jsonl_df(raw_path), 'run_raw'

    mode_frames = [df for p in mode_paths for df in [_read_jsonl_df(p)] if not df.empty]
    if mode_frames:
        return pd.concat(mode_frames, ignore_index=True), 'benchmark_raw'

    if summary_path.exists():
        return pd.read_csv(summary_path), 'summary_csv'

    raise SystemExit(f'No usable data found in {out_dir} (expected raw.jsonl, *_raw.jsonl, or summary.csv).')


def _recompute_run_scores(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        'checklist_quality',
        'source_evidence_recall',
        'domain_rule_recall',
        'continuity_recall',
        'token_efficiency',
        'latency_efficiency',
    ]

    def score_row(row):
        metrics = {}
        for c in metric_cols:
            v = _safe_float(row.get(c))
            if v is None:
                return row.get('score')
            metrics[c] = v
        return v4_weighted_score(metrics, weights=BASE_WEIGHTS)

    if all(c in df.columns for c in metric_cols):
        df = df.copy()
        df['score'] = df.apply(score_row, axis=1)
    return df


def _recompute_benchmark_scores(df: pd.DataFrame, run_config: dict) -> pd.DataFrame:
    token_high = float(run_config.get('token_high', 12000))
    seconds_high = float(run_config.get('seconds_high', 120.0))

    def score_row(row):
        task_quality = _safe_float(row.get('task_quality'))
        evidence_recall = _safe_float(row.get('evidence_recall'))
        memory_recall = _safe_float(row.get('memory_recall'))
        prompt_tokens = _safe_float(row.get('prompt_tokens'))
        seconds = _safe_float(row.get('seconds'))
        if None in (task_quality, evidence_recall, memory_recall, prompt_tokens, seconds):
            return row.get('score')
        return continuity_score(
            task_quality=task_quality,
            evidence_recall=evidence_recall,
            memory_recall=memory_recall,
            prompt_tokens=int(prompt_tokens),
            seconds=seconds,
            token_high=token_high,
            seconds_high=seconds_high,
        )

    df = df.copy()
    df['score'] = df.apply(score_row, axis=1)
    return df


def _build_plot_frame(df: pd.DataFrame, run_config: dict) -> pd.DataFrame:
    plot_df = df.copy()

    if 'task_quality' in plot_df.columns and 'checklist_quality' not in plot_df.columns:
        plot_df['checklist_quality'] = pd.to_numeric(plot_df['task_quality'], errors='coerce') * 100.0

    if 'memory_recall' in plot_df.columns and 'continuity_recall' not in plot_df.columns:
        plot_df['continuity_recall'] = pd.to_numeric(plot_df['memory_recall'], errors='coerce') * 100.0

    if 'evidence_recall' in plot_df.columns and _is_probability_series(plot_df['evidence_recall']):
        plot_df['evidence_recall'] = pd.to_numeric(plot_df['evidence_recall'], errors='coerce') * 100.0
    if 'source_evidence_recall' not in plot_df.columns and 'evidence_recall' in plot_df.columns:
        plot_df['source_evidence_recall'] = pd.to_numeric(plot_df['evidence_recall'], errors='coerce')
    if 'domain_rule_recall' not in plot_df.columns and 'evidence_recall' in plot_df.columns:
        plot_df['domain_rule_recall'] = pd.to_numeric(plot_df['evidence_recall'], errors='coerce')
    if 'evidence_recall' not in plot_df.columns and {'source_evidence_recall', 'domain_rule_recall'}.issubset(plot_df.columns):
        plot_df['evidence_recall'] = (
            pd.to_numeric(plot_df['source_evidence_recall'], errors='coerce') +
            pd.to_numeric(plot_df['domain_rule_recall'], errors='coerce')
        ) / 2.0
    if 'source_evidence_recall' in plot_df.columns and _is_probability_series(plot_df['source_evidence_recall']):
        plot_df['source_evidence_recall'] = pd.to_numeric(plot_df['source_evidence_recall'], errors='coerce') * 100.0
    if 'domain_rule_recall' in plot_df.columns and _is_probability_series(plot_df['domain_rule_recall']):
        plot_df['domain_rule_recall'] = pd.to_numeric(plot_df['domain_rule_recall'], errors='coerce') * 100.0
    if 'contradiction_penalty' not in plot_df.columns:
        plot_df['contradiction_penalty'] = 0.0

    if 'seconds' in plot_df.columns and 'latency_seconds' not in plot_df.columns:
        plot_df['latency_seconds'] = pd.to_numeric(plot_df['seconds'], errors='coerce')

    token_low = 500.0
    token_high = float(run_config.get('token_high', 12000.0))
    seconds_low = 1.0
    seconds_high = float(run_config.get('seconds_high', 120.0))

    if 'token_efficiency' not in plot_df.columns and 'prompt_tokens' in plot_df.columns:
        prompt_tokens = pd.to_numeric(plot_df['prompt_tokens'], errors='coerce')
        plot_df['token_efficiency'] = (1.0 - prompt_tokens.apply(lambda v: _safe_norm_cost(v, token_low, token_high))) * 100.0

    if 'latency_efficiency' not in plot_df.columns and 'latency_seconds' in plot_df.columns:
        latency_seconds = pd.to_numeric(plot_df['latency_seconds'], errors='coerce')
        plot_df['latency_efficiency'] = (1.0 - latency_seconds.apply(lambda v: _safe_norm_cost(v, seconds_low, seconds_high))) * 100.0

    required = ['mode', 'trial', 'task_index', 'score', 'checklist_quality', 'source_evidence_recall', 'domain_rule_recall', 'evidence_recall', 'continuity_recall', 'token_efficiency', 'latency_efficiency', 'contradiction_penalty', 'prompt_tokens', 'latency_seconds']
    missing = [c for c in required if c not in plot_df.columns]
    if missing:
        raise SystemExit(f'Cannot plot; missing required columns after normalization: {missing}')

    return plot_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir')
    parser.add_argument('--trial-limit', type=int, default=None)
    parser.add_argument('--summary-name', default='summary_regen.csv')
    parser.add_argument('--plot-summary-name', default='summary_plot_ready.csv')
    parser.add_argument('--raw-trials', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--theme', choices=['light', 'dark'], default='light')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        raise SystemExit(f'Output directory not found: {out_dir}')

    run_config_path = out_dir / 'run_config.json'
    run_config = {}
    if run_config_path.exists():
        run_config = json.loads(run_config_path.read_text(encoding='utf-8'))

    df, source = _resolve_source_dataframe(out_dir)
    if df.empty:
        raise SystemExit(f'No rows found to process in {out_dir}')

    if args.trial_limit is not None and 'trial' in df.columns:
        df = df[pd.to_numeric(df['trial'], errors='coerce') <= args.trial_limit]

    if df.empty:
        raise SystemExit('No rows left after filters.')

    if {'task_quality', 'memory_recall'}.issubset(df.columns):
        regen_df = _recompute_benchmark_scores(df, run_config)
        schema = 'benchmark'
    elif {'checklist_quality', 'continuity_recall'}.issubset(df.columns):
        regen_df = _recompute_run_scores(df)
        schema = 'run'
    else:
        regen_df = df.copy()
        schema = 'unknown'

    regen_summary_path = out_dir / args.summary_name
    regen_df.to_csv(regen_summary_path, index=False)

    plot_df = _build_plot_frame(regen_df, run_config)
    plot_summary_path = out_dir / args.plot_summary_name
    plot_df.to_csv(plot_summary_path, index=False)

    make_plots(plot_summary_path, out_dir, raw_trials=args.raw_trials, theme=args.theme)

    weight_note_path = out_dir / 'regen_weights_used.json'
    weight_note = {
        'source': source,
        'detected_schema': schema,
        'theme': args.theme,
        'v4_weights': BASE_WEIGHTS,
        'continuity_formula': 'cag_bench.eval.composite_score',
    }
    weight_note_path.write_text(json.dumps(weight_note, indent=2), encoding='utf-8')

    print(f'Regenerated summary: {regen_summary_path}')
    print(f'Plot input summary: {plot_summary_path}')
    print(f'Wrote plots in: {out_dir}')
    print(f'Wrote trend summary: {out_dir / "SUMMARY.md"}')
    print(f'Weights metadata: {weight_note_path}')


if __name__ == '__main__':
    main()
PY

echo "Graph regeneration complete: $OUT_DIR"
