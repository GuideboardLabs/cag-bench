from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import read_jsonl


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (
                        json.dumps(v, ensure_ascii=False)
                        if isinstance(v, (list, dict))
                        else ''
                        if v is None
                        else v
                    )
                    for k, v in row.items()
                }
            )


def main() -> None:
    p = argparse.ArgumentParser(description='Recover summary.csv from mode raw jsonl files after interrupted runs.')
    p.add_argument('--out-dir', default='results/run')
    p.add_argument('--modes', nargs='+', default=['rag', 'dag', 'cag'])
    p.add_argument('--summary-name', default='summary_recovered.csv')
    p.add_argument('--trial-limit', type=int, default=None, help='Only include trials up to this number')
    p.add_argument(
        '--complete-only',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Keep only trials complete for every mode (disable with --no-complete-only)',
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    run_config = out_dir / 'run_config.json'
    task_count = None
    if run_config.exists():
        task_count = int(json.loads(run_config.read_text(encoding='utf-8')).get('task_count') or 0)

    by_mode_trial: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for mode in args.modes:
        raw_path = out_dir / f'{mode}_raw.jsonl'
        if not raw_path.exists():
            continue
        for row in read_jsonl(raw_path):
            trial = int(row.get('trial') or 0)
            if trial <= 0:
                continue
            if args.trial_limit is not None and trial > args.trial_limit:
                continue
            by_mode_trial[mode][trial].append(row)

    rows: list[dict[str, Any]] = []
    if args.complete_only:
        trial_candidates = Counter()
        for mode in args.modes:
            for trial, trial_rows in by_mode_trial.get(mode, {}).items():
                if task_count is None or len(trial_rows) >= task_count:
                    trial_candidates[trial] += 1
        complete_trials = sorted(t for t, count in trial_candidates.items() if count == len(args.modes))
        for mode in args.modes:
            for trial in complete_trials:
                rows.extend(sorted(by_mode_trial.get(mode, {}).get(trial, []), key=lambda r: int(r.get('task_index') or 0)))
    else:
        for mode in args.modes:
            trials = sorted(by_mode_trial.get(mode, {}).keys())
            for trial in trials:
                rows.extend(sorted(by_mode_trial[mode][trial], key=lambda r: int(r.get('task_index') or 0)))

    if not rows:
        raise SystemExit('No rows matched recovery criteria.')

    summary_path = out_dir / args.summary_name
    write_csv(rows, summary_path)
    print(summary_path)
    print(f'rows={len(rows)}')


if __name__ == '__main__':
    main()
