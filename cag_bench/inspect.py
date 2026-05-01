from __future__ import annotations

import argparse
import json
from pathlib import Path

MODE_ORDER = ['rag', 'dag', 'cag', 'cag_scoped', 'cag_oracle_memory']


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except Exception:
                continue
    return rows


def _fmt(value) -> str:
    if value is None:
        return 'null'
    if isinstance(value, float):
        return f'{value:.2f}'
    return str(value)


def _mode_key(mode: str) -> tuple[int, str]:
    mode = (mode or '').strip().lower()
    if mode in MODE_ORDER:
        return (MODE_ORDER.index(mode), mode)
    return (999, mode)


def _concept_hit_summary(row: dict) -> str:
    hits = (row.get('concept_hits') or {}).get('continuity_recall') or []
    if not isinstance(hits, list) or not hits:
        return 'none'
    parts = []
    for hit in hits:
        concept = str(hit.get('concept', '')).strip() or 'unnamed'
        ok = bool(hit.get('hit'))
        matched = hit.get('matched_term')
        parts.append(f"{concept}={'hit' if ok else 'miss'}({matched or '-'})")
    return '; '.join(parts)


def _answer_preview(row: dict, max_chars: int = 600) -> str:
    answer = str(row.get('answer', '') or '').strip()
    if len(answer) <= max_chars:
        return answer
    return answer[:max_chars].rstrip() + ' ...[truncated]'


def main() -> None:
    p = argparse.ArgumentParser(description='Inspect one or more tasks from raw benchmark jsonl output.')
    p.add_argument('--raw', required=True, help='Path to raw.jsonl file')
    p.add_argument('--task', action='append', required=True, help='Task id to inspect (repeatable)')
    args = p.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.exists():
        raise SystemExit(f'Raw file not found: {raw_path}')

    wanted = {str(t).strip().upper() for t in args.task if str(t).strip()}
    rows = _read_jsonl(raw_path)
    filtered = [r for r in rows if str(r.get('task_id', '')).strip().upper() in wanted]
    if not filtered:
        raise SystemExit(f'No rows found for task(s): {sorted(wanted)}')

    by_task: dict[str, list[dict]] = {}
    for row in filtered:
        task_id = str(row.get('task_id', '')).strip().upper() or 'UNKNOWN'
        by_task.setdefault(task_id, []).append(row)

    for task_id in sorted(by_task.keys()):
        print(f'===== {task_id} =====')
        task_rows = sorted(by_task[task_id], key=lambda r: (_mode_key(str(r.get('mode', ''))), int(r.get('trial') or 0)))
        for row in task_rows:
            mode = str(row.get('mode', '')).strip()
            trial = row.get('trial')
            print(f'[{mode}] trial={trial}')
            print(f"  score={_fmt(row.get('score'))}")
            print(f"  continuity_recall={_fmt(row.get('continuity_recall'))}")
            print(f"  memory_recall={_fmt(row.get('memory_recall'))}")
            print(f"  memory_precision={_fmt(row.get('memory_precision'))}")
            print(f"  memory_tokens_used={_fmt(row.get('memory_tokens_used'))}")
            print(f"  selected_memory_ids={row.get('selected_memory_ids') or []}")
            print(f"  concept_hits={_concept_hit_summary(row)}")
            print(f"  contradiction_hits={len(row.get('contradiction_hits') or [])}")
            print('  answer_preview=')
            print(f"    {_answer_preview(row)}")
        print('')


if __name__ == '__main__':
    main()
