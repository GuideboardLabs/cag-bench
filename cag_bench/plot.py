
import argparse
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

CURVE_METRICS = [
    ('score', 'Composite score'),
    ('checklist_quality', 'Checklist quality'),
    ('source_evidence_recall', 'Source evidence recall'),
    ('domain_rule_recall', 'Domain rule recall'),
    ('evidence_recall', 'Evidence recall'),
    ('continuity_recall', 'Continuity recall'),
    ('token_efficiency', 'Token efficiency'),
    ('latency_efficiency', 'Latency efficiency'),
    ('contradiction_penalty', 'Contradiction penalty (lower is better)'),
]
GRID_METRICS = [
    ('score', 'Composite score'),
    ('checklist_quality', 'Checklist quality'),
    ('source_evidence_recall', 'Source evidence recall'),
    ('domain_rule_recall', 'Domain rule recall'),
    ('continuity_recall', 'Continuity recall'),
    ('token_efficiency', 'Token efficiency'),
    ('latency_efficiency', 'Latency efficiency'),
    ('contradiction_penalty', 'Contradiction penalty (lower is better)'),
]
LABELS={'rag':'RAG: fresh retrieval','dag':'DAG: fixed workflow','cag':'CAG: persistent context'}
MODE_COLORS={'rag':'#0072B2','dag':'#D55E00','cag':'#009E73'}
THEMES={
    'light':{
        'facecolor':'#FFFFFF',
        'axes_facecolor':'#FFFFFF',
        'text':'#111827',
        'grid':'#D1D5DB',
    },
    'dark':{
        'facecolor':'#1F232A',
        'axes_facecolor':'#242A33',
        'text':'#E5E7EB',
        'grid':'#4B5563',
    },
}

def _trend_word(delta: float, eps: float = 0.25) -> str:
    if delta > eps:
        return 'increase'
    if delta < -eps:
        return 'decline'
    return 'flat trend'

def _pct_change(start: float, end: float) -> float | None:
    if abs(start) < 1e-9:
        return None
    return ((end - start) / abs(start)) * 100.0

def _fmt_num(value: float) -> str:
    return f'{value:.2f}'

def _fmt_pct(value: float | None) -> str:
    if value is None:
        return 'n/a (start value is 0)'
    sign = '+' if value >= 0 else ''
    return f'{sign}{value:.2f}%'

def _pct_vs(reference: float, value: float) -> float | None:
    if abs(reference) < 1e-9:
        return None
    return ((value - reference) / abs(reference)) * 100.0

def _safe_float(value) -> float | None:
    try:
        return float(value)
    except Exception:
        return None

def _load_raw_rows(out_dir: Path) -> list[dict]:
    paths = [out_dir / 'raw.jsonl']
    if not paths[0].exists():
        paths = [out_dir / 'rag_raw.jsonl', out_dir / 'dag_raw.jsonl', out_dir / 'cag_raw.jsonl']
    rows: list[dict] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows

def _task30_rows(raw_rows: list[dict]) -> list[dict]:
    out = []
    for row in raw_rows:
        task_index = row.get('task_index')
        task_id = str(row.get('task_id', '')).strip().upper()
        if str(task_index) == '30' or task_id == 'T30':
            out.append(row)
    return out

def _append_final_boss_sections(lines: list[str], out_dir: Path) -> None:
    raw_rows = _load_raw_rows(out_dir)
    if not raw_rows:
        return
    rows_30 = _task30_rows(raw_rows)
    if not rows_30:
        return

    scored_rows = []
    for row in rows_30:
        score = _safe_float(row.get('score'))
        if score is None:
            continue
        scored_rows.append((score, row))
    if not scored_rows:
        return

    scored_rows.sort(key=lambda x: x[0], reverse=True)
    best_score, best_row = scored_rows[0]
    answer = str(best_row.get('answer', '') or '').strip()
    if len(answer) > 4500:
        answer = answer[:4500].rstrip() + '\n\n[truncated]'

    lines.append('## Final Boss: Best Rated Task 30 Answer')
    lines.append(f"- Best score: {_fmt_num(best_score)}")
    lines.append(f"- Mode: {str(best_row.get('mode', '')).upper()}")
    lines.append(f"- Trial: {best_row.get('trial', 'n/a')}")
    lines.append('- Answer:')
    lines.append('```text')
    lines.append(answer if answer else '[empty answer]')
    lines.append('```')
    lines.append('')

    lines.append('## Final Boss: Task 30 Comparison (Across Runs)')
    mode_order = {'rag': 0, 'dag': 1, 'cag': 2}
    by_mode: dict[str, list[dict]] = {}
    for row in rows_30:
        mode = str(row.get('mode', '')).lower().strip()
        if not mode:
            continue
        by_mode.setdefault(mode, []).append(row)

    for mode in sorted(by_mode.keys(), key=lambda m: mode_order.get(m, 99)):
        mode_rows = by_mode[mode]
        score_vals = [_safe_float(r.get('score')) for r in mode_rows]
        score_vals = [v for v in score_vals if v is not None]
        continuity_vals = [_safe_float(r.get('continuity_recall')) for r in mode_rows]
        continuity_vals = [v for v in continuity_vals if v is not None]
        checklist_vals = [_safe_float(r.get('checklist_quality')) for r in mode_rows]
        checklist_vals = [v for v in checklist_vals if v is not None]
        source_vals = [_safe_float(r.get('source_evidence_recall')) for r in mode_rows]
        source_vals = [v for v in source_vals if v is not None]
        domain_vals = [_safe_float(r.get('domain_rule_recall')) for r in mode_rows]
        domain_vals = [v for v in domain_vals if v is not None]
        contradiction_vals = [_safe_float(r.get('contradiction_penalty')) for r in mode_rows]
        contradiction_vals = [v for v in contradiction_vals if v is not None]
        if not score_vals:
            continue
        mode_label = mode.upper()
        mean = lambda values: (sum(values) / len(values)) if values else 0.0
        lines.append(
            f"- {mode_label}: score mean {_fmt_num(mean(score_vals))} (n={len(score_vals)}), "
            f"continuity mean {_fmt_num(mean(continuity_vals))}, checklist mean {_fmt_num(mean(checklist_vals))}, "
            f"source evidence mean {_fmt_num(mean(source_vals))}, domain rule mean {_fmt_num(mean(domain_vals))}, "
            f"contradiction penalty mean {_fmt_num(mean(contradiction_vals))}."
        )
    lines.append('')

def _write_summary_md(df: pd.DataFrame, out_dir: Path, summary_csv: Path) -> None:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    task_index = pd.to_numeric(df['task_index'], errors='coerce')
    valid_tasks = sorted(task_index.dropna().astype(int).unique().tolist())
    if not valid_tasks:
        return
    task_start = valid_tasks[0]
    task_end = valid_tasks[-1]

    lines = []
    lines.append('# Run Summary')
    lines.append('')
    lines.append(f'- Generated: {ts}')
    lines.append(f'- Source summary: `{summary_csv}`')
    lines.append(f'- Task range: {task_start} to {task_end}')
    lines.append(f'- Modes detected: {", ".join(sorted(df["mode"].dropna().astype(str).unique().tolist()))}')
    if 'trial' in df.columns:
        trial_vals = pd.to_numeric(df['trial'], errors='coerce').dropna()
        if not trial_vals.empty:
            lines.append(f'- Trials detected: {int(trial_vals.nunique())}')
    lines.append(f'- Total rows analyzed: {len(df)}')
    lines.append('')
    lines.append('Primary comparison uses overall means across all rows (all tasks and all trials), not a single last run.')
    lines.append('')

    for metric, title in CURVE_METRICS:
        lines.append(f'## {title}')
        metric_rows = []
        mode_means = {}
        for mode in ['rag', 'dag', 'cag']:
            sub = df[df['mode'] == mode]
            if sub.empty or metric not in sub.columns:
                continue
            metric_num = pd.to_numeric(sub[metric], errors='coerce').dropna()
            if metric_num.empty:
                continue
            mode_means[mode] = float(metric_num.mean())

        if mode_means:
            lines.append('- Overall mean across all tasks/trials:')
            for mode in ['rag', 'dag', 'cag']:
                if mode not in mode_means:
                    continue
                mode_label = LABELS.get(mode, mode).split(':', 1)[0]
                mean_val = mode_means[mode]
                comps = []
                for other in ['rag', 'dag', 'cag']:
                    if other == mode or other not in mode_means:
                        continue
                    other_label = LABELS.get(other, other).split(':', 1)[0]
                    comps.append(f'vs {other_label} {_fmt_pct(_pct_vs(mode_means[other], mean_val))}')
                comp_text = '; '.join(comps) if comps else 'no comparisons available'
                lines.append(f'  - {mode_label}: {_fmt_num(mean_val)} ({comp_text}).')

            reverse = metric != 'contradiction_penalty'
            ranked_means = sorted(mode_means.items(), key=lambda x: x[1], reverse=reverse)
            ranking = ', '.join([f'{LABELS.get(m, m).split(":", 1)[0]} {_fmt_num(v)}' for m, v in ranked_means])
            lines.append(f'- Overall ranking by mean: {ranking}.')
            lines.append('')

        lines.append('- Task progression trend (first task mean to last task mean):')
        for mode in ['rag', 'dag', 'cag']:
            sub = df[df['mode'] == mode]
            if sub.empty or metric not in sub.columns:
                continue
            mode_series = (
                sub.assign(task_index_num=pd.to_numeric(sub['task_index'], errors='coerce'), metric_num=pd.to_numeric(sub[metric], errors='coerce'))
                .dropna(subset=['task_index_num', 'metric_num'])
                .groupby('task_index_num')['metric_num']
                .mean()
                .sort_index()
            )
            if mode_series.empty:
                continue
            start_val = float(mode_series.iloc[0])
            end_val = float(mode_series.iloc[-1])
            delta = end_val - start_val
            pct = _pct_change(start_val, end_val)
            trend = _trend_word(delta)
            mode_label = LABELS.get(mode, mode).split(':', 1)[0]
            sign = '+' if delta >= 0 else ''
            lines.append(
                f'  - {mode_label}: started at {_fmt_num(start_val)}, ended at {_fmt_num(end_val)}, '
                f'change {sign}{_fmt_num(delta)} points ({_fmt_pct(pct)}), overall {trend}.'
            )
            metric_rows.append((mode, end_val))

        if metric_rows:
            reverse = metric != 'contradiction_penalty'
            ranked = sorted(metric_rows, key=lambda x: x[1], reverse=reverse)
            ranking = ', '.join([f'{LABELS.get(m, m).split(":", 1)[0]} {_fmt_num(v)}' for m, v in ranked])
            lines.append(f'- Final-task ranking (mean): {ranking}.')
        lines.append('')

    _append_final_boss_sections(lines, out_dir)

    (out_dir / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')

def _plot_metric(df, metric, title, out_path, raw_trials=False, theme='light'):
    palette = THEMES[theme]
    task_count = len(sorted(df['task_index'].dropna().unique()))
    fig, ax = plt.subplots(figsize=(12,7), facecolor=palette['facecolor'])
    ax.set_facecolor(palette['axes_facecolor'])
    for mode in ['rag','dag','cag']:
        color = MODE_COLORS[mode]
        sub=df[df['mode']==mode]
        if raw_trials and sub['trial'].nunique()>1:
            for trial,tdf in sub.groupby('trial'):
                ax.plot(tdf['task_index'], tdf[metric], alpha=0.15, linewidth=1, color=color)
        grouped=sub.groupby('task_index')[metric]
        mean=grouped.mean(); std=grouped.std().fillna(0); x=mean.index.to_numpy(); y=mean.to_numpy(); s=std.to_numpy()
        ax.plot(x,y,marker='o',linewidth=2.5,label=LABELS[mode], color=color)
        if sub['trial'].nunique()>1: ax.fill_between(x,y-s,y+s,alpha=0.12, color=color)
    ax.set_title(f'RAG vs DAG vs CAG over {task_count} iterative tasks ({title})', color=palette['text'])
    ax.set_xlabel('Sequential related task in same project', color=palette['text']); ax.set_ylabel(f'{title} (mean across trials when repeated)', color=palette['text'])
    ax.set_xticks(sorted(df['task_index'].unique()))
    if metric == 'contradiction_penalty':
        ymax = max(10.0, float(pd.to_numeric(df[metric], errors='coerce').fillna(0).max()) * 1.15)
        ax.set_ylim(0, ymax)
    else:
        ax.set_ylim(0,100)
    ax.grid(True, alpha=0.3, color=palette['grid']); ax.legend(facecolor=palette['axes_facecolor'], edgecolor=palette['grid'], labelcolor=palette['text'])
    ax.tick_params(colors=palette['text'])
    fig.tight_layout(); fig.savefig(out_path,dpi=200, facecolor=palette['facecolor']); plt.close(fig)

def make_plots(summary_csv, out_dir, raw_trials=False, theme='light'):
    if theme not in THEMES:
        raise ValueError(f'Unknown theme: {theme}. Expected one of: {list(THEMES)}')
    palette = THEMES[theme]
    out_dir=Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True); summary_csv=Path(summary_csv); df=pd.read_csv(summary_csv)
    for metric,title in CURVE_METRICS:
        if metric in df.columns:
            _plot_metric(df, metric, title, out_dir/f'{metric}_curve.png', raw_trials, theme)
    fig,axes=plt.subplots(4,2,figsize=(15,18), facecolor=palette['facecolor']); axes=axes.flatten()
    for ax,(metric,title) in zip(axes,GRID_METRICS):
        ax.set_facecolor(palette['axes_facecolor'])
        if metric not in df.columns:
            ax.set_title(f'{title} (missing)', color=palette['text'])
            ax.axis('off')
            continue
        for mode in ['rag','dag','cag']:
            color = MODE_COLORS[mode]
            sub=df[df['mode']==mode]; mean=sub.groupby('task_index')[metric].mean(); ax.plot(mean.index, mean.values, marker='o', linewidth=2, label=LABELS[mode], color=color)
        ax.set_title(title, color=palette['text'])
        if metric == 'contradiction_penalty':
            ymax = max(10.0, float(pd.to_numeric(df[metric], errors='coerce').fillna(0).max()) * 1.15)
            ax.set_ylim(0, ymax)
        else:
            ax.set_ylim(0,100)
        ax.set_xticks(sorted(df['task_index'].unique())); ax.grid(True, alpha=0.25, color=palette['grid']); ax.tick_params(colors=palette['text'])
    handles,labels=axes[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='lower center',ncol=3, facecolor=palette['axes_facecolor'], edgecolor=palette['grid'], labelcolor=palette['text']); fig.suptitle('RAG vs DAG vs CAG: composite and weighted score components',fontsize=16, color=palette['text']); fig.tight_layout(rect=[0,0.04,1,0.97]); fig.savefig(out_dir/'all_metrics_grid.png',dpi=200, facecolor=palette['facecolor']); plt.close(fig)
    agg_cols = [m for m,_ in CURVE_METRICS if m in df.columns] + ['prompt_tokens','latency_seconds']
    agg=df.groupby(['mode','task_index'])[agg_cols].agg(['mean','std']).reset_index(); agg.to_csv(out_dir/'aggregated_metrics.csv', index=False)
    _write_summary_md(df, out_dir, summary_csv)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--summary',required=True); p.add_argument('--out-dir',required=True); p.add_argument('--raw-trials',action='store_true'); p.add_argument('--theme',choices=['light','dark'],default='light'); args=p.parse_args(); make_plots(args.summary,args.out_dir,args.raw_trials,args.theme)
if __name__=='__main__': main()
