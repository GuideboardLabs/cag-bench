from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CURVE_METRICS = [
    ("score", "Composite score"),
    ("checklist_quality", "Checklist quality"),
    ("source_evidence_recall", "Source evidence recall"),
    ("domain_rule_recall", "Domain rule recall"),
    ("evidence_recall", "Evidence recall"),
    ("continuity_recall", "Continuity recall"),
    ("token_efficiency", "Token efficiency"),
    ("latency_efficiency", "Latency efficiency"),
    ("contradiction_penalty", "Contradiction penalty (lower is better)"),
    ("memory_precision", "Memory precision"),
    ("memory_recall", "Memory recall"),
    ("continuity_per_memory_token", "Continuity per memory token"),
    ("memory_tokens_used", "Memory tokens used"),
    ("brief_tokens_used", "Brief tokens used"),
    ("source_tokens_used", "Source tokens used"),
    ("contradictions_before_revision", "Contradictions before revision"),
    ("contradictions_after_revision", "Contradictions after revision"),
]

GRID_METRICS = [
    ("score", "Composite score"),
    ("checklist_quality", "Checklist quality"),
    ("source_evidence_recall", "Source evidence recall"),
    ("domain_rule_recall", "Domain rule recall"),
    ("continuity_recall", "Continuity recall"),
    ("token_efficiency", "Token efficiency"),
    ("latency_efficiency", "Latency efficiency"),
    ("contradiction_penalty", "Contradiction penalty (lower is better)"),
    ("memory_precision", "Memory precision"),
    ("memory_recall", "Memory recall"),
    ("continuity_per_memory_token", "Continuity per memory token"),
    ("memory_tokens_used", "Memory tokens used"),
    ("brief_tokens_used", "Brief tokens used"),
    ("source_tokens_used", "Source tokens used"),
    ("contradictions_before_revision", "Contradictions before revision"),
    ("contradictions_after_revision", "Contradictions after revision"),
]

MODE_LABELS = {
    "rag": "RAG: fresh retrieval",
    "dag": "DAG: fixed workflow",
    "cag_naive": "CAG naive",
    "cag_scoped": "CAG scoped",
    "cag_briefed": "CAG briefed",
    "cag_full": "CAG full",
}
MODE_ORDER = ["rag", "dag", "cag_naive", "cag_scoped", "cag_briefed", "cag_full"]
MODE_COLORS = {
    "rag": "#0072B2",
    "dag": "#D55E00",
    "cag_naive": "#6D28D9",
    "cag_scoped": "#0F766E",
    "cag_briefed": "#2563EB",
    "cag_full": "#059669",
}
THEMES = {
    "light": {
        "facecolor": "#FFFFFF",
        "axes_facecolor": "#FFFFFF",
        "text": "#111827",
        "grid": "#D1D5DB",
    },
    "dark": {
        "facecolor": "#1F232A",
        "axes_facecolor": "#242A33",
        "text": "#E5E7EB",
        "grid": "#4B5563",
    },
}


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _detected_modes(df: pd.DataFrame) -> list[str]:
    found = sorted(df["mode"].dropna().astype(str).unique().tolist(), key=lambda m: MODE_ORDER.index(m) if m in MODE_ORDER else 99)
    return found


def _mode_label(mode: str) -> str:
    return MODE_LABELS.get(mode, mode)


def _mode_color(mode: str) -> str:
    return MODE_COLORS.get(mode, "#374151")


def _plot_metric(df, modes, metric, title, out_path, raw_trials=False, theme="light"):
    palette = THEMES[theme]
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=palette["facecolor"])
    ax.set_facecolor(palette["axes_facecolor"])

    for mode in modes:
        color = _mode_color(mode)
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        if raw_trials and "trial" in sub.columns and sub["trial"].nunique() > 1:
            for _, tdf in sub.groupby("trial"):
                ax.plot(tdf["task_index"], tdf[metric], alpha=0.15, linewidth=1.0, color=color)
        grouped = sub.groupby("task_index")[metric]
        mean = grouped.mean()
        std = grouped.std().fillna(0)
        x = mean.index.to_numpy()
        y = mean.to_numpy()
        s = std.to_numpy()
        ax.plot(x, y, marker="o", linewidth=2.3, label=_mode_label(mode), color=color)
        if "trial" in sub.columns and sub["trial"].nunique() > 1:
            ax.fill_between(x, y - s, y + s, alpha=0.12, color=color)

    ax.set_title(title, color=palette["text"])
    ax.set_xlabel("Task index", color=palette["text"])
    ax.set_ylabel(title, color=palette["text"])
    ax.set_xticks(sorted(pd.to_numeric(df["task_index"], errors="coerce").dropna().astype(int).unique()))

    if metric in {"contradiction_penalty", "memory_tokens_used", "brief_tokens_used", "source_tokens_used", "continuity_per_memory_token", "contradictions_before_revision", "contradictions_after_revision"}:
        ymax = float(pd.to_numeric(df[metric], errors="coerce").fillna(0).max())
        ax.set_ylim(0, max(1.0, ymax * 1.15))
    else:
        ax.set_ylim(0, 100)

    ax.grid(True, alpha=0.3, color=palette["grid"])
    ax.legend(facecolor=palette["axes_facecolor"], edgecolor=palette["grid"], labelcolor=palette["text"])
    ax.tick_params(colors=palette["text"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=palette["facecolor"])
    plt.close(fig)


def _write_summary_md(df: pd.DataFrame, out_dir: Path, summary_csv: Path) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    modes = _detected_modes(df)
    lines = [
        "# Run Summary",
        "",
        f"- Generated: {ts}",
        f"- Source summary: `{summary_csv}`",
        f"- Modes detected: {', '.join(modes)}",
        f"- Total rows analyzed: {len(df)}",
        "",
    ]

    key_metrics = [
        "score",
        "continuity_recall",
        "memory_precision",
        "memory_recall",
        "continuity_per_memory_token",
        "token_efficiency",
        "contradiction_penalty",
    ]

    for metric in key_metrics:
        if metric not in df.columns:
            continue
        lines.append(f"## {metric}")
        for mode in modes:
            sub = df[df["mode"] == mode]
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
            if vals.empty:
                continue
            lines.append(f"- {_mode_label(mode)}: mean {vals.mean():.3f} | std {vals.std():.3f} | n={len(vals)}")
        lines.append("")

    # Keep a short task-30 section because late-stage behavior is central.
    if "task_index" in df.columns and "score" in df.columns:
        t30 = df[pd.to_numeric(df["task_index"], errors="coerce") == 30]
        if not t30.empty:
            lines.append("## Task 30 Snapshot")
            for mode in modes:
                sub = t30[t30["mode"] == mode]
                vals = pd.to_numeric(sub["score"], errors="coerce").dropna()
                if vals.empty:
                    continue
                lines.append(f"- {_mode_label(mode)} score mean at task 30: {vals.mean():.3f}")
            lines.append("")

    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def make_plots(summary_csv, out_dir, raw_trials=False, theme="light"):
    if theme not in THEMES:
        raise ValueError(f"Unknown theme: {theme}. Expected one of: {list(THEMES)}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = Path(summary_csv)
    df = pd.read_csv(summary_csv)
    if df.empty:
        raise ValueError("Summary CSV is empty.")

    if "task_index" in df.columns:
        df["task_index"] = pd.to_numeric(df["task_index"], errors="coerce")

    modes = _detected_modes(df)
    if not modes:
        raise ValueError("No modes detected in summary CSV.")

    for metric, title in CURVE_METRICS:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
            _plot_metric(df, modes, metric, title, out_dir / f"{metric}_curve.png", raw_trials, theme)

    fig_rows = 4
    fig_cols = 4
    palette = THEMES[theme]
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(20, 18), facecolor=palette["facecolor"])
    axes_flat = axes.flatten()

    for ax, (metric, title) in zip(axes_flat, GRID_METRICS):
        ax.set_facecolor(palette["axes_facecolor"])
        if metric not in df.columns:
            ax.set_title(f"{title} (missing)", color=palette["text"])
            ax.axis("off")
            continue

        for mode in modes:
            sub = df[df["mode"] == mode]
            if sub.empty:
                continue
            mean = sub.groupby("task_index")[metric].mean()
            ax.plot(mean.index, mean.values, marker="o", linewidth=2, label=_mode_label(mode), color=_mode_color(mode))

        ax.set_title(title, color=palette["text"])
        if metric in {"contradiction_penalty", "memory_tokens_used", "brief_tokens_used", "source_tokens_used", "continuity_per_memory_token", "contradictions_before_revision", "contradictions_after_revision"}:
            ymax = float(pd.to_numeric(df[metric], errors="coerce").fillna(0).max())
            ax.set_ylim(0, max(1.0, ymax * 1.15))
        else:
            ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.25, color=palette["grid"])
        ax.tick_params(colors=palette["text"])

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(handles)), facecolor=palette["axes_facecolor"], edgecolor=palette["grid"], labelcolor=palette["text"])

    fig.suptitle("Benchmark curves (suite-aware mode comparison)", fontsize=16, color=palette["text"])
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_dir / "all_metrics_grid.png", dpi=200, facecolor=palette["facecolor"])
    plt.close(fig)

    agg_cols = [m for m, _ in CURVE_METRICS if m in df.columns] + [c for c in ["prompt_tokens", "latency_seconds"] if c in df.columns]
    agg = df.groupby(["mode", "task_index"])[agg_cols].agg(["mean", "std"]).reset_index()
    agg.to_csv(out_dir / "aggregated_metrics.csv", index=False)

    _write_summary_md(df, out_dir, summary_csv)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--raw-trials", action="store_true")
    p.add_argument("--theme", choices=["light", "dark"], default="light")
    args = p.parse_args()
    make_plots(args.summary, args.out_dir, args.raw_trials, args.theme)


if __name__ == "__main__":
    main()
