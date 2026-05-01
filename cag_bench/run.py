from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

from .cag_engine import (
    CAG_VARIANTS,
    CAGMemoryEngine,
    expected_prior_concept_groups,
    expected_prior_concepts,
    memory_precision_recall,
)
from .ollama_client import OllamaClient, approx_tokens
from .prompts import dag_prompt, dry_answer, rag_prompt
from .retrieval import KeywordRetriever, SourceChunk
from .scoring import BASE_WEIGHTS, composite_score, metric_scores
from .utils import append_jsonl, load_jsonl

SUITE_MODES = {
    "baselines": ["rag", "dag", "cag_full"],
    "cag-ablation": ["cag_naive", "cag_scoped", "cag_briefed", "cag_full"],
    "all": ["rag", "dag", "cag_naive", "cag_scoped", "cag_briefed", "cag_full"],
}
VALID_MODES = {"rag", "dag", "cag_naive", "cag_scoped", "cag_briefed", "cag_full", "cag"}


def prompt_text(messages: List[Dict[str, str]]) -> str:
    return "\n\n".join([f"{m['role']}: {m['content']}" for m in messages])


def _extract_json_object(content: str) -> dict:
    text = (content or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return {}


def _clamp_score(value):
    try:
        num = float(value)
    except Exception:
        return None
    return max(0.0, min(100.0, num))


def _parse_modes(values: List[str] | str | None) -> List[str]:
    if not values:
        return []
    parts: List[str] = []
    if isinstance(values, str):
        parts = [values]
    else:
        parts = list(values)
    out: List[str] = []
    for part in parts:
        for sub in str(part).split(","):
            mode = sub.strip().lower()
            if not mode:
                continue
            if mode == "cag":
                mode = "cag_full"
            if mode not in VALID_MODES:
                raise ValueError(f"Unknown mode: {mode}")
            if mode not in out:
                out.append(mode)
    return out


def _resolve_modes(suite: str, modes_raw: List[str] | str | None) -> List[str]:
    override = _parse_modes(modes_raw)
    if override:
        return override
    return list(SUITE_MODES[suite])


def _source_dicts(chunks: List[SourceChunk]) -> List[Dict[str, Any]]:
    return [{"source_id": c.source_id, "title": c.title, "text": c.text} for c in chunks]


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _judge_answer(client, task, answer, judge_model, num_ctx):
    schema = '{"judge_continuity_score":0-100,"judge_actionability_score":0-100,"judge_contradiction_score":0-100,"judge_notes":"brief explanation"}'
    judge_messages = [
        {
            "role": "system",
            "content": (
                "You are a strict benchmark judge. Return JSON only. "
                "Do not include markdown or extra text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task: {task.get('title','')}\n"
                f"Prompt: {task.get('prompt','')}\n"
                f"Answer:\n{answer}\n\n"
                f"Return this JSON shape exactly: {schema}"
            ),
        },
    ]
    result = client.chat(model=judge_model, messages=judge_messages, temperature=0.0, num_ctx=num_ctx)
    parsed = _extract_json_object(result.get("content", ""))
    return {
        "judge_continuity_score": _clamp_score(parsed.get("judge_continuity_score")),
        "judge_actionability_score": _clamp_score(parsed.get("judge_actionability_score")),
        "judge_contradiction_score": _clamp_score(parsed.get("judge_contradiction_score")),
        "judge_notes": str(parsed.get("judge_notes", "")).strip() or None,
        "judge_raw": result.get("raw", {}),
    }


def _continuity_per_memory_token(continuity_recall: float, memory_tokens_used: int) -> float:
    if memory_tokens_used <= 0:
        return 0.0
    return continuity_recall / memory_tokens_used


def _dry_answer_for_mode(mode: str, task: Dict[str, Any], selected_memory: List[Dict[str, Any]]) -> str:
    if mode.startswith("cag"):
        lines = [dry_answer("cag", task, selected_memory)]
        if selected_memory:
            lines.append("Memory IDs: " + ", ".join(str(r.get("memory_id")) for r in selected_memory[:10]))
        return "\n".join(lines)
    return dry_answer(mode, task, selected_memory)


def _build_cag_engine(mode: str, path: Path, args) -> CAGMemoryEngine:
    if mode not in CAG_VARIANTS:
        raise ValueError(f"Not a CAG mode: {mode}")
    return CAGMemoryEngine(
        path,
        mode,
        brief_update_every_tasks=args.brief_update_every_tasks,
        brief_trigger_tokens=args.brief_trigger_tokens,
        compression_trigger_tokens=args.compression_trigger_tokens,
        max_project_brief_tokens=args.max_project_brief_tokens,
        max_scoped_memory_tokens=args.max_scoped_memory_tokens,
        max_recent_memory_tokens=args.max_recent_memory_tokens,
        max_source_evidence_tokens=args.max_source_evidence_tokens,
        max_total_context_tokens=args.max_total_context_tokens,
        semantic_backend=args.cag_semantic_backend,
        semantic_embed_model=args.cag_semantic_embed_model,
        brief_log_path=path.with_name(path.stem + "_briefs.jsonl"),
    )


def run_one(
    *,
    mode: str,
    task: Dict[str, Any],
    task_index: int,
    retriever: KeywordRetriever,
    client: OllamaClient,
    model: str,
    engine: CAGMemoryEngine | None,
    dry_run: bool,
    temperature: float,
    num_ctx: int,
    retrieval_k: int,
    judge: bool,
    judge_model: str | None,
):
    query = task["title"] + "\n" + task["prompt"] + "\n" + " ".join(task.get("tags", []))
    source_chunks = retriever.retrieve(query, k=retrieval_k)
    source_dicts = _source_dicts(source_chunks)

    retrieval = {
        "task_profile": None,
        "selected_memory": [],
        "recent_memory": [],
        "retrieval_scores": [],
        "brief_tokens_used": 0,
        "memory_tokens_used": 0,
    }

    if mode in CAG_VARIANTS:
        if engine is None:
            raise ValueError(f"Engine missing for CAG mode {mode}")
        retrieval = engine.retrieve_for_task(task, task_index, client=client)
        bundle = engine.build_context_bundle(task, source_dicts, retrieval)
        messages = bundle["messages"]
        memory_tokens_used = int(bundle["memory_tokens_used"])
        brief_tokens_used = int(bundle["brief_tokens_used"])
        source_tokens_used = int(bundle["source_tokens_used"])
        total_context_tokens = int(bundle["total_context_tokens"])
    else:
        messages = rag_prompt(task, source_chunks) if mode == "rag" else dag_prompt(task, source_chunks)
        memory_tokens_used = 0
        brief_tokens_used = 0
        source_tokens_used = approx_tokens("\n\n".join(s.text for s in source_chunks))
        total_context_tokens = approx_tokens(prompt_text(messages))

    ptext = prompt_text(messages)
    prompt_tokens = approx_tokens(ptext)

    if dry_run:
        answer = _dry_answer_for_mode(mode, task, retrieval.get("selected_memory", []))
        latency = 0.01
        raw = {"dry_run": True}
    else:
        result = client.chat(model=model, messages=messages, temperature=temperature, num_ctx=num_ctx)
        answer = result["content"]
        latency = result["latency_seconds"]
        raw = result["raw"]

    contradiction_check = {
        "checked": False,
        "contradictions_before_revision": 0,
        "contradictions_after_revision": 0,
        "revision_used": False,
        "answer": answer,
        "check_notes": [],
    }
    if mode in CAG_VARIANTS and engine is not None:
        contradiction_check = engine.contradiction_check(
            client=client,
            model=model,
            task=task,
            answer=answer,
            selected_memory=retrieval.get("selected_memory", []),
            dry_run=dry_run,
            num_ctx=num_ctx,
        )
        answer = contradiction_check.get("answer", answer)

    metrics = metric_scores(answer, task, prompt_tokens, latency)
    score = composite_score(metrics)

    expected_groups = expected_prior_concept_groups(task)
    expected_concepts = [str(g.get("concept", "")) for g in expected_groups if str(g.get("concept", ""))]
    memory_pr = memory_precision_recall(retrieval.get("selected_memory", []), expected_concepts, expected_groups)
    if mode not in CAG_VARIANTS:
        # Baselines do not retrieve persistent memory.
        memory_pr = {
            "memory_precision": 1.0,
            "memory_recall": 0.0 if expected_concepts else 1.0,
            "relevant_hits": 0,
            "selected_total": 0,
            "concept_hits": {c: False for c in expected_concepts},
            "irrelevant_memory_used": 0,
            "deprecated_memory_used": 0,
        }

    memory_items = retrieval.get("selected_memory", []) + retrieval.get("recent_memory", [])
    memory_reuse_count = int(sum(int(row.get("reuse_count") or 0) for row in memory_items))

    row = {
        "mode": mode,
        "task_id": task["id"],
        "task_index": task["index"],
        "task_title": task["title"],
        "score": score,
        **metrics,
        "memory_precision": round(float(memory_pr.get("memory_precision", 0.0)) * 100.0, 4),
        "memory_recall": round(float(memory_pr.get("memory_recall", 0.0)) * 100.0, 4),
        "memory_items_used": len(memory_items),
        "memory_tokens_used": int(memory_tokens_used),
        "brief_tokens_used": int(brief_tokens_used),
        "source_tokens_used": int(source_tokens_used),
        "total_context_tokens": int(total_context_tokens),
        "continuity_per_memory_token": round(
            _continuity_per_memory_token(_safe_float(metrics.get("continuity_recall", 0.0)), max(1, int(memory_tokens_used))),
            6,
        ),
        "deprecated_memory_used": int(memory_pr.get("deprecated_memory_used", 0)),
        "irrelevant_memory_used": int(memory_pr.get("irrelevant_memory_used", 0)),
        "contradictions_before_revision": int(contradiction_check.get("contradictions_before_revision", 0)),
        "contradictions_after_revision": int(contradiction_check.get("contradictions_after_revision", 0)),
        "revision_used": bool(contradiction_check.get("revision_used", False)),
        "memory_reuse_count": memory_reuse_count,
        "prompt_tokens": prompt_tokens,
        "latency_seconds": latency,
        "retrieved_sources": [s.source_id for s in source_chunks],
        "memory_task_ids_used": [r.get("source_task_id") for r in memory_items],
        "answer": answer,
        "raw": raw,
        "task_profile": retrieval.get("task_profile"),
        "retrieval_scores": retrieval.get("retrieval_scores", []),
        "expected_prior_concepts": expected_concepts,
        "memory_concept_hits": memory_pr.get("concept_hits", {}),
        "contradiction_check_notes": contradiction_check.get("check_notes", []),
    }

    if judge and not dry_run:
        row.update(_judge_answer(client, task, answer, judge_model or model, num_ctx))

    post = {
        "index_candidates": [],
        "promotion_decisions": [],
        "added_count": 0,
        "updated_count": 0,
        "discarded_count": 0,
        "task_local_count": 0,
    }
    brief_update = {"briefs_updated": False, "brief_reason": "n/a"}
    compression = {"compression_ran": False, "compression_reason": "n/a"}

    if mode in CAG_VARIANTS and engine is not None:
        post = engine.index_and_update(
            client=client,
            model=model,
            task=task,
            answer=answer,
            source_ids=row["retrieved_sources"],
            retrieved_memory=retrieval.get("selected_memory", []),
            dry_run=dry_run,
            num_ctx=num_ctx,
            temperature=temperature,
            current_task_index=task_index,
        )
        brief_update = engine.maybe_refresh_briefs(task_index)
        compression = engine.maybe_compress_memory(task_index)

    row["memory_update_candidates"] = post.get("index_candidates", [])
    row["memory_promotion_decisions"] = post.get("promotion_decisions", [])
    row["memory_added_count"] = int(post.get("added_count", 0))
    row["memory_updated_count"] = int(post.get("updated_count", 0))
    row["memory_discarded_count"] = int(post.get("discarded_count", 0))
    row["memory_task_local_count"] = int(post.get("task_local_count", 0))
    row["briefs_updated"] = bool(brief_update.get("briefs_updated", False))
    row["brief_update_reason"] = str(brief_update.get("brief_reason", ""))
    row["compression_ran"] = bool(compression.get("compression_ran", False))
    row["compression_reason"] = str(compression.get("compression_reason", ""))
    row["compression_merged_items"] = int(compression.get("merged_items", 0) or 0)

    return row


def failed_row(mode, task, err):
    return {
        "mode": mode,
        "task_id": task["id"],
        "task_index": task["index"],
        "task_title": task.get("title", ""),
        "score": None,
        "checklist_quality": None,
        "source_evidence_recall": None,
        "domain_rule_recall": None,
        "evidence_recall": None,
        "continuity_recall": None,
        "token_efficiency": None,
        "latency_efficiency": None,
        "contradiction_penalty": None,
        "memory_precision": None,
        "memory_recall": None,
        "memory_items_used": None,
        "memory_tokens_used": None,
        "brief_tokens_used": None,
        "source_tokens_used": None,
        "total_context_tokens": None,
        "continuity_per_memory_token": None,
        "deprecated_memory_used": None,
        "irrelevant_memory_used": None,
        "contradictions_before_revision": None,
        "contradictions_after_revision": None,
        "revision_used": None,
        "memory_reuse_count": None,
        "checklist_hit_count": None,
        "checklist_total": None,
        "source_evidence_hit_count": None,
        "source_evidence_total": None,
        "domain_rule_hit_count": None,
        "domain_rule_total": None,
        "evidence_hit_count": None,
        "evidence_total": None,
        "continuity_hit_count": None,
        "continuity_total": None,
        "contradiction_hit_count": None,
        "concept_hits": None,
        "contradiction_hits": None,
        "prompt_tokens": None,
        "latency_seconds": None,
        "retrieved_sources": [],
        "memory_task_ids_used": [],
        "answer": "",
        "raw": {},
        "task_profile": None,
        "retrieval_scores": [],
        "memory_update_candidates": [],
        "memory_promotion_decisions": [],
        "expected_prior_concepts": expected_prior_concepts(task),
        "memory_concept_hits": {},
        "memory_added_count": None,
        "memory_updated_count": None,
        "memory_discarded_count": None,
        "memory_task_local_count": None,
        "briefs_updated": None,
        "brief_update_reason": None,
        "compression_ran": None,
        "compression_reason": None,
        "compression_merged_items": None,
        "error": f"{type(err).__name__}: {err}",
    }


def write_phase_summary(summary_csv: Path, out_dir: Path) -> None:
    import pandas as pd

    df = pd.read_csv(summary_csv)
    needed = [
        "mode",
        "task_index",
        "score",
        "continuity_recall",
        "memory_precision",
        "memory_recall",
        "continuity_per_memory_token",
        "token_efficiency",
        "contradiction_penalty",
    ]
    for col in needed:
        if col not in df.columns:
            return

    df = df.copy()
    for col in needed[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["task_index"] = pd.to_numeric(df["task_index"], errors="coerce")
    df = df.dropna(subset=["task_index"])

    phase_rows = []

    phase_specs = [
        ("tasks_1_10", (1, 10)),
        ("tasks_11_20", (11, 20)),
        ("tasks_21_30", (21, 30)),
        ("tasks_20_30", (20, 30)),
        ("task_30_only", (30, 30)),
    ]

    for mode in sorted(df["mode"].dropna().astype(str).unique().tolist()):
        md = df[df["mode"] == mode]
        for phase_name, (start, end) in phase_specs:
            pdx = md[(md["task_index"] >= start) & (md["task_index"] <= end)]
            if pdx.empty:
                continue
            row = {
                "mode": mode,
                "phase": phase_name,
                "task_range": f"{start}-{end}" if start != end else f"{start}",
                "rows": len(pdx),
            }
            for metric in needed[2:]:
                row[metric] = float(pdx[metric].mean())
            phase_rows.append(row)

    if not phase_rows:
        return

    phase_csv = out_dir / "phase_summary.csv"
    with phase_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(phase_rows[0].keys()))
        writer.writeheader()
        for row in phase_rows:
            writer.writerow(row)

    lines = ["# Phase Summary", "", "Averages by mode for late-stage degradation visibility.", ""]
    for row in phase_rows:
        lines.append(
            f"- {row['mode']} | {row['phase']} ({row['task_range']}): "
            f"composite={row['score']:.2f}, continuity={row['continuity_recall']:.2f}, "
            f"memory_precision={row['memory_precision']:.2f}, memory_recall={row['memory_recall']:.2f}, "
            f"continuity_per_memory_token={row['continuity_per_memory_token']:.6f}, "
            f"token_efficiency={row['token_efficiency']:.2f}, contradiction_penalty={row['contradiction_penalty']:.2f}"
        )
    (out_dir / "phase_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_ablation_delta_summary(summary_csv: Path, out_dir: Path) -> None:
    import pandas as pd

    df = pd.read_csv(summary_csv)
    needed_modes = ["cag_naive", "cag_scoped", "cag_briefed", "cag_full"]
    present_modes = set(df.get("mode", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    if len(present_modes.intersection(needed_modes)) < 2:
        return

    df = df.copy()
    if "task_index" not in df.columns:
        return
    df["task_index"] = pd.to_numeric(df["task_index"], errors="coerce")
    late = df[(df["task_index"] >= 20) & (df["task_index"] <= 30)]
    if late.empty:
        return

    metric_cols = [
        "score",
        "continuity_recall",
        "memory_precision",
        "memory_recall",
        "continuity_per_memory_token",
        "memory_tokens_used",
        "brief_tokens_used",
        "contradictions_before_revision",
        "contradictions_after_revision",
        "contradiction_penalty",
    ]
    for col in metric_cols:
        if col in late.columns:
            late[col] = pd.to_numeric(late[col], errors="coerce")

    agg_rows: List[Dict[str, Any]] = []
    by_mode: Dict[str, Dict[str, float]] = {}
    for mode in needed_modes:
        md = late[late["mode"] == mode]
        if md.empty:
            continue
        row: Dict[str, Any] = {"mode": mode, "rows": len(md)}
        for m in metric_cols:
            if m in md.columns:
                row[m] = float(md[m].mean())
        agg_rows.append(row)
        by_mode[mode] = {k: v for k, v in row.items() if k in metric_cols}

    if not agg_rows:
        return

    agg_csv = out_dir / "ablation_summary_late_stage.csv"
    with agg_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        for row in agg_rows:
            writer.writerow(row)

    chain = [("cag_naive", "cag_scoped"), ("cag_scoped", "cag_briefed"), ("cag_briefed", "cag_full")]
    delta_rows: List[Dict[str, Any]] = []
    for left, right in chain:
        if left not in by_mode or right not in by_mode:
            continue
        delta = {"from_mode": left, "to_mode": right}
        for m in metric_cols:
            if m not in by_mode[left] or m not in by_mode[right]:
                continue
            delta[f"delta_{m}"] = float(by_mode[right][m] - by_mode[left][m])
        delta_rows.append(delta)

    if delta_rows:
        delta_csv = out_dir / "ablation_delta_summary.csv"
        with delta_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(delta_rows[0].keys()))
            writer.writeheader()
            for row in delta_rows:
                writer.writerow(row)

    lines = [
        "# CAG Ablation Delta Summary",
        "",
        "Late-stage window: tasks 20-30.",
        "",
    ]
    for row in agg_rows:
        lines.append(
            f"- {row['mode']}: score={row.get('score', 0.0):.2f}, continuity={row.get('continuity_recall', 0.0):.2f}, "
            f"memory_precision={row.get('memory_precision', 0.0):.2f}, memory_recall={row.get('memory_recall', 0.0):.2f}, "
            f"continuity_per_memory_token={row.get('continuity_per_memory_token', 0.0):.6f}, "
            f"memory_tokens={row.get('memory_tokens_used', 0.0):.2f}, brief_tokens={row.get('brief_tokens_used', 0.0):.2f}, "
            f"contradiction_penalty={row.get('contradiction_penalty', 0.0):.2f}"
        )
    if delta_rows:
        lines.append("")
        lines.append("## Incremental Deltas")
        for row in delta_rows:
            lines.append(
                f"- {row['from_mode']} -> {row['to_mode']}: "
                f"Δscore={row.get('delta_score', 0.0):+.2f}, "
                f"Δcontinuity={row.get('delta_continuity_recall', 0.0):+.2f}, "
                f"Δmemory_recall={row.get('delta_memory_recall', 0.0):+.2f}, "
                f"Δcontinuity_per_memory_token={row.get('delta_continuity_per_memory_token', 0.0):+.6f}, "
                f"Δmemory_tokens={row.get('delta_memory_tokens_used', 0.0):+.2f}, "
                f"Δcontradiction_penalty={row.get('delta_contradiction_penalty', 0.0):+.2f}"
            )
    (out_dir / "ablation_delta_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Benchmark RAG, DAG, and standalone CAG variants over one iterative project task sequence.")
    p.add_argument("--model", default="qwen2.5-coder:7b")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--tasks", default="data/tasks.jsonl")
    p.add_argument("--sources", default="data/sources")
    p.add_argument("--out", default="results/run")
    p.add_argument("--max-tasks", type=int, default=30)
    p.add_argument("--task-start", type=int, default=1)
    p.add_argument("--repeat-runs", type=int, default=1)
    p.add_argument("--retrieval-k", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--raw-trials", action="store_true")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--request-retries", type=int, default=1)
    p.add_argument("--retry-backoff", type=float, default=3.0)
    p.add_argument("--judge", action="store_true", help="Enable optional LLM-as-judge metrics in raw outputs")
    p.add_argument("--judge-model", default=None, help="Judge model; defaults to --model when --judge is enabled")
    p.add_argument("--run-id", default=None, help="Optional run folder name under --out")
    p.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--suite", choices=["baselines", "cag-ablation", "all"], default="baselines")
    p.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help="Mode override. Supports comma-separated and/or space-separated values. Example: --modes rag,dag,cag_full or --modes rag dag cag_full",
    )

    p.add_argument("--brief-update-every-tasks", type=int, default=5)
    p.add_argument("--brief-trigger-tokens", type=int, default=1200)
    p.add_argument("--compression-trigger-tokens", type=int, default=2600)
    p.add_argument("--max-project-brief-tokens", type=int, default=350)
    p.add_argument("--max-scoped-memory-tokens", type=int, default=900)
    p.add_argument("--max-recent-memory-tokens", type=int, default=250)
    p.add_argument("--max-source-evidence-tokens", type=int, default=1000)
    p.add_argument("--max-total-context-tokens", type=int, default=2600)
    p.add_argument("--cag-semantic-backend", choices=["tfidf", "ollama"], default="tfidf")
    p.add_argument("--cag-semantic-embed-model", default="nomic-embed-text")

    args = p.parse_args()

    out_base = Path(args.out)
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    out = out_base / run_id
    out.mkdir(parents=True, exist_ok=True)

    modes = _resolve_modes(args.suite, args.modes)

    tasks = [t for t in load_jsonl(args.tasks) if int(t.get("index", 0)) >= args.task_start][: args.max_tasks]
    if not tasks:
        raise ValueError("No tasks selected. Check --task-start and --max-tasks.")

    retriever = KeywordRetriever.from_sources_dir(args.sources)
    client = OllamaClient(
        args.ollama_url,
        timeout=args.timeout,
        retries=args.request_retries,
        retry_backoff=args.retry_backoff,
    )

    summary_path = out / "summary.csv"
    raw_path = out / "raw.jsonl"

    fieldnames = [
        "trial",
        "mode",
        "task_id",
        "task_index",
        "task_title",
        "score",
        "checklist_quality",
        "source_evidence_recall",
        "domain_rule_recall",
        "evidence_recall",
        "continuity_recall",
        "memory_precision",
        "memory_recall",
        "memory_items_used",
        "memory_tokens_used",
        "brief_tokens_used",
        "source_tokens_used",
        "total_context_tokens",
        "continuity_per_memory_token",
        "token_efficiency",
        "latency_efficiency",
        "contradiction_penalty",
        "deprecated_memory_used",
        "irrelevant_memory_used",
        "contradictions_before_revision",
        "contradictions_after_revision",
        "revision_used",
        "memory_reuse_count",
        "checklist_hit_count",
        "checklist_total",
        "source_evidence_hit_count",
        "source_evidence_total",
        "domain_rule_hit_count",
        "domain_rule_total",
        "evidence_hit_count",
        "evidence_total",
        "continuity_hit_count",
        "continuity_total",
        "contradiction_hit_count",
        "prompt_tokens",
        "latency_seconds",
        "retrieved_sources",
        "memory_task_ids_used",
        "memory_added_count",
        "memory_updated_count",
        "memory_discarded_count",
        "memory_task_local_count",
        "briefs_updated",
        "brief_update_reason",
        "compression_ran",
        "compression_reason",
        "compression_merged_items",
        "error",
    ]

    run_config = vars(args).copy()
    run_config["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_config["resolved_out_dir"] = str(out)
    run_config["task_count"] = len(tasks)
    run_config["resolved_modes"] = modes
    run_config["suite_modes"] = SUITE_MODES
    run_config["cag_variants"] = sorted(CAG_VARIANTS)
    run_config["phase_buckets"] = ["tasks_1_10", "tasks_11_20", "tasks_21_30", "tasks_20_30", "task_30_only"]
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for trial in range(1, args.repeat_runs + 1):
            print(f"Trial {trial}/{args.repeat_runs}", flush=True)
            engines: Dict[str, CAGMemoryEngine] = {}
            for mode in modes:
                if mode in CAG_VARIANTS:
                    engines[mode] = _build_cag_engine(mode, out / f"{mode}_memory_trial_{trial:02d}.jsonl", args)

            for task_i, task in enumerate(tasks, start=1):
                for mode in modes:
                    print(f"Running {mode} task {task_i}/{len(tasks)} ({task.get('id')})", flush=True)
                    try:
                        row = run_one(
                            mode=mode,
                            task=task,
                            task_index=int(task.get("index", task_i)),
                            retriever=retriever,
                            client=client,
                            model=args.model,
                            engine=engines.get(mode),
                            dry_run=args.dry_run,
                            temperature=args.temperature,
                            num_ctx=args.num_ctx,
                            retrieval_k=args.retrieval_k,
                            judge=args.judge,
                            judge_model=args.judge_model,
                        )
                    except requests.exceptions.RequestException as exc:
                        print(f"WARN trial={trial} mode={mode} task={task.get('id')} request failed: {exc}")
                        row = failed_row(mode, task, exc)
                        if not args.continue_on_error:
                            raise
                    except Exception as exc:
                        print(f"WARN trial={trial} mode={mode} task={task.get('id')} unexpected error: {exc}")
                        row = failed_row(mode, task, exc)
                        if not args.continue_on_error:
                            raise

                    row["trial"] = trial
                    append_jsonl(raw_path, row)
                    writer.writerow({k: row.get(k) for k in fieldnames})
                    f.flush()

    (out / "weights.json").write_text(json.dumps(BASE_WEIGHTS, indent=2), encoding="utf-8")

    try:
        from .plot import make_plots

        make_plots(summary_path, out, args.raw_trials)
        print(f"Wrote plots to {out}")
    except Exception as exc:
        print(f"WARN plotting skipped: {exc}")

    try:
        write_phase_summary(summary_path, out)
        print(f"Wrote phase summary to {out}")
    except Exception as exc:
        print(f"WARN phase summary skipped: {exc}")

    try:
        write_ablation_delta_summary(summary_path, out)
        print(f"Wrote ablation delta summary to {out}")
    except Exception as exc:
        print(f"WARN ablation delta summary skipped: {exc}")

    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
