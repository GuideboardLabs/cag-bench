import argparse
import csv
import json
import sys
import time
from pathlib import Path

import requests

from .memory import ProjectMemory
from .ollama_client import OllamaClient, approx_tokens
from .prompts import cag_prompt, dag_prompt, dry_answer, format_memory, rag_prompt
from .retrieval import KeywordRetriever
from .scoring import BASE_WEIGHTS, composite_score, metric_scores
from .utils import append_jsonl, coerce_concept_groups, contains_term, load_jsonl


SUITES = {
    "baselines": ["rag", "dag", "cag"],
    "cag-ablation": ["cag", "cag_scoped", "cag_oracle_memory"],
    "all": ["rag", "dag", "cag", "cag_scoped", "cag_oracle_memory"],
}
ALL_MODES = tuple(sorted({m for values in SUITES.values() for m in values}))


def prompt_text(messages):
    return "\n\n".join([m["role"] + ": " + m["content"] for m in messages])


def accepted_memory_summary(task):
    return task.get("promote_summary", "")


def _norm_terms(values):
    terms = []
    for group in coerce_concept_groups(values):
        terms.extend(group.get("accepted_terms", []))
    return {str(v).strip().lower() for v in terms if str(v).strip()}


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


def judge_answer(client, task, answer, judge_model, num_ctx):
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
    result = client.chat(
        model=judge_model, messages=judge_messages, temperature=0.0, num_ctx=num_ctx
    )
    parsed = _extract_json_object(result.get("content", ""))
    return {
        "judge_continuity_score": _clamp_score(parsed.get("judge_continuity_score")),
        "judge_actionability_score": _clamp_score(parsed.get("judge_actionability_score")),
        "judge_contradiction_score": _clamp_score(
            parsed.get("judge_contradiction_score")
        ),
        "judge_notes": str(parsed.get("judge_notes", "")).strip() or None,
        "judge_raw": result.get("raw", {}),
    }


def validate_continuity_ladder(tasks):
    if not tasks:
        raise ValueError("No tasks loaded.")
    sorted_tasks = sorted(tasks, key=lambda t: int(t.get("index", 0)))
    if [t.get("id") for t in sorted_tasks] != [t.get("id") for t in tasks]:
        raise ValueError("Tasks must be sorted by ascending index.")
    if _norm_terms(tasks[0].get("continuity_terms", [])):
        raise ValueError(
            f"Task {tasks[0].get('id')} must be fresh and have empty continuity_terms."
        )

    for i, task in enumerate(tasks):
        expected_index = i + 1
        if int(task.get("index", 0)) != expected_index:
            raise ValueError(
                f"Task ordering invalid at {task.get('id')}: expected index {expected_index}, got {task.get('index')}."
            )
        if i == 0:
            continue
        continuity = _norm_terms(task.get("continuity_terms", []))
        missing_prior = []
        for prev in tasks[:i]:
            promoted = _norm_terms(prev.get("promote_terms", []))
            if not promoted:
                missing_prior.append(prev.get("id"))
                continue
            if not (promoted & continuity):
                missing_prior.append(prev.get("id"))
        if missing_prior:
            raise ValueError(
                f"Task {task.get('id')} continuity ladder violation: missing references to prior tasks {missing_prior}. "
                "Each task N>1 must include at least one promoted term from every prior task."
            )


def _parse_modes_csv(text: str) -> list[str]:
    modes = [part.strip() for part in str(text).split(",") if part.strip()]
    if not modes:
        return []
    invalid = [mode for mode in modes if mode not in ALL_MODES]
    if invalid:
        raise ValueError(
            f"Invalid mode(s): {invalid}. Expected one of: {sorted(ALL_MODES)}"
        )
    deduped = []
    seen = set()
    for mode in modes:
        if mode in seen:
            continue
        deduped.append(mode)
        seen.add(mode)
    return deduped


def _resolve_modes(args, parser, suite_explicit: bool) -> list[str]:
    if args.modes and suite_explicit:
        parser.error("--modes cannot be combined with --suite.")
    if args.modes:
        try:
            modes = _parse_modes_csv(args.modes)
        except ValueError as exc:
            parser.error(str(exc))
        if not modes:
            parser.error("--modes cannot be empty.")
        return modes
    return list(SUITES[args.suite])


def _concept_key(group: dict) -> str:
    concept = str(group.get("concept") or "").strip().lower()
    if concept:
        return concept
    accepted = [str(v).strip().lower() for v in group.get("accepted_terms", []) if str(v).strip()]
    return accepted[0] if accepted else ""


def _build_continuity_catalog(tasks: list[dict]) -> list[dict]:
    merged: dict[str, set[str]] = {}
    for task in tasks:
        for group in coerce_concept_groups(task.get("continuity_terms", [])):
            key = _concept_key(group)
            if not key:
                continue
            accepted = {
                str(v).strip().lower()
                for v in group.get("accepted_terms", [])
                if str(v).strip()
            }
            if key not in merged:
                merged[key] = set()
            merged[key].update(accepted)
    return [
        {"key": key, "accepted_terms": sorted(values)}
        for key, values in sorted(merged.items())
    ]


def _row_hits_concept(row: dict, concept: dict) -> bool:
    text = str(row.get("text", "") or "")
    promoted_terms = {
        str(v).strip().lower()
        for v in row.get("promoted_terms", [])
        if str(v).strip()
    }
    accepted_terms = concept.get("accepted_terms", [])
    if not accepted_terms:
        return False
    if any(contains_term(text, term) for term in accepted_terms):
        return True
    return bool(promoted_terms & set(accepted_terms))


def _task_continuity_concepts(task: dict) -> list[dict]:
    concepts = []
    for group in coerce_concept_groups(task.get("continuity_terms", [])):
        key = _concept_key(group)
        if not key:
            continue
        accepted = {
            str(v).strip().lower()
            for v in group.get("accepted_terms", [])
            if str(v).strip()
        }
        concepts.append({"key": key, "accepted_terms": sorted(accepted)})
    return concepts


def _memory_metrics(
    task: dict,
    memory_rows: list[dict],
    continuity_catalog: list[dict],
    mode_uses_memory: bool,
) -> dict:
    if not mode_uses_memory:
        return {
            "memory_recall": None,
            "memory_precision": None,
            "memory_tokens_used": 0,
            "continuity_per_memory_token": None,
            "irrelevant_memory_used": None,
            "selected_memory_ids": [],
        }

    selected_memory_ids = [row.get("memory_id") for row in memory_rows if row.get("memory_id")]
    memory_tokens_used = approx_tokens(format_memory(memory_rows)) if memory_rows else 0

    task_concepts = _task_continuity_concepts(task)
    if task_concepts:
        hit_count = 0
        for concept in task_concepts:
            if any(_row_hits_concept(row, concept) for row in memory_rows):
                hit_count += 1
        memory_recall = (hit_count / len(task_concepts)) * 100.0
    else:
        memory_recall = None

    present_concepts = set()
    for concept in continuity_catalog:
        if any(_row_hits_concept(row, concept) for row in memory_rows):
            present_concepts.add(concept["key"])
    task_concept_keys = {concept["key"] for concept in task_concepts}
    if present_concepts:
        memory_precision = (
            len(present_concepts & task_concept_keys) / len(present_concepts)
        ) * 100.0
    else:
        memory_precision = None

    if memory_recall is not None and memory_tokens_used > 0:
        continuity_per_memory_token = memory_recall / memory_tokens_used
    else:
        continuity_per_memory_token = None

    task_terms = {
        str(v).strip().lower()
        for concept in task_concepts
        for v in concept.get("accepted_terms", [])
        if str(v).strip()
    }
    irrelevant_memory_used = 0
    for row in memory_rows:
        promoted = {
            str(v).strip().lower()
            for v in row.get("promoted_terms", [])
            if str(v).strip()
        }
        if not (promoted & task_terms):
            irrelevant_memory_used += 1

    return {
        "memory_recall": memory_recall,
        "memory_precision": memory_precision,
        "memory_tokens_used": memory_tokens_used,
        "continuity_per_memory_token": continuity_per_memory_token,
        "irrelevant_memory_used": irrelevant_memory_used,
        "selected_memory_ids": selected_memory_ids,
    }


def _retrieve_oracle_memory(
    memory: ProjectMemory, task: dict, k: int | None = None
) -> list[dict]:
    continuity_terms = {
        str(v).strip().lower()
        for group in coerce_concept_groups(task.get("continuity_terms", []))
        for v in group.get("accepted_terms", [])
        if str(v).strip()
    }
    if not continuity_terms:
        return []
    selected = []
    for row in memory.rows():
        promoted = {
            str(v).strip().lower()
            for v in row.get("promoted_terms", [])
            if str(v).strip()
        }
        if promoted & continuity_terms:
            selected.append(row)
    selected = list(reversed(selected))
    if k is None:
        return selected
    return selected[:k]


def run_one(
    mode,
    task,
    retriever,
    client,
    model,
    memory,
    dry_run,
    temperature,
    num_ctx,
    retrieval_k,
    memory_top_k,
    continuity_catalog,
    judge=False,
    judge_model=None,
):
    query = task["title"] + "\n" + task["prompt"] + "\n" + " ".join(task.get("tags", []))
    sources = retriever.retrieve(query, k=retrieval_k)

    retrieval_scores = None
    if mode in ("cag", "cag_scoped", "cag_oracle_memory") and memory is not None:
        if mode == "cag_scoped":
            memory_rows, retrieval_scores = memory.retrieve_scoped(
                task, k=memory_top_k, return_scores=True
            )
        elif mode == "cag_oracle_memory":
            memory_rows = _retrieve_oracle_memory(memory, task, k=memory_top_k)
        else:
            memory_rows = memory.retrieve(task)
    else:
        memory_rows = []

    if mode == "rag":
        messages = rag_prompt(task, sources)
    elif mode == "dag":
        messages = dag_prompt(task, sources)
    else:
        messages = cag_prompt(task, sources, memory_rows)

    ptext = prompt_text(messages)
    prompt_tokens = approx_tokens(ptext)

    if dry_run:
        answer = dry_answer("cag" if mode.startswith("cag") else mode, task, memory_rows)
        latency = 0.01
        raw = {"dry_run": True}
    else:
        result = client.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            num_ctx=num_ctx,
        )
        answer = result["content"]
        latency = result["latency_seconds"]
        raw = result["raw"]

    metrics = metric_scores(answer, task, prompt_tokens, latency)
    memory_metrics = _memory_metrics(
        task,
        memory_rows,
        continuity_catalog,
        mode_uses_memory=mode.startswith("cag"),
    )
    metrics.update(memory_metrics)

    score = composite_score(metrics)
    row = {
        "status": "ok",
        "mode": mode,
        "task_id": task["id"],
        "task_index": task["index"],
        "task_title": task["title"],
        "score": score,
        **metrics,
        "prompt_tokens": prompt_tokens,
        "latency_seconds": latency,
        "retrieved_sources": [s.source_id for s in sources],
        "memory_items_used": len(memory_rows),
        "memory_task_ids_used": [r.get("task_id") for r in memory_rows],
        "answer": answer,
        "raw": raw,
        "error": None,
    }
    if retrieval_scores is not None:
        row["retrieval_scores"] = retrieval_scores
    if judge and not dry_run:
        row.update(judge_answer(client, task, answer, judge_model or model, num_ctx))
    return row


def failed_row(mode, task, err):
    return {
        "status": "failed",
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
        "memory_recall": None,
        "memory_precision": None,
        "memory_tokens_used": None,
        "continuity_per_memory_token": None,
        "irrelevant_memory_used": None,
        "token_efficiency": None,
        "latency_efficiency": None,
        "contradiction_penalty": None,
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
        "selected_memory_ids": [],
        "prompt_tokens": None,
        "latency_seconds": None,
        "retrieved_sources": [],
        "memory_items_used": 0,
        "memory_task_ids_used": [],
        "answer": "",
        "raw": {},
        "retrieval_scores": None,
        "error": f"{type(err).__name__}: {err}",
    }


def main():
    p = argparse.ArgumentParser(
        description="Benchmark RAG vs DAG vs CAG over one iterative project task sequence."
    )
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
    p.add_argument(
        "--memory-top-k",
        type=int,
        default=5,
        help="Top-K memory rows for cag_scoped and cag_oracle_memory modes.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--raw-trials", action="store_true")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--request-retries", type=int, default=1)
    p.add_argument("--retry-backoff", type=float, default=3.0)
    p.add_argument(
        "--judge",
        action="store_true",
        help="Enable optional LLM-as-judge metrics in raw outputs",
    )
    p.add_argument(
        "--judge-model",
        default=None,
        help="Judge model; defaults to --model when --judge is enabled",
    )
    p.add_argument("--run-id", default=None, help="Optional run folder name under --out")
    p.add_argument(
        "--continue-on-error", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--suite", choices=sorted(SUITES.keys()), default="baselines")
    p.add_argument(
        "--modes",
        default=None,
        help="Comma-separated mode list. Overrides --suite (example: rag,dag,cag)",
    )
    suite_explicit = "--suite" in sys.argv[1:]
    args = p.parse_args()
    if args.memory_top_k <= 0:
        p.error("--memory-top-k must be > 0.")
    resolved_modes = _resolve_modes(args, p, suite_explicit)

    out_base = Path(args.out)
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    out = out_base / run_id
    out.mkdir(parents=True, exist_ok=True)

    tasks = [t for t in load_jsonl(args.tasks) if t["index"] >= args.task_start][: args.max_tasks]
    validate_continuity_ladder(tasks)
    continuity_catalog = _build_continuity_catalog(tasks)

    retriever = KeywordRetriever.from_sources_dir(args.sources)
    client = OllamaClient(
        args.ollama_url,
        timeout=args.timeout,
        retries=args.request_retries,
        retry_backoff=args.retry_backoff,
    )
    summary_path = out / "summary.csv"
    raw_path = out / "raw.jsonl"
    failures_path = out / "failures.csv"
    fieldnames = [
        "trial",
        "status",
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
        "memory_recall",
        "memory_precision",
        "memory_tokens_used",
        "continuity_per_memory_token",
        "irrelevant_memory_used",
        "token_efficiency",
        "latency_efficiency",
        "contradiction_penalty",
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
        "memory_items_used",
        "retrieved_sources",
        "memory_task_ids_used",
        "selected_memory_ids",
        "error",
    ]

    run_config = vars(args).copy()
    run_config["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_config["resolved_out_dir"] = str(out)
    run_config["task_count"] = len(tasks)
    run_config["resolved_modes"] = resolved_modes
    (out / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    failure_counts = {mode: 0 for mode in resolved_modes}
    with open(summary_path, "w", newline="", encoding="utf-8") as f_sum, open(
        failures_path, "w", newline="", encoding="utf-8"
    ) as f_fail:
        writer = csv.DictWriter(f_sum, fieldnames=fieldnames)
        writer.writeheader()
        failures_writer = csv.DictWriter(
            f_fail,
            fieldnames=["mode", "trial", "task_id", "task_index", "error"],
        )
        failures_writer.writeheader()

        for trial in range(1, args.repeat_runs + 1):
            print(f"Trial {trial}/{args.repeat_runs}", flush=True)
            memory = ProjectMemory(out / f"cag_memory_trial_{trial:02d}.jsonl", trial=trial)
            for task_i, task in enumerate(tasks, start=1):
                for mode in resolved_modes:
                    print(
                        f"Running {mode} task {task_i}/{len(tasks)} ({task.get('id')})",
                        flush=True,
                    )
                    try:
                        row = run_one(
                            mode,
                            task,
                            retriever,
                            client,
                            args.model,
                            memory if mode.startswith("cag") else None,
                            args.dry_run,
                            args.temperature,
                            args.num_ctx,
                            args.retrieval_k,
                            args.memory_top_k,
                            continuity_catalog,
                            judge=args.judge,
                            judge_model=args.judge_model,
                        )
                    except requests.exceptions.RequestException as exc:
                        print(
                            f"WARN trial={trial} mode={mode} task={task.get('id')} request failed: {exc}"
                        )
                        row = failed_row(mode, task, exc)
                        failure_counts[mode] += 1
                        failures_writer.writerow(
                            {
                                "mode": mode,
                                "trial": trial,
                                "task_id": task.get("id"),
                                "task_index": task.get("index"),
                                "error": row.get("error"),
                            }
                        )
                        f_fail.flush()
                        if not args.continue_on_error:
                            raise
                    except Exception as exc:
                        print(
                            f"WARN trial={trial} mode={mode} task={task.get('id')} unexpected error: {exc}"
                        )
                        row = failed_row(mode, task, exc)
                        failure_counts[mode] += 1
                        failures_writer.writerow(
                            {
                                "mode": mode,
                                "trial": trial,
                                "task_id": task.get("id"),
                                "task_index": task.get("index"),
                                "error": row.get("error"),
                            }
                        )
                        f_fail.flush()
                        if not args.continue_on_error:
                            raise

                    row["trial"] = trial
                    append_jsonl(raw_path, row)
                    writer.writerow({k: row.get(k) for k in fieldnames})
                    f_sum.flush()

                memory.add(task, accepted_memory_summary(task))

    (out / "weights.json").write_text(json.dumps(BASE_WEIGHTS, indent=2), encoding="utf-8")

    try:
        from .plot import make_plots

        make_plots(summary_path, out, args.raw_trials)
        print(f"Wrote plots to {out}")
    except Exception as exc:
        print(f"WARN plotting skipped: {exc}")

    failed_total = sum(failure_counts.values())
    print(f"Failures: {failed_total}")
    for mode in resolved_modes:
        print(f"  {mode}: {failure_counts.get(mode, 0)}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
