from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any

import requests

from .eval import composite_score, id_recall, semantic_coverage
from .io import append_jsonl, read_jsonl, write_jsonl
from .ollama_client import OllamaClient, parse_json_object
from .prompts import answer_prompt, dag_stage_prompt, memory_update_prompt
from .vector import OllamaEmbedder, TfidfEmbedder, VectorIndex


def make_embedder(args, client):
    if args.embedder == 'ollama':
        return OllamaEmbedder(client, args.embed_model)
    return TfidfEmbedder()


def select_docs(index: VectorIndex, query: str, k: int) -> list[dict[str, Any]]:
    hits = index.search(query, k=k)
    docs = []
    for h in hits:
        d = dict(h.meta)
        d['score'] = h.score
        docs.append(d)
    return docs


def run_chat(client: OllamaClient, args, messages: list[dict[str, str]], json_mode: bool = True):
    if args.dry_run:
        content = messages[-1]['content']
        return {
            'obj': {'answer': 'DRY RUN answer using visible context. ' + content[:500], 'used_context_ids': [], 'risk_notes': [], 'memory_updates': []},
            'text': content[:1000],
            'seconds': 0.01,
            'prompt_eval_count': len(content.split()),
            'eval_count': 20,
            'raw': {},
        }
    result = client.chat(
        args.model,
        messages,
        temperature=args.temperature,
        seed=args.seed,
        json_mode=json_mode,
        num_ctx=args.num_ctx,
    )
    return {
        'obj': parse_json_object(result.text),
        'text': result.text,
        'seconds': result.seconds,
        'prompt_eval_count': result.prompt_eval_count,
        'eval_count': result.eval_count,
        'raw': result.raw,
    }


def memory_docs(memory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs = []
    for i, m in enumerate(memory):
        mid = m.get('id') or f"mem_{i+1:04d}"
        docs.append({'id': mid, 'text': m.get('text', ''), 'scope': m.get('scope', 'unknown'), **m})
    return docs


def transient_docs_for_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    docs = []
    for i, text in enumerate(task.get('transient_context', []) or [], start=1):
        docs.append({'id': f"{task['id']}_note_{i:02d}", 'scope': 'current-task', 'text': str(text)})
    return docs


def task_window(tasks: list[dict[str, Any]], task_start: int, max_tasks: int | None) -> list[dict[str, Any]]:
    if task_start < 1:
        raise ValueError('--task-start must be 1 or greater')
    start = task_start - 1
    end = None if max_tasks is None else start + max_tasks
    selected = tasks[start:end]
    if not selected:
        raise ValueError('Task selection returned no tasks. Check --task-start and --max-tasks.')
    return selected


def write_summary_csv(rows: list[dict[str, Any]], summary_path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with summary_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def run_mode(mode: str, args, corpus: list[dict], tasks: list[dict], out_dir: Path, trial: int) -> list[dict[str, Any]]:
    client = OllamaClient(
        args.ollama_host,
        timeout=args.timeout,
        retries=args.request_retries,
        retry_backoff=args.retry_backoff,
    )
    embedder = make_embedder(args, client)
    corpus_index = VectorIndex(embedder)
    corpus_index.build(corpus)
    memory: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    raw_path = out_dir / f'{mode}_raw.jsonl'

    for task in tasks:
        query = task['text'] + ' ' + ' '.join(task.get('query_terms', []))
        corpus_context = select_docs(corpus_index, query, args.top_k)
        transient_context = transient_docs_for_task(task)
        memory_context: list[dict[str, Any]] = []

        total_seconds = 0.0
        total_prompt_tokens = 0
        total_eval_tokens = 0
        answer_obj: dict[str, Any] = {}
        answer_text: str

        try:
            if mode == 'cag':
                mem = memory_docs(memory)
                if mem:
                    mem_index = VectorIndex(embedder)
                    mem_index.build(mem)
                    memory_context = select_docs(mem_index, query, args.memory_top_k)
                ans = run_chat(client, args, answer_prompt(task, corpus_context, memory_context, 'cag', transient_context))
                answer_obj = ans['obj']
                answer_text = str(answer_obj.get('answer') or ans['text'])
                total_seconds += ans['seconds']
                total_prompt_tokens += ans['prompt_eval_count']
                total_eval_tokens += ans['eval_count']

                upd = run_chat(client, args, memory_update_prompt(task, answer_text, corpus_context, memory_context, transient_context))
                total_seconds += upd['seconds']
                total_prompt_tokens += upd['prompt_eval_count']
                total_eval_tokens += upd['eval_count']
                updates = upd['obj'].get('memory_updates', []) if isinstance(upd['obj'], dict) else []
                if isinstance(updates, list):
                    for u in updates:
                        if not isinstance(u, dict):
                            continue
                        text = str(u.get('text', '')).strip()
                        conf = float(u.get('confidence') or 0.0)
                        if text and conf >= args.memory_min_confidence:
                            u['id'] = f"t{trial:02d}_mem_{len(memory) + 1:04d}"
                            u['trial'] = trial
                            u['created_after_task'] = task['id']
                            memory.append(u)
            elif mode == 'dag':
                previous = ''
                for stage in ['requirements', 'implementation_plan', 'tests_and_risks']:
                    res = run_chat(client, args, dag_stage_prompt(task, corpus_context, stage, previous, transient_context))
                    obj = res['obj']
                    previous += '\n' + str(obj.get('output') or obj.get('answer') or res['text'])
                    total_seconds += res['seconds']
                    total_prompt_tokens += res['prompt_eval_count']
                    total_eval_tokens += res['eval_count']
                answer_obj = {'answer': previous, 'used_context_ids': [d['id'] for d in corpus_context + transient_context], 'risk_notes': []}
                answer_text = previous
            else:
                ans = run_chat(client, args, answer_prompt(task, corpus_context, [], 'rag', transient_context))
                answer_obj = ans['obj']
                answer_text = str(answer_obj.get('answer') or ans['text'])
                total_seconds += ans['seconds']
                total_prompt_tokens += ans['prompt_eval_count']
                total_eval_tokens += ans['eval_count']
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.RequestException) as exc:
            row = {
                'trial': trial,
                'seed': args.seed,
                'mode': mode,
                'task_id': task['id'],
                'task_index': task.get('index'),
                'score': None,
                'task_quality': None,
                'evidence_recall': None,
                'memory_recall': None,
                'prompt_tokens': total_prompt_tokens,
                'eval_tokens': total_eval_tokens,
                'seconds': round(total_seconds, 4),
                'selected_context_ids': [str(d['id']) for d in corpus_context],
                'selected_memory_ids': [str(d['id']) for d in memory_context],
                'current_task_note_ids': [str(d['id']) for d in transient_context],
                'memory_size_after': len(memory),
                'answer': '',
                'checklist_details': [],
                'memory_details': [],
                'error': f'{type(exc).__name__}: {exc}',
            }
            rows.append(row)
            append_jsonl(raw_path, row)
            print(f'WARN mode={mode} trial={trial} task={task.get("id")} request failed: {exc}')
            if not args.continue_on_error:
                raise
            continue

        selected_corpus_ids = [str(d['id']) for d in corpus_context]
        selected_mem_ids = [str(d['id']) for d in memory_context]
        selected_transient_ids = [str(d['id']) for d in transient_context]
        expected_context_ids = [str(x) for x in task.get('expected_context_ids', [])]
        expected_memory_text = [str(x) for x in task.get('expected_memory_facts', [])]
        checklist = [str(x) for x in task.get('checklist', [])]

        judge_embedder = TfidfEmbedder() if args.judge_embedder == 'tfidf' else embedder
        task_quality, checklist_details = semantic_coverage(answer_text, checklist, judge_embedder, threshold=args.quality_threshold)
        continuity_text = answer_text + '\n' + '\n'.join([m.get('text', '') for m in memory_context])
        memory_recall, memory_details = semantic_coverage(continuity_text, expected_memory_text, judge_embedder, threshold=args.quality_threshold)
        evidence_recall = id_recall(selected_corpus_ids, expected_context_ids)
        score = composite_score(
            task_quality=task_quality,
            evidence_recall=evidence_recall,
            memory_recall=memory_recall,
            prompt_tokens=total_prompt_tokens,
            seconds=total_seconds,
            token_high=args.token_high,
            seconds_high=args.seconds_high,
        )
        row = {
            'trial': trial,
            'seed': args.seed,
            'mode': mode,
            'task_id': task['id'],
            'task_index': task.get('index'),
            'score': score,
            'task_quality': round(task_quality, 4),
            'evidence_recall': round(evidence_recall, 4),
            'memory_recall': round(memory_recall, 4),
            'prompt_tokens': total_prompt_tokens,
            'eval_tokens': total_eval_tokens,
            'seconds': round(total_seconds, 4),
            'selected_context_ids': selected_corpus_ids,
            'selected_memory_ids': selected_mem_ids,
            'current_task_note_ids': selected_transient_ids,
            'memory_size_after': len(memory),
            'answer': answer_text,
            'checklist_details': checklist_details,
            'memory_details': memory_details,
        }
        rows.append(row)
        append_jsonl(raw_path, row)
    write_jsonl(out_dir / f'{mode}_memory_trial_{trial:02d}.jsonl', memory)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description='Benchmark RAG vs DAG vs CAG with local Ollama.')
    p.add_argument('--model', default='qwen2.5-coder:7b', help='Ollama chat model')
    p.add_argument('--embed-model', default='nomic-embed-text', help='Ollama embedding model')
    p.add_argument('--ollama-host', default='http://localhost:11434')
    p.add_argument('--embedder', choices=['ollama', 'tfidf'], default='ollama')
    p.add_argument('--judge-embedder', choices=['same', 'tfidf'], default='tfidf')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--out-dir', default='results/run')
    p.add_argument('--modes', nargs='+', default=['rag', 'dag', 'cag'], choices=['rag', 'dag', 'cag'])
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--memory-top-k', type=int, default=5)
    p.add_argument('--memory-min-confidence', type=float, default=0.45)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--num-ctx', type=int, default=8192)
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--request-retries', type=int, default=1, help='Retries for transient Ollama request failures')
    p.add_argument('--retry-backoff', type=float, default=3.0, help='Base retry backoff seconds; doubles each retry')
    p.add_argument(
        '--continue-on-error',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Continue benchmark after per-task request failures; disable with --no-continue-on-error',
    )
    p.add_argument('--quality-threshold', type=float, default=0.42)
    p.add_argument('--token-high', type=int, default=12000)
    p.add_argument('--seconds-high', type=float, default=120.0)
    p.add_argument('--task-start', type=int, default=1, help='1-based task offset into data/tasks.jsonl')
    p.add_argument('--max-tasks', type=int, default=None, help='Maximum number of sequential tasks to run')
    p.add_argument('--repeat-runs', type=int, default=1, help='Number of independent trials per mode')
    p.add_argument('--dry-run', action='store_true', help='Run without Ollama for smoke testing only')
    args = p.parse_args()

    if args.repeat_runs < 1:
        raise ValueError('--repeat-runs must be 1 or greater')
    if args.max_tasks is not None and args.max_tasks < 1:
        raise ValueError('--max-tasks must be 1 or greater')
    if args.request_retries < 0:
        raise ValueError('--request-retries must be 0 or greater')
    if args.retry_backoff < 0:
        raise ValueError('--retry-backoff must be 0 or greater')

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = read_jsonl(data_dir / 'corpus.jsonl')
    all_tasks = read_jsonl(data_dir / 'tasks.jsonl')
    tasks = task_window(all_tasks, args.task_start, args.max_tasks)

    for mode in args.modes:
        raw_path = out_dir / f'{mode}_raw.jsonl'
        if raw_path.exists():
            raw_path.unlink()

    all_rows: list[dict[str, Any]] = []
    summary_path = out_dir / 'summary.csv'
    started = time.strftime('%Y-%m-%dT%H:%M:%S')
    meta = vars(args).copy()
    meta['started_at'] = started
    meta['task_ids'] = [t.get('id') for t in tasks]
    meta['task_count'] = len(tasks)
    meta['benchmark_note'] = 'Composite score is transparent: 40% checklist quality, 20% evidence recall, 25% continuity recall, 10% token efficiency, 5% latency efficiency. This benchmark is continuity-trap focused: later tasks require durable decisions introduced only in earlier task notes.'
    (out_dir / 'run_config.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    for trial in range(1, args.repeat_runs + 1):
        trial_args = copy.copy(args)
        trial_args.seed = args.seed + trial - 1
        print(f'Trial {trial}/{args.repeat_runs} using seed {trial_args.seed}')
        for mode in args.modes:
            print(f'Running {mode}...')
            all_rows.extend(run_mode(mode, trial_args, corpus, tasks, out_dir, trial))
            write_summary_csv(all_rows, summary_path)
    print(f'Wrote {summary_path}')


if __name__ == '__main__':
    main()
