import argparse, csv, json, time
from pathlib import Path
import requests
from .ollama_client import OllamaClient, approx_tokens
from .retrieval import KeywordRetriever
from .memory import ProjectMemory
from .prompts import rag_prompt, dag_prompt, cag_prompt, dry_answer
from .scoring import metric_scores, composite_score, BASE_WEIGHTS
from .utils import load_jsonl, append_jsonl, coerce_concept_groups

def prompt_text(messages): return '\n\n'.join([m['role']+': '+m['content'] for m in messages])
def accepted_memory_summary(task): return task.get('promote_summary','')
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
        return json.loads(text[start:end+1])
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
    result = client.chat(model=judge_model, messages=judge_messages, temperature=0.0, num_ctx=num_ctx)
    parsed = _extract_json_object(result.get("content", ""))
    return {
        "judge_continuity_score": _clamp_score(parsed.get("judge_continuity_score")),
        "judge_actionability_score": _clamp_score(parsed.get("judge_actionability_score")),
        "judge_contradiction_score": _clamp_score(parsed.get("judge_contradiction_score")),
        "judge_notes": str(parsed.get("judge_notes", "")).strip() or None,
        "judge_raw": result.get("raw", {}),
    }

def validate_continuity_ladder(tasks):
    if not tasks:
        raise ValueError("No tasks loaded.")
    sorted_tasks = sorted(tasks, key=lambda t: int(t.get('index', 0)))
    if [t.get('id') for t in sorted_tasks] != [t.get('id') for t in tasks]:
        raise ValueError("Tasks must be sorted by ascending index.")
    if _norm_terms(tasks[0].get('continuity_terms', [])):
        raise ValueError(f"Task {tasks[0].get('id')} must be fresh and have empty continuity_terms.")

    for i, task in enumerate(tasks):
        expected_index = i + 1
        if int(task.get('index', 0)) != expected_index:
            raise ValueError(f"Task ordering invalid at {task.get('id')}: expected index {expected_index}, got {task.get('index')}.")
        if i == 0:
            continue
        continuity = _norm_terms(task.get('continuity_terms', []))
        missing_prior = []
        for prev in tasks[:i]:
            promoted = _norm_terms(prev.get('promote_terms', []))
            if not promoted:
                missing_prior.append(prev.get('id'))
                continue
            if not (promoted & continuity):
                missing_prior.append(prev.get('id'))
        if missing_prior:
            raise ValueError(
                f"Task {task.get('id')} continuity ladder violation: missing references to prior tasks {missing_prior}. "
                "Each task N>1 must include at least one promoted term from every prior task."
            )

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
    judge=False,
    judge_model=None,
):
    query=task['title']+'\n'+task['prompt']+'\n'+' '.join(task.get('tags',[])); sources=retriever.retrieve(query,k=retrieval_k)
    memory_rows = memory.retrieve(task) if (mode=='cag' and memory is not None) else []
    messages = rag_prompt(task,sources) if mode=='rag' else dag_prompt(task,sources) if mode=='dag' else cag_prompt(task,sources,memory_rows)
    ptext=prompt_text(messages); prompt_tokens=approx_tokens(ptext)
    if dry_run:
        answer=dry_answer(mode, task, memory_rows); latency=0.01; raw={'dry_run':True}
    else:
        result=client.chat(model=model,messages=messages,temperature=temperature,num_ctx=num_ctx); answer=result['content']; latency=result['latency_seconds']; raw=result['raw']
    metrics=metric_scores(answer, task, prompt_tokens, latency); score=composite_score(metrics)
    row = {
        "mode":mode,
        "task_id":task['id'],
        "task_index":task['index'],
        "task_title":task['title'],
        "score":score,
        **metrics,
        "prompt_tokens":prompt_tokens,
        "latency_seconds":latency,
        "retrieved_sources":[s.source_id for s in sources],
        "memory_items_used":len(memory_rows),
        "memory_task_ids_used":[r.get('task_id') for r in memory_rows],
        "answer":answer,
        "raw":raw,
    }
    if judge and not dry_run:
        row.update(judge_answer(client, task, answer, judge_model or model, num_ctx))
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
        "memory_items_used": 0,
        "memory_task_ids_used": [],
        "answer": "",
        "raw": {},
        "error": f"{type(err).__name__}: {err}",
    }

def main():
    p=argparse.ArgumentParser(description='Benchmark RAG vs DAG vs CAG over one iterative project task sequence.')
    p.add_argument('--model',default='qwen2.5-coder:7b'); p.add_argument('--ollama-url',default='http://localhost:11434'); p.add_argument('--tasks',default='data/tasks.jsonl'); p.add_argument('--sources',default='data/sources'); p.add_argument('--out',default='results/run')
    p.add_argument('--max-tasks',type=int,default=30); p.add_argument('--task-start',type=int,default=1); p.add_argument('--repeat-runs',type=int,default=1); p.add_argument('--retrieval-k',type=int,default=3); p.add_argument('--temperature',type=float,default=0.1); p.add_argument('--num-ctx',type=int,default=8192); p.add_argument('--dry-run',action='store_true'); p.add_argument('--raw-trials',action='store_true')
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--request-retries', type=int, default=1)
    p.add_argument('--retry-backoff', type=float, default=3.0)
    p.add_argument('--judge', action='store_true', help='Enable optional LLM-as-judge metrics in raw outputs')
    p.add_argument('--judge-model', default=None, help='Judge model; defaults to --model when --judge is enabled')
    p.add_argument('--run-id', default=None, help='Optional run folder name under --out')
    p.add_argument('--continue-on-error', action=argparse.BooleanOptionalAction, default=True)
    args=p.parse_args(); out_base=Path(args.out)
    run_id = args.run_id or time.strftime('%Y%m%d_%H%M%S')
    out = out_base / run_id
    out.mkdir(parents=True, exist_ok=True)
    tasks=[t for t in load_jsonl(args.tasks) if t['index']>=args.task_start][:args.max_tasks]
    validate_continuity_ladder(tasks)
    retriever=KeywordRetriever.from_sources_dir(args.sources); client=OllamaClient(args.ollama_url, timeout=args.timeout, retries=args.request_retries, retry_backoff=args.retry_backoff)
    summary_path=out/'summary.csv'; raw_path=out/'raw.jsonl'
    fieldnames=[
        'trial','mode','task_id','task_index','task_title','score',
        'checklist_quality','source_evidence_recall','domain_rule_recall','evidence_recall','continuity_recall',
        'token_efficiency','latency_efficiency','contradiction_penalty',
        'checklist_hit_count','checklist_total',
        'source_evidence_hit_count','source_evidence_total',
        'domain_rule_hit_count','domain_rule_total',
        'evidence_hit_count','evidence_total',
        'continuity_hit_count','continuity_total',
        'contradiction_hit_count',
        'prompt_tokens','latency_seconds','memory_items_used','retrieved_sources','memory_task_ids_used','error'
    ]
    run_config = vars(args).copy()
    run_config['started_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    run_config['resolved_out_dir'] = str(out)
    run_config['task_count'] = len(tasks)
    (out/'run_config.json').write_text(json.dumps(run_config, indent=2), encoding='utf-8')
    with open(summary_path,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames); writer.writeheader()
        for trial in range(1,args.repeat_runs+1):
            print(f"Trial {trial}/{args.repeat_runs}", flush=True)
            memory=ProjectMemory(out/f'cag_memory_trial_{trial:02d}.jsonl')
            for task_i, task in enumerate(tasks, start=1):
                for mode in ['rag','dag','cag']:
                    print(f"Running {mode} task {task_i}/{len(tasks)} ({task.get('id')})", flush=True)
                    try:
                        row=run_one(
                            mode,task,retriever,client,args.model,memory if mode=='cag' else None,
                            args.dry_run,args.temperature,args.num_ctx,args.retrieval_k,
                            judge=args.judge, judge_model=args.judge_model
                        )
                    except requests.exceptions.RequestException as exc:
                        print(f'WARN trial={trial} mode={mode} task={task.get("id")} request failed: {exc}')
                        row=failed_row(mode, task, exc)
                        if not args.continue_on_error:
                            raise
                    except Exception as exc:
                        print(f'WARN trial={trial} mode={mode} task={task.get("id")} unexpected error: {exc}')
                        row=failed_row(mode, task, exc)
                        if not args.continue_on_error:
                            raise
                    row['trial']=trial; append_jsonl(raw_path,row); writer.writerow({k:row.get(k) for k in fieldnames}); f.flush()
                memory.add(task, accepted_memory_summary(task))
    (out/'weights.json').write_text(json.dumps(BASE_WEIGHTS,indent=2),encoding='utf-8')
    try:
        from .plot import make_plots
        make_plots(summary_path,out,args.raw_trials)
        print(f'Wrote plots to {out}')
    except Exception as exc:
        print(f'WARN plotting skipped: {exc}')
    print(f'Wrote {summary_path}')
if __name__=='__main__': main()
