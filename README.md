# cag-bench

A standalone benchmark for testing memory behavior over a 30-task iterative project.

This benchmark compares:

- `rag`: fresh retrieval only for the current task
- `dag`: fixed workflow structure only for the current task
- CAG variants that accumulate and reuse durable project memory

No external routing, topic types, lane logic, or app-specific scaffolding are required.

## What CAG Variants Test

- `cag_naive`: append-and-retrieve memory baseline
- `cag_scoped`: scoped memory indexing + scored retrieval
- `cag_briefed`: scoped retrieval + rolling project briefs + memory compression
- `cag_full`: scoped retrieval + rolling briefs + memory compression + contradiction check/revision pass

This setup is designed to expose late-task degradation and show which CAG layer helps most.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Suites

### Baselines (default)

```bash
python -m cag_bench.run \
  --model qwen2.5-coder:3b-instruct \
  --out results/baselines \
  --max-tasks 30 \
  --repeat-runs 10 \
  --suite baselines
```

`baselines` runs:

- `rag`
- `dag`
- `cag_full`

### CAG Ablation

```bash
python -m cag_bench.run \
  --model qwen2.5-coder:3b-instruct \
  --out results/cag_ablation \
  --max-tasks 30 \
  --repeat-runs 5 \
  --suite cag-ablation
```

`cag-ablation` runs:

- `cag_naive`
- `cag_scoped`
- `cag_briefed`
- `cag_full`

### Full Matrix

```bash
python -m cag_bench.run \
  --model qwen2.5-coder:3b-instruct \
  --out results/full_matrix \
  --max-tasks 30 \
  --repeat-runs 10 \
  --suite all
```

`all` runs:

- `rag`
- `dag`
- `cag_naive`
- `cag_scoped`
- `cag_briefed`
- `cag_full`

## Full CLI Flag Reference (`python -m cag_bench.run`)

| Flag | Default | Description |
|---|---|---|
| `--model` | `qwen2.5-coder:7b` | Ollama chat model to benchmark. |
| `--ollama-url` | `http://localhost:11434` | Ollama server URL. |
| `--tasks` | `data/tasks.jsonl` | Task sequence JSONL file. |
| `--sources` | `data/sources` | Source markdown directory for fresh retrieval. |
| `--out` | `results/run` | Base output directory (run-id subfolder created inside). |
| `--run-id` | timestamp | Optional fixed run folder name under `--out`. |
| `--suite` | `baselines` | Mode bundle: `baselines`, `cag-ablation`, or `all`. |
| `--modes` | unset | Mode override. Supports comma and/or space forms (overrides `--suite`). |
| `--max-tasks` | `30` | Max number of tasks to execute after `--task-start`. |
| `--task-start` | `1` | 1-based task index offset. |
| `--repeat-runs` | `1` | Number of repeated trials. |
| `--retrieval-k` | `3` | Fresh source retrieval top-k. |
| `--temperature` | `0.1` | Generation temperature for model calls. |
| `--num-ctx` | `8192` | Context window (`num_ctx`) passed to Ollama. |
| `--dry-run` | `false` | Run without model calls (smoke/shape testing). |
| `--raw-trials` | `false` | Plot raw trial traces in addition to means. |
| `--timeout` | `600` | Request timeout seconds. |
| `--request-retries` | `1` | Retries for request failures. |
| `--retry-backoff` | `3.0` | Base backoff seconds (exponential). |
| `--continue-on-error` / `--no-continue-on-error` | `--continue-on-error` | Continue after task/mode errors or fail fast. |
| `--judge` | `false` | Enable optional LLM-as-judge metrics in raw outputs. |
| `--judge-model` | unset (`--model`) | Judge model override when `--judge` is enabled. |
| `--brief-update-every-tasks` | `5` | Rolling brief update interval. |
| `--brief-trigger-tokens` | `1200` | Refresh briefs when memory token load exceeds threshold. |
| `--compression-trigger-tokens` | `2600` | Trigger memory compression threshold. |
| `--max-project-brief-tokens` | `350` | Budget for brief section in prompt context. |
| `--max-scoped-memory-tokens` | `900` | Budget for scoped durable memory in prompt context. |
| `--max-recent-memory-tokens` | `250` | Budget for recent memory lane in prompt context. |
| `--max-source-evidence-tokens` | `1000` | Budget for fresh retrieved sources in prompt context. |
| `--max-total-context-tokens` | `2600` | Overall context budget cap used by CAG builder. |
| `--cag-semantic-backend` | `tfidf` | Scoped retrieval semantic backend: `tfidf` or `ollama`. |
| `--cag-semantic-embed-model` | `nomic-embed-text` | Ollama embedding model when semantic backend is `ollama`. |

## Explicit Mode Override

`--modes` overrides `--suite`.

```bash
python -m cag_bench.run --modes rag,dag,cag_full
python -m cag_bench.run --modes cag_naive,cag_scoped,cag_briefed,cag_full
python -m cag_bench.run --modes rag dag cag_full
```

Semantic retrieval backend for scoped/briefed/full:

```bash
python -m cag_bench.run --suite cag-ablation --cag-semantic-backend tfidf
python -m cag_bench.run --suite cag-ablation --cag-semantic-backend ollama --cag-semantic-embed-model nomic-embed-text
```

## Dry Run Compatibility

The existing dry-run workflow still works:

```bash
python -m cag_bench.run \
  --dry-run \
  --out results/dry_run \
  --max-tasks 30 \
  --repeat-runs 2
```

## Standalone CAG Memory System

For scoped/briefed/full variants, each task runs a post-task indexing pass that extracts and validates durable memory candidates.

Tasks include `expected_prior_concepts` (derived from continuity dependencies) for scoring only. They are not injected into the model prompt.

Durable memory schema:

```json
{
  "memory_id": "string",
  "source_task_id": "T03",
  "created_at_task_index": 3,
  "status": "active",
  "scope": "global_project | feature | module | decision | test | safety | temporary",
  "type": "architecture | domain_rule | implementation_decision | testing_rule | safety_rule | naming_convention | known_issue | open_question | task_summary",
  "text": "durable memory text",
  "tags": ["sqlite", "persistence", "dogprofile"],
  "entities": ["DogProfile", "SessionLog"],
  "depends_on": ["memory_id"],
  "supersedes": [],
  "confidence": 0.0,
  "specificity": 0.0,
  "reuse_count": 0
}
```

Promotion filter actions:

- `promote`
- `update_existing`
- `supersede_existing`
- `keep_task_local`
- `discard`

`cag_full` adds contradiction check + single revision pass on top of scoped retrieval, briefs, and compression.

## Context Budgeting

CAG context builder uses explicit budgets:

- `max_project_brief_tokens`
- `max_scoped_memory_tokens`
- `max_recent_memory_tokens`
- `max_source_evidence_tokens`
- `max_total_context_tokens`

Raw outputs track:

- `memory_tokens_used`
- `brief_tokens_used`
- `source_tokens_used`
- `total_context_tokens`

## New Metrics

Alongside existing metrics:

- `memory_precision`
- `memory_recall`
- `memory_items_used`
- `memory_tokens_used`
- `brief_tokens_used`
- `source_tokens_used`
- `continuity_per_memory_token`
- `deprecated_memory_used`
- `irrelevant_memory_used`
- `contradictions_before_revision`
- `contradictions_after_revision`
- `memory_reuse_count`

## Outputs

Per run folder:

- `summary.csv`
- `raw.jsonl`
- `<mode>_memory_trial_XX.jsonl` for each CAG mode
- `<mode>_memory_trial_XX_briefs.jsonl` brief snapshots for briefed/full variants
- core curves (score/checklist/evidence/continuity/token/latency/contradiction)
- CAG internal curves (memory precision/recall, continuity per memory token, memory/brief/source token usage, contradiction before/after revision)
- `all_metrics_grid.png`
- `aggregated_metrics.csv`
- `SUMMARY.md`
- `phase_summary.csv`
- `phase_summary.md`
- `ablation_summary_late_stage.csv`
- `ablation_delta_summary.csv`
- `ablation_delta_summary.md`

Phase summaries include:

- tasks `1-10`
- tasks `11-20`
- tasks `21-30`
- tasks `20-30`
- task `30` only

## Helper Scripts

```bash
./scripts/run_baselines.sh [model] [out_dir]
./scripts/run_cag_ablation.sh [model] [out_dir]
./scripts/run_all.sh [model] [out_dir]
```

Each script accepts extra flags after the first two arguments.


## License and attribution

CAG Bench is licensed under the GNU Affero General Public License version 3 or later.

SPDX-License-Identifier: AGPL-3.0-or-later

You are free to use, study, modify, and redistribute this project under the terms of the AGPL-3.0-or-later license.

Please preserve attribution to the original creator:

```text
CAG Bench was originally created by Seth Canfield of Guideboard Labs.
```

If you use this benchmark in a post, paper, report, benchmark comparison, fork, or derivative project, please cite the project using the included `CITATION.cff` file.

### Important note

AGPL-3.0-or-later is an open source license and does not prohibit commercial use. Its protection comes from strong copyleft obligations, including source-sharing obligations for modified versions under the license terms.