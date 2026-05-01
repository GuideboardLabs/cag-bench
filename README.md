# cag-bench

A standalone Linux/Ollama benchmark for testing memory behavior over a 30-task iterative project.

This benchmark compares:

- `rag`: fresh retrieval only for the current task
- `dag`: fixed workflow structure with fresh retrieval for the current task
- CAG variants that accumulate and reuse durable project memory

CAG means Context Accumulation Generation: accepted project decisions from task metadata (`promote_summary` and `promote_terms`) are promoted into persistent memory and reused on later tasks. Model output is not promoted into durable memory — only validated decisions are.

No external routing, topic types, lane logic, or app-specific scaffolding are required.

## What CAG Variants Test

- `cag`: baseline persistent memory retrieval (unbounded memory dump)
- `cag_scoped`: deterministic top-K retrieval scored by `3.0*concept_overlap + 1.0*tag_overlap + 0.5*task_text_overlap + 1.0*recency_weight`
- `cag_oracle_memory`: top-K rows that pass the continuity-overlap filter, sorted by `concept_overlap` descending with creation-order tiebreak (diagnostic ceiling, not a fair baseline)

This setup is designed to expose late-task degradation and to localize whether retrieval or memory content is the bottleneck.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5-coder:7b
```

## Run Suites

### Baselines (default)

```bash
python -m cag_bench.run \
  --model qwen2.5-coder:7b \
  --out results/baselines \
  --max-tasks 30 \
  --repeat-runs 5 \
  --suite baselines
```

`baselines` runs:

- `rag`
- `dag`
- `cag`

### CAG Ablation

```bash
python -m cag_bench.run \
  --model qwen2.5-coder:7b \
  --out results/cag_ablation \
  --max-tasks 30 \
  --repeat-runs 5 \
  --suite cag-ablation
```

`cag-ablation` runs:

- `cag`
- `cag_scoped`
- `cag_oracle_memory`

### Full Matrix

```bash
python -m cag_bench.run \
  --model qwen2.5-coder:7b \
  --out results/full_matrix \
  --max-tasks 30 \
  --repeat-runs 5 \
  --suite all
```

`all` runs:

- `rag`
- `dag`
- `cag`
- `cag_scoped`
- `cag_oracle_memory`

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
| `--modes` | unset | Comma-separated mode list. Overrides `--suite`. |
| `--max-tasks` | `30` | Max number of tasks to execute after `--task-start`. |
| `--task-start` | `1` | 1-based task index offset. |
| `--repeat-runs` | `1` | Number of repeated trials. |
| `--retrieval-k` | `3` | Fresh source retrieval top-k. |
| `--memory-top-k` | `5` | Cap memory rows for `cag_scoped` and `cag_oracle_memory`. |
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

## Explicit Mode Override

`--modes` overrides `--suite`.

```bash
python -m cag_bench.run --modes rag,dag,cag
python -m cag_bench.run --modes cag,cag_scoped,cag_oracle_memory
```

## Dry Run Compatibility

Dry runs skip Ollama calls and exercise the full pipeline shape:

```bash
python -m cag_bench.run \
  --dry-run \
  --out results/dry_run \
  --max-tasks 30 \
  --repeat-runs 2
```

Or via the helper script:

```bash
./scripts/dry_run.sh
```

## Project Memory Model

For CAG variants, each task runs a post-task indexing pass that promotes durable memory from validated task metadata (`promote_summary` and `promote_terms`). Tasks include `expected_prior_concepts` (derived from continuity dependencies) for scoring only — they are not injected into the model prompt.

Promoted memory is reused on later tasks via either an unbounded dump (`cag`), deterministic scored retrieval (`cag_scoped`), or an oracle continuity-overlap filter (`cag_oracle_memory`).

## Scoring Method

Deterministic scoring is concept-group based:

- each concept has a `concept` label plus `accepted_terms`
- a concept counts as a hit if any accepted synonym appears in the answer
- plain string terms still work (treated as one-concept/one-term groups)

Evidence is split into:

- `source_evidence_recall`
- `domain_rule_recall`

Compatibility field `evidence_recall` is derived from both.

Contradictions are scored via `contradiction_terms` and subtracted from the composite:

- `contradiction_penalty`
- `contradiction_hit_count`
- `contradiction_hits` (in `raw.jsonl`)

When a metric bucket has no concepts for a row, that metric is `null` and excluded from the composite denominator (renormalized over present metrics).

## Composite Score Weights

`BASE_WEIGHTS` in `cag_bench/scoring.py`:

- 34% checklist quality
- 12% source evidence recall
- 12% domain rule recall
- 40% continuity recall
- 1% token efficiency
- 1% latency efficiency
- minus contradiction penalty

## Outputs

Per run folder:

- `summary.csv`
- `raw.jsonl` (concept-level breakdowns and `retrieval_scores` for `cag_scoped`)
- `failures.csv`
- `cag_memory_trial_XX.jsonl` per CAG trial
- core curves: `score_curve.png`, `checklist_quality_curve.png`, `source_evidence_recall_curve.png`, `domain_rule_recall_curve.png`, `evidence_recall_curve.png`, `continuity_recall_curve.png`, `token_efficiency_curve.png`, `latency_efficiency_curve.png`, `contradiction_penalty_curve.png`
- CAG internal curves: `memory_recall_curve.png`, `memory_precision_curve.png`, `continuity_per_memory_token_curve.png`
- `all_metrics_grid.png`
- `aggregated_metrics.csv`
- `phase_summary.csv`
- `SUMMARY.md`

Phase summaries include:

- tasks `1-10`
- tasks `11-20`
- tasks `21-30`
- tasks `20-30`
- task `30` only

## Inspect a Task

```bash
python -m cag_bench.inspect --raw results/<run>/raw.jsonl --task T30 --task T01
```

## Helper Scripts

```bash
./scripts/run_baselines.sh [model] [out_dir]
./scripts/run_cag_ablation.sh [model] [out_dir]
./scripts/dry_run.sh
./scripts/run_smoke.sh
./scripts/regen_graphs.sh <run_dir> [--trial-limit N] [--raw-trials|--no-raw-trials] [--theme light|dark]
```

`run_baselines.sh` and `run_cag_ablation.sh` accept extra flags after the first two arguments. `regen_graphs.sh` rebuilds plots and recomputes composite scores from an existing run folder.

## Interpretation Guidance

Diagnostic split to localize bottlenecks:

- oracle high, scoped low: retrieval is the bottleneck
- oracle low: memory content is weak or the model ignores memory

CAG modes are expected to spend more tokens than RAG/DAG due to memory context.

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
