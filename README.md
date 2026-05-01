# cag-bench v4

A Linux/Ollama benchmark for comparing retrieval and memory modes over one progressing 30-task project.

CAG means Context Accumulation Generation: accepted project decisions from task metadata (`promote_summary` and `promote_terms`) are promoted into persistent memory and reused later. Model output is not promoted into durable memory.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5-coder:7b
./scripts/run_baselines.sh qwen2.5-coder:7b results/qwen_v4 --repeat-runs 5 --raw-trials
```

CAG ablation suite:

```bash
./scripts/run_cag_ablation.sh qwen2.5-coder:7b results/qwen_v4_ablation --repeat-runs 5 --raw-trials
```

Dry run (no Ollama):

```bash
./scripts/dry_run.sh
```

Inspect a task from raw output:

```bash
python -m cag_bench.inspect --raw results/<run>/raw.jsonl --task T30 --task T01
```

## Mode and Suite Flags

`cag_bench.run` supports:

- `--suite baselines` (default): `rag,dag,cag`
- `--suite cag-ablation`: `cag,cag_scoped,cag_oracle_memory`
- `--suite all`: `rag,dag,cag,cag_scoped,cag_oracle_memory`
- `--modes mode1,mode2,...`: explicit mode list (cannot be combined with an explicit `--suite`)
- `--memory-top-k N` (default `5`): cap memory rows for `cag_scoped` and `cag_oracle_memory`

Mode behavior:

- `rag`: fresh retrieval only
- `dag`: fixed process baseline with fresh retrieval
- `cag`: baseline persistent memory retrieval (unbounded memory dump)
- `cag_scoped`: deterministic top-K CAG retrieval with score `3.0*concept_overlap + 1.0*tag_overlap + 0.5*task_text_overlap + 1.0*recency_weight`
- `cag_oracle_memory`: diagnostic top-K ceiling that peeks at continuity metadata (not a fair baseline)

## Scoring Method

Deterministic scoring is concept-group based:

- each concept has a `concept` label plus `accepted_terms`
- a concept counts as a hit if any accepted synonym appears
- plain string terms still work (treated as one-concept/one-term groups)

Evidence is split into:

- `source_evidence_recall`
- `domain_rule_recall`

Compatibility field `evidence_recall` is derived from both.

Contradictions are scored via `contradiction_terms` and subtracted from composite:

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

- `summary.csv`
- `raw.jsonl`
- `failures.csv`
- `phase_summary.csv`
- `score_curve.png`
- `checklist_quality_curve.png`
- `source_evidence_recall_curve.png`
- `domain_rule_recall_curve.png`
- `evidence_recall_curve.png`
- `continuity_recall_curve.png`
- `memory_recall_curve.png`
- `memory_precision_curve.png`
- `continuity_per_memory_token_curve.png`
- `token_efficiency_curve.png`
- `latency_efficiency_curve.png`
- `contradiction_penalty_curve.png`
- `all_metrics_grid.png`
- `aggregated_metrics.csv`
- `SUMMARY.md`

`raw.jsonl` includes concept-level breakdowns and scoped retrieval traces (`retrieval_scores`) for `cag_scoped`.

## Interpretation Guidance

Diagnostic split to localize bottlenecks:

- oracle high, scoped low: retrieval is the bottleneck
- oracle low: memory content is weak or the model ignores memory

CAG modes are expected to spend more tokens than RAG/DAG due to memory context.
