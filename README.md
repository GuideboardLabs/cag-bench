# cag-bench v4

A Linux/Ollama benchmark for comparing RAG, DAG, and CAG over one progressing 30-task project.

CAG means Context Accumulation Generation: accepted project decisions are promoted into persistent memory and reused across later tasks.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5-coder:7b
./scripts/run_ollama.sh qwen2.5-coder:7b results/qwen_v4 --repeat-runs 5 --raw-trials
```

Judge mode (optional):

```bash
./scripts/run_ollama.sh qwen2.5-coder:7b results/qwen_v4_judge --repeat-runs 5 --raw-trials --judge --judge-model qwen2.5-coder:7b
```

Smoke test without Ollama:

```bash
./scripts/dry_run.sh
```

## Scoring Method

Baseline deterministic scoring is concept-group based (not raw keyword-only matching):

- each concept has a `concept` label plus `accepted_terms`
- a concept counts as a hit if any accepted synonym appears
- plain string terms still work (they are treated as one-concept/one-term groups)

Evidence is split into two metrics:

- `source_evidence_recall`
- `domain_rule_recall`

For compatibility, `evidence_recall` is still emitted and derived from both:

- average of source/domain when both exist
- the non-empty side when only one exists

Contradictions are scored via `contradiction_terms` and subtracted from the composite:

- `contradiction_penalty`
- `contradiction_hit_count`
- `contradiction_hits` (in `raw.jsonl`)

Deterministic score remains primary. Optional judge fields are secondary diagnostics only.

## Composite Score

- 30% checklist quality
- 15% source evidence recall
- 15% domain rule recall
- 30% continuity recall
- 5% token efficiency
- 5% latency efficiency
- minus contradiction penalty

## Outputs

- `summary.csv`
- `raw.jsonl`
- `score_curve.png`
- `checklist_quality_curve.png`
- `source_evidence_recall_curve.png`
- `domain_rule_recall_curve.png`
- `evidence_recall_curve.png`
- `continuity_recall_curve.png`
- `token_efficiency_curve.png`
- `latency_efficiency_curve.png`
- `contradiction_penalty_curve.png`
- `all_metrics_grid.png`
- `aggregated_metrics.csv`

`raw.jsonl` now includes concept-level breakdowns for inspection (`concept_hits` + `contradiction_hits`).

## Interpretation Guidance

The strongest CAG validation signal is still continuity recall.

CAG should not be expected to win token efficiency because it carries memory forward.

A strong CAG result typically shows:

- higher continuity recall
- competitive or better composite score
- tolerable token overhead
- low contradiction penalty
