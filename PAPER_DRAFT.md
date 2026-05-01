# Benchmarking Persistent Project Memory in Local Language Models: CAG vs RAG vs DAG over 30 Sequential Tasks

**Seth Canfield, Guideboard Labs**

---

## Abstract

We present CAG-Bench, a longitudinal benchmark for evaluating how well local language models maintain project context across a sequence of 30 iterative software development tasks. We compare three retrieval paradigms — Retrieval-Augmented Generation (RAG), Directed-Acyclic-Graph workflow generation (DAG), and Context Accumulation Generation (CAG) — along with two selective retrieval variants of CAG. On `qwen2.5-coder:7b`, CAG achieves a composite score of 48.5 vs. 29.6 (RAG) and 28.5 (DAG), with continuity recall of 54.2% vs. 17.1% and 17.0% respectively. The advantage is sustained across both a 7B and a 3B instruction-tuned model. Within the CAG family, a deterministic top-K scoped retrieval variant (`cag_scoped`) reaches an oracle memory ceiling at K=5, revealing that retrieval scoring is not the primary bottleneck — the K capacity cap and memory uptake are. We introduce `memory_usage_rate` as a new diagnostic metric that quantifies whether retrieved memory actually reaches the final answer. This rate declines monotonically across task phases (57.8% → 47.5% → 38.8% for base CAG), identifying memory uptake as the dominant unsolved challenge. All memory is grounded in task-defined facts; model output is never promoted into durable memory. Benchmark data, prompts, scoring code, and raw outputs are released under AGPL-3.0-or-later.

---

## 1. Introduction

Large language models performing software development tasks face a challenge that single-turn benchmarks cannot capture: context continuity across a long-running project. A model asked to implement a "Review lane with ProgressSnapshot" on task 28 must recall that ProgressSnapshot was defined on task 11, that the persistence layer uses SQLite with soft-delete semantics from task 3, and that the project explicitly rejects cloud-sync from task 1. No single retrieved document contains all of this; it is accumulated project knowledge.

Existing benchmarks measure isolated task quality. They do not measure whether a model-assisted workflow degrades over time as context grows, whether retrieval-based approaches can compensate for that growth, or where the failure modes lie when they cannot.

CAG-Bench is designed to answer these questions precisely. It presents a model with 30 sequential, explicitly interdependent tasks — each task's correct solution requires recalling decisions made in prior tasks. The benchmark compares three fundamentally different context strategies, instruments them with five distinct memory-quality metrics, and produces per-phase diagnostics to localize where each strategy succeeds and fails.

The benchmark is local-only (Ollama), deterministic in its scoring, and runs without external services. All memory is grounded: durable project facts are promoted from task-defined ground truth, not from model output. This design prevents the most common failure mode in memory-augmented generation — hallucinated facts becoming trusted context.

---

## 2. Background

### 2.1 RAG and its limits in iterative work

Retrieval-Augmented Generation retrieves relevant documents from a static corpus at each query. For iterative software development, the limitation is structural: prior model decisions are not documents. When the model chooses SQLite over PostgreSQL on task 3, that choice lives in a prior model response — not in a source file. RAG cannot retrieve it unless it is explicitly stored and indexed.

### 2.2 CAG: Context Accumulation Generation

CAG addresses this by maintaining a growing store of project decisions, accumulating accepted decisions after each task and providing them as context for subsequent tasks. The core principle: **accepted project decisions from ground-truth task metadata are promoted into persistent memory and reused on later tasks**. Model output is not directly promoted; only task-defined `promote_summary` and `promote_terms` fields enter durable memory. This grounding contract ensures the memory store contains only validated facts.

### 2.3 Selective retrieval within CAG

As the memory store grows across 30 tasks, indiscriminate dumping of all accumulated context becomes expensive and may overwhelm a model's effective context window. We evaluate two retrieval strategies within CAG:

- **`cag_scoped`**: Deterministic top-K scoring using `3.0 × concept_overlap + 1.0 × tag_overlap + 0.5 × task_text_overlap + 1.0 × recency_weight`
- **`cag_oracle_memory`**: A diagnostic upper bound that uses the task's known expected prior concepts to filter and rank memory rows. The model sees only the selected memory text — the scoring key is never exposed.

---

## 3. Related Work

### 3.1 RAG evaluation frameworks

The RAGAS framework (Es et al., 2023) provides the most widely adopted suite for evaluating retrieval-augmented pipelines. It defines four per-call metrics — faithfulness, answer relevancy, context precision, and context recall — which are computed independently for each query and averaged. Typical well-tuned RAG systems report faithfulness in the 0.75–0.90 range and context precision in the 0.55–0.80 range, depending on retrieval quality and corpus size. RAGAS does not define a composite score with fixed weights; each metric is reported independently. Crucially, RAGAS is designed for single-turn or independent-query evaluation. It has no notion of accumulated context, no continuity across calls, and no mechanism for measuring whether prior decisions from query N are honored at query N+28. CAG-Bench's composite weighting (40% continuity recall, 34% checklist quality, 12% each source evidence and domain rule recall) reflects a deliberate choice to prioritize longitudinal coherence over single-call retrieval fidelity.

TruLens (Truera, 2023) extends similar principles with LLM-as-judge groundedness and relevance scoring. Like RAGAS, it is oriented toward retrieval quality on a per-call basis and does not address context accumulation across a task sequence.

### 3.2 Long-context comprehension benchmarks

LongBench (Bai et al., 2023) evaluates long-context comprehension across 21 tasks in six categories, including single-document QA, multi-document QA, and code completion. Reported means across tasks: GPT-3.5-Turbo 44.0, GPT-4 54.4, Llama-2-13B-Chat 24.4. LongBench v2 (2024) extends this to 503 challenging multiple-choice questions with contexts up to 2M words, including long-dialogue history understanding. Both versions establish that model capability degrades on tasks requiring information from early portions of long documents, particularly for smaller models. However, LongBench provides the long context as a given input — the model is a passive reader of a pre-assembled document. CAG-Bench differs fundamentally: there is no pre-assembled document. The model must perform tasks in sequence, with each task's correct answer depending on what was decided in prior model interactions. The challenge is not reading a long document; it is accumulating a coherent project history across independent calls.

LongMemEval (Wu et al., 2024; ICLR 2025) is the most directly relevant long-context benchmark. It evaluates five memory abilities — information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention — across 500 questions embedded within scalable chat histories of up to 1.5M tokens. Key result: commercial chat assistants show a 30% accuracy drop on memorizing information across sustained interactions, and long-context LLMs including GPT-4o show a 30–60% performance decline compared to oracle retrieval settings. This directly parallels CAG-Bench's finding that RAG/DAG continuity recall collapses from ~32% to ~6% across the three task phases. The critical difference: LongMemEval tests factual Q&A retrieval from chat history; CAG-Bench tests decision integration into actively generated software development plans. The output is an action, not a lookup.

MemoryAgentBench (Hu et al., 2025; ICLR 2026) evaluates memory agents across four competencies derived from cognitive science: accurate retrieval, test-time learning, long-range understanding, and selective forgetting. It reconstructs existing long-context datasets into incremental multi-turn formats. This is the closest structural parallel to CAG-Bench in the memory literature — both benchmark memory across sequences of incremental inputs. The distinction is domain: MemoryAgentBench evaluates conversational and factual memory; CAG-Bench evaluates whether project-level architectural decisions made in prior development tasks are honored in subsequent ones. Additionally, MemoryAgentBench allows model-generated memory; CAG-Bench's grounding constraint prevents unvalidated model output from entering durable memory, enabling cleaner isolation of retrieval and uptake as independent failure modes.

SCROLLS (Shaham et al., 2022) and ∞BENCH (Zhang et al., 2024) address comprehension at greater length. ∞BENCH reports GPT-4 performance degrading sharply beyond 100K tokens in retrieval tasks. These benchmarks treat context as externally provided input; none model the accumulation problem.

### 3.3 Memory-augmented generation systems

MemGPT (Packer et al., 2023) introduces OS-inspired hierarchical memory management for LLMs, with explicit paging between main context and external storage. On multi-session conversation and document analysis tasks exceeding a model's context window, MemGPT substantially outperforms fixed-context baselines. The key architectural difference from CAG-Bench's design: MemGPT allows the model to manage memory operations — deciding what to store, retrieve, and evict. CAG-Bench's grounding constraint takes the opposite position: memory content comes from task-defined facts, not model output. This is a deliberate benchmark choice to isolate retrieval and uptake problems from memory formation quality; it does not imply model-managed memory is wrong for production use.

Generative Agents (Park et al., 2023) introduce the retrieve-reflect-plan memory loop. MemoryBank (Zhong et al., 2023) applies Ebbinghaus forgetting curve mechanics to LLM memory and reports ~15–20% improvement in personality and fact consistency over vanilla ChatGPT. Neither addresses the grounding problem — both allow model-generated content to enter persistent memory without validation. The downstream risk (confirmed empirically in earlier versions of this benchmark's development) is that hallucinated technical choices — MongoDB, JWT, cloud sync — can become trusted project facts. PersistBench (2026) takes this concern further, evaluating cross-domain memory leakage and memory-induced sycophancy in persistent memory systems, quantifying failure modes that arise specifically when model-generated memories are stored without validation.

### 3.4 Iterative software engineering benchmarks

SWE-bench (Jimenez et al., 2023) evaluates LLMs on resolving real GitHub issues against a test suite. Each issue is evaluated independently: the model receives the repository context as a snapshot without accumulated interaction history. Early GPT-4 results were approximately 1.7–2.7% resolution rate; top systems now exceed 49% on the Verified subset. SWE-bench measures capability to modify existing code; CAG-Bench measures capability to make consistent incremental decisions across a project lifecycle where no codebase exists yet.

SWE-EVO (Nguyen et al., 2024; arXiv:2512.18470) is a significant recent advance: it benchmarks coding agents on long-horizon software evolution, comprising 48 tasks spanning an average of 21 files across seven mature open-source Python projects. GPT-5.4 with OpenHands achieves only 25% on SWE-EVO versus 72.8% on SWE-bench Verified, demonstrating that state-of-the-art agents fail dramatically on sustained multi-file reasoning. CAG-Bench and SWE-EVO are complementary: SWE-EVO tests evolution of existing codebases against real test suites; CAG-Bench tests accumulation of project decisions from scratch with concept-group scoring. Neither answers the other's question.

Most directly parallel is **SlopCodeBench** (arXiv:2603.24755, 2026), which evaluates how coding agents degrade over 20 problems and 93 checkpoints as specifications evolve and agents carry their own prior code forward. Key findings: no agent solves any problem end-to-end across 11 models; the highest checkpoint solve rate is 17.2%; agent code is 2.2× more verbose than comparable open-source projects and degrades structurally faster than human code. This confirms CAG-Bench's central finding — that iterative context degrades model performance — from the code quality angle. The distinction is dimension: SlopCodeBench measures structural code degradation (verbosity, complexity erosion); CAG-Bench measures decision continuity (whether project-level commitments are honored in subsequent outputs). Both findings are necessary; neither is sufficient alone.

HumanEval (Chen et al., 2021) and DevBench (Li et al., 2024) evaluate isolated or single-pass code generation and are entirely stateless with respect to project history.

### 3.5 Where CAG-Bench stands

Recent work establishes the iterative degradation problem from multiple angles: LongMemEval shows 30–60% accuracy loss in conversational memory; SWE-EVO shows a 47-point gap between single-issue and multi-evolution performance; SlopCodeBench shows no agent completes an iterative specification sequence end-to-end. CAG-Bench contributes three things these works do not: (1) a controlled comparison of retrieval paradigms (RAG, DAG, CAG, scoped, oracle) on the same task sequence; (2) the `memory_usage_rate` metric that separates retrieval failure from uptake failure; and (3) a grounding contract that prevents model hallucinations from polluting the memory store, enabling clean measurement of retrieval and uptake independently of memory formation quality.

---

## 4. Benchmark Design

### 3.1 Task structure

The benchmark uses 30 sequential tasks describing an iterative software project (`PawLedger`, a local-first dog training workspace). Tasks are explicitly interdependent: each task from T02 onward contains `continuity_terms` — concepts that must appear in a correct answer, derived from decisions made in prior tasks. Task 1 has no continuity requirements. By task 30, a correct answer must reference 31 distinct prior concepts.

Each task also carries:
- `promote_summary`: the canonical one-sentence project decision that this task establishes
- `promote_terms`: the key technical terms associated with that decision
- `checklist_terms`: implementation checklist items expected in the answer
- `source_evidence_terms`: concepts from the source document corpus
- `domain_rule_terms`: domain-specific constraints (e.g. "humane training", "no cloud sync")
- `contradiction_terms`: terms that indicate a model invented something that conflicts with the project spec

### 3.2 Memory model

After each task runs, `ProjectMemory.add()` writes one row to a per-trial JSONL store using the task's `promote_summary` as the memory text. This promotion is deterministic and auditable: a companion `_promotions.jsonl` file records every promotion decision with its reason (`supported_by_promote_summary` or `task_local`). No model output is ever promoted. In all runs reported here, 30/30 tasks produce a `promote` decision since all tasks carry a non-empty `promote_summary`.

### 3.3 Modes

| Mode | Memory source | Selection strategy | Token budget (T30 mean) |
|---|---|---|---|
| `rag` | None | Top-3 keyword source retrieval | 418 |
| `dag` | None | Fixed workflow prompt + top-3 retrieval | 448 |
| `cag` | Full accumulation | Top-K by keyword score, cap=25 | 1,850 |
| `cag_scoped` | Full accumulation | Top-5 by composite score | 692 |
| `cag_oracle_memory` | Full accumulation | Top-5 by continuity-overlap filter | 692 |

### 3.4 Scoring

All scoring is deterministic and concept-group based. Each term group has a canonical label and a set of accepted synonyms. A concept counts as a hit if any accepted synonym appears in the model answer; plain string terms are treated as singleton groups.

**Composite score:**
```
score = 0.34 × checklist_quality
      + 0.12 × source_evidence_recall
      + 0.12 × domain_rule_recall
      + 0.40 × continuity_recall
      + 0.01 × token_efficiency
      + 0.01 × latency_efficiency
      − contradiction_penalty
```

Continuity recall carries the highest weight (40%) because it measures the core benchmark property: are prior project decisions present in the current answer?

### 3.5 Memory diagnostic metrics (new in this work)

Four metrics instrument the memory pipeline at distinct stages:

| Metric | Measures | Stage |
|---|---|---|
| `memory_recall` | % of task's expected concepts present in selected memory | Retrieval |
| `memory_precision` | % of selected memory concepts relevant to current task | Retrieval |
| `continuity_recall` | % of expected concepts appearing in final answer | Output |
| `memory_usage_rate` | % of selected memory concepts appearing in final answer | Uptake |

`memory_usage_rate` is introduced in this work. It bridges the gap between `memory_recall` (what was retrieved) and `continuity_recall` (what appeared in the answer), enabling diagnosis of whether retrieval is the bottleneck or whether retrieved memory is being ignored by the model.

---

## 4. Experimental Setup

All experiments run locally via Ollama on the same hardware. Two models are evaluated:

- `qwen2.5-coder:7b` — primary model
- `qwen2.5-coder:3b-instruct` — secondary, lower-capacity model

Fixed parameters across all runs: temperature 0.1, context window 8192 tokens, source retrieval K=3 (keyword-based). Memory top-K defaults: 5 for `cag_scoped`/`cag_oracle_memory`, 25 for base `cag`. All 30 tasks are run in sequence; each trial is a fresh independent pass through all 30 tasks with a fresh memory store.

| Run | Model | Suite | Tasks | Trials |
|---|---|---|---|---|
| Baseline 7B | qwen2.5-coder:7b | rag, dag, cag | 30 | 3 |
| Baseline 3B | qwen2.5-coder:3b-instruct | rag, dag, cag | 30 | 10 |
| CAG Ablation 7B | qwen2.5-coder:7b | cag, cag_scoped, cag_oracle | 30 | 3 |

---

## 5. Results

### 5.1 RAG vs DAG vs CAG: baseline comparison

**Table 1. Overall means across 30 tasks (qwen2.5-coder:7b, 3 trials)**

| Mode | Composite | Continuity | Checklist | Src. Evidence | Domain Rule | Contradiction | Tokens |
|---|---|---|---|---|---|---|---|
| RAG | 29.6 | 17.1 | 33.1 | 37.3 | 47.2 | 1.50 | 374 |
| DAG | 28.5 | 17.0 | 33.7 | 40.0 | 41.7 | 2.00 | 404 |
| CAG | **48.5** | **54.2** | **42.4** | 38.1 | **56.4** | 1.00 | 1,087 |

CAG outperforms RAG by +18.9 composite points and +37.1 continuity points. The pattern is nearly identical at 3B scale (RAG 28.0, DAG 28.6, CAG 45.7), establishing that the effect is not model-size-specific.

Two findings stand out beyond the headline numbers. First, **source evidence recall is essentially equivalent across all three modes** (37–40%), confirming that all modes retrieve relevant documentation similarly well — CAG's advantage is not from better source access. Second, **domain rule recall is substantially higher for CAG** (56.4 vs. 47.2 for RAG on the 7B model; 59.8 vs. 40.3 on 3B). Domain rules are project-specific constraints that are encoded in the accumulated memory decisions. The model is more reliably applying them when they are explicitly carried in context.

DAG does not improve over RAG. The structured workflow prompt adds token overhead (+8%) without improving any recall metric, and shows the highest mean contradiction penalty (2.00). This suggests that fixed-structure prompting may interfere with task-specific reasoning without compensating through better context access.

**Table 2. Phase breakdown — composite score and continuity recall (qwen2.5-coder:7b)**

| Phase | RAG score | DAG score | CAG score | RAG cont. | DAG cont. | CAG cont. |
|---|---|---|---|---|---|---|
| T01–T10 | 44.6 | 40.5 | 57.8 | 31.8 | 31.0 | 70.7 |
| T11–T20 | 31.8 | 30.5 | 56.7 | 15.1 | 15.8 | 59.0 |
| T21–T30 | 12.4 | 14.6 | 30.9 | 5.9 | 5.6 | 34.6 |

Two patterns are notable here. First, **RAG and DAG degrade sharply and symmetrically** — by the final 10 tasks, both modes score below 15 and continuity drops to ~6%. Fresh retrieval cannot compensate for the growth in expected prior concepts: at task 28, a correct answer needs to reference 28+ decisions that live exclusively in the accumulated project history.

Second, **CAG's largest relative advantage occurs in the middle phase (T11–T20)**, not at the end. CAG leads RAG by +13.2 points in T01–T10, widens to +24.9 in T11–T20, then narrows to +18.5 in T21–T30. The middle-phase peak suggests that accumulated context is most effective once a meaningful project history has formed but before the memory store becomes large enough to dilute the model's attention.

![Figure 1. Composite score by task index — RAG and DAG collapse monotonically while CAG holds a substantially higher floor through all 30 tasks.](results/qwen7b_baseline_smoke/20260501_074401/score_curve.png)

![Figure 2. Continuity recall by task index — the metric most directly measuring prior-decision coverage. RAG and DAG approach zero by task 20; CAG holds a meaningful floor.](results/qwen7b_baseline_smoke/20260501_074401/continuity_recall_curve.png)

At task 30 specifically — the final implementation handoff task — the gap is stark: CAG achieves 46.0% continuity recall vs. 9.2% for RAG and 4.6% for DAG. This is the task where all 30 prior decisions are simultaneously required. Even CAG does not reach the theoretical maximum, but it is the only mode that provides a practically useful response.

### 5.2 CAG internal ablation

Having established CAG's advantage over retrieval-only approaches, we examine whether selective retrieval improves over CAG's default generous-dump strategy.

**Table 3. CAG ablation — overall means (qwen2.5-coder:7b, 3 trials, K=5 for scoped/oracle)**

| Mode | Composite | Continuity | Memory Recall | Memory Precision | Usage Rate | Tokens |
|---|---|---|---|---|---|---|
| `cag` (cap=25) | 47.3 | 53.3 | 100.0 | 39.3 | 47.7 | 1,072 |
| `cag_scoped` (K=5) | 44.3 | 51.2 | 80.7 | 53.0 | 53.1 | 612 |
| `cag_oracle` (K=5) | 42.6 | 48.7 | 80.7 | 53.0 | 52.2 | 612 |

The most striking finding from the ablation is that **`cag_scoped` and `cag_oracle_memory` produce statistically identical memory_recall and memory_precision values** — both 80.7% and 53.0% across all phases. The oracle mode, which uses ground-truth expected concepts to filter memory, does not outperform the deterministic scoring formula. This means the scoring formula has effectively saturated the oracle ceiling at K=5.

This convergence is structurally explained: the benchmark's continuity ladder ensures that every task's `promote_terms` overlap significantly with the next task's `continuity_terms`. Scoring by `concept_overlap` — which checks for those very terms — reliably identifies the same rows that oracle filtering selects. The retrieval formula is not the bottleneck.

**The binding constraint is K.** Late tasks require 20–31 prior concept recalls simultaneously. A K=5 cap can cover at most the concepts contained in five memory rows; as the store grows past 10–15 rows, both scoped and oracle plateau at 62% memory_recall in tasks T21–T30 while base CAG (cap=25) holds 100%.

**Table 4. Phase breakdown — CAG ablation (memory recall and continuity)**

| Phase | CAG mem_recall | Scoped mem_recall | Oracle mem_recall | CAG cont. | Scoped cont. | Oracle cont. |
|---|---|---|---|---|---|---|
| T01–T10 | 100.0 | 99.2 | 99.2 | 69.7 | 71.1 | 68.9 |
| T11–T20 | 100.0 | 82.6 | 82.6 | 56.2 | 52.8 | 46.9 |
| T21–T30 | 100.0 | 62.1 | 62.1 | 35.6 | 31.8 | 32.2 |

Despite holding 100% memory_recall, base CAG's continuity recall declines significantly in the late phase (35.6%), confirming that concept availability in the prompt does not guarantee concept appearance in the answer. A high-coverage dump strategy is necessary but not sufficient.

`cag_scoped` achieves slightly better continuity than `cag_oracle` in early and middle phases (71.1 vs. 68.9 at T01–T10; 52.8 vs. 46.9 at T11–T20), suggesting that the recency and tag-overlap components of the scoped formula provide marginally better practical retrieval despite the oracle's theoretically superior selection. By the late phase, both converge.

Token cost is a real consideration: base CAG uses 1,072 tokens per call on average, growing to 1,850 at task 30. `cag_scoped` uses 612 tokens throughout (the K=5 cap produces a near-flat token budget), representing a 43% reduction in context cost. If memory_recall could be improved through a higher K or better formula, scoped retrieval would be clearly preferable on efficiency grounds.

![Figure 3. Memory recall by task index across CAG variants — base CAG holds 100% throughout (unbounded store); scoped and oracle converge at ~62% by tasks 21–30 as the K=5 cap becomes the binding constraint.](results/qwen7b_variants_smoke/20260501_130904/memory_recall_curve.png)

![Figure 4. Memory precision by task index — scoped and oracle maintain higher precision than base CAG (53% vs 39% overall) by filtering to task-relevant rows, at the cost of lower coverage.](results/qwen7b_variants_smoke/20260501_130904/memory_precision_curve.png)

### 5.3 Memory uptake: a new diagnostic axis

The most actionable finding from this benchmark concerns `memory_usage_rate`: the fraction of selected memory concepts that actually appear in the final answer.

**Table 5. Memory usage rate by phase (qwen2.5-coder:7b, CAG ablation)**

| Phase | CAG usage | Scoped usage | Oracle usage |
|---|---|---|---|
| T01–T10 | 57.8% | 60.9% | 60.4% |
| T11–T20 | 47.5% | 53.2% | 47.2% |
| T21–T30 | 38.8% | 46.1% | 50.0% |

Memory usage rate declines across all three modes as task complexity increases. For base CAG, which provides the highest memory coverage (100% recall), less than 40% of the retrieved concepts make it into the final answer in the late phase. This means that even when the correct prior decisions are present in the prompt, the model is not consistently using them.

The gap between `memory_recall` and `continuity_recall` quantifies the uptake deficit. For base CAG at the overall level, this gap is approximately −47 percentage points (100.0% recall, 53.3% continuity). For `cag_scoped`, the gap is −29 points. **The model is not the bottleneck when it comes to finding information in the prompt; it is the bottleneck when it comes to using that information in the output.**

Notably, `cag_scoped` consistently outperforms base CAG on memory usage rate despite having lower memory recall. Presenting five targeted memory items appears more effective at driving uptake than presenting twenty-five items that include the five relevant ones. This is consistent with attention dilution: a model given a large context block containing the relevant items alongside many less-relevant items is less likely to reference the most relevant items than one given only the five most relevant.

![Figure 5. Memory usage rate by task index — the fraction of selected memory concepts that appear in the final answer. All three modes decline over time; base CAG shows the steepest fall despite holding 100% memory recall, consistent with attention dilution under growing context load.](results/qwen7b_variants_smoke/20260501_130904/memory_usage_rate_curve.png)

![Figure 6. Continuity per memory token — efficiency of context spend. Scoped retrieval is roughly 3× more efficient than base CAG in the late phase, sustaining similar continuity output at one-third the token cost.](results/qwen7b_variants_smoke/20260501_130904/continuity_per_memory_token_curve.png)

---

## 6. Discussion

### 6.1 The value of accumulated project memory

The primary question CAG-Bench was designed to answer — does accumulated project memory meaningfully improve model performance over a long iterative task sequence? — receives a clear positive answer. A +18.9 composite point advantage for 7B models, replicated at 3B scale with comparable magnitude, is robust and practically significant. The advantage is not an artifact of prompt design: source retrieval recall is flat across modes, and checklist quality (which measures immediate task-specific content, not continuity) shows a smaller but consistent CAG advantage (+9.3 points), confirming that memory improves both types of recall.

The T30 result is particularly illustrative. At the final task — produce a complete implementation handoff for a 30-decision project — RAG achieves 9.2% continuity recall and a composite score of 13.0. CAG achieves 46.0% continuity recall and 30.7 composite. This is not a marginal improvement. It represents the difference between an assistant that can describe the project from its accumulated context and one that can only describe what appears in a single retrieved source document.

### 6.2 Why RAG and DAG fail on long sequences

RAG's failure mode is structural. By task 20, the benchmark expects a model answer to reference 20 prior decisions, none of which are in any source document. Fresh retrieval cannot produce what was never written. RAG's continuity recall of 5.9% in T21–T30 is effectively near-zero — the model has no mechanism to access those decisions.

DAG's failure is more subtle. The structured workflow prompt does not degrade faster than RAG (both reach ~6% continuity at T21–T30), but DAG consistently shows higher contradiction penalties (2.00 vs. 1.50 for RAG) and does not improve source evidence recall over RAG despite using a more structured prompt. Workflow structure without memory cannot compensate for the absence of project context.

### 6.3 Where CAG falls short

CAG's late-phase degradation (from 70.7% continuity at T01–T10 to 34.6% at T21–T30) is meaningful. Two mechanisms drive this. First, **context saturation**: as the memory store grows, prompts become increasingly long (1,850 tokens of memory context at task 30 vs. 209 at task 2), and the model's effective use of that context — as measured by memory usage rate — declines. Second, **concept density**: late tasks require recalling many more concepts simultaneously. Even when all 25 memory rows are present (100% memory_recall), the model consistently references only about 35–46% of them in its answer.

The uptake gap is the most important unsolved problem identified by this benchmark. Improving retrieval — either through better scoring formulas or higher K — does not address the fundamental challenge that retrieved context must be actively incorporated into model outputs.

### 6.4 Retrieval formula vs. capacity as the scoped retrieval bottleneck

A key result for the retrieval literature: at K=5, the deterministic scoring formula for `cag_scoped` reaches the oracle ceiling. Improving the formula — entity weighting, IDF rarity, generic penalties — would not improve outcomes at this capacity level. The correct intervention is capacity: a higher K that allows scoped retrieval to cover the 20+ concepts required by late tasks. This finding suggests that for benchmarks with rich inter-task concept dependencies, capacity management (how much to retrieve) matters more than selection quality (which items to retrieve).

### 6.5 Domain rules as persistent project constraints

An underappreciated finding is CAG's advantage on domain rule recall: 56.4% for 7B CAG vs. 47.2% for RAG, and 59.8% vs. 40.3% at 3B. Domain rules in this benchmark encode project-specific constraints such as "never diagnose," "no cloud sync," "use humane training methods only," and "implement local passcode." These rules, once established in memory, persist across all subsequent tasks. RAG can only apply them if the relevant source document is retrieved. CAG carries them explicitly. For real-world engineering projects, this class of finding — architectural constraints and behavioral boundaries that must be honored project-wide — represents exactly the kind of durable knowledge that CAG is designed to preserve.

### 6.6 Limitations

Several limitations constrain the generalizability of these findings.

**Single task domain.** All 30 tasks describe the same fictional application (`PawLedger`). Whether the CAG advantage generalizes to other domains — data pipelines, infrastructure engineering, API design — is an open question. The benchmark design is domain-agnostic in principle; we release it to enable replication with other task sequences.

**Grounded memory only.** Durable memory in this benchmark comes exclusively from task-defined ground truth, not from model output. Real-world CAG deployments would need to decide what model outputs are trustworthy enough to promote, a problem this benchmark does not address. The grounding constraint makes the benchmark cleaner but means it does not test the hardest real-world case.

**Two local models.** Experiments are limited to `qwen2.5-coder:3b-instruct` and `qwen2.5-coder:7b`. Larger models, models from other families, and API-hosted models may show different uptake rates and saturation behaviors. The relative ordering (CAG > RAG ≈ DAG) is likely to hold but the magnitude of the advantage may vary.

**Deterministic scoring.** The benchmark measures concept presence in output text, not semantic correctness or code quality. A model that lists prior decisions verbatim as a preamble scores identically to one that integrates those decisions naturally into its implementation plan. Supplementing deterministic scoring with LLM-as-judge evaluation is left for future work.

---

## 7. Future Work and Research Agenda

The findings from this benchmark define a sequence of open questions that we intend to pursue as a series of follow-on studies. We outline them here both as a research agenda and as an invitation for replication by others using the released benchmark.

### 7.1 Solving the uptake problem (near-term)

The dominant bottleneck identified in this work is not retrieval — it is uptake. Only 38–58% of retrieved concepts appear in model outputs, and this rate declines monotonically as task complexity grows. The benchmark as designed cannot distinguish between two possible causes: (a) the model attends to the memory context but fails to incorporate it into its output, or (b) the model's attention mechanism deprioritizes memory context when the prompt is long. These produce the same observable symptom but require different interventions.

A follow-on study would instrument this at the attention level using open-weight models where attention patterns are accessible, directly measuring whether retrieved memory rows are attended to before scoring whether they appear in outputs. This would also evaluate whether diversity-aware selection (choosing K memory rows that cover maximum concept variety rather than the top-K by score) improves uptake by reducing redundancy in the context block.

### 7.2 Model-generated memory formation (near-term)

CAG-Bench deliberately uses grounded memory — only task-defined facts enter the store. This was the right design choice for a controlled first study, but it defers the harder real-world question: how should a system decide which model outputs are trustworthy enough to promote into durable memory?

A follow-on benchmark would allow model-generated memory and evaluate several validation strategies: (a) pure model output with no validation (the baseline, expected to pollute quickly with hallucinations); (b) model output filtered by overlap with source documents; (c) model output filtered by contradiction with existing memory; (d) model output validated against concept-group scoring before promotion. The grounded-memory runs reported here would serve as the oracle ceiling for this study.

### 7.3 Multi-domain task sequences (near-term)

All 30 tasks in CAG-Bench describe a single application domain (dog training software). Whether the measured advantage of CAG generalizes across domains is unknown. A data engineering pipeline, an API design sequence, an infrastructure provisioning workflow, and a compliance documentation project all have different prior-decision structures and different kinds of continuity dependencies. We plan to release additional 30-task sequences in at least three domains and evaluate whether the CAG advantage, the phase degradation pattern, and the uptake bottleneck are consistent across domains or domain-specific.

### 7.4 Larger models and API-hosted systems (near-term)

All experiments reported here use 3B and 7B local models via Ollama. We expect larger models to show better uptake rates — their stronger instruction-following may improve concept integration — but we also expect the fundamental retrieval capacity problem (K=5 insufficient for late tasks requiring 20+ concepts) to persist regardless of model size, since it is a function of architecture constraints, not capability. A study comparing 7B, 13B, 34B, and 70B models on the same task sequence would establish whether uptake scales with model size or whether it requires architectural changes to improve.

### 7.5 Structured vs. unstructured memory representations (medium-term)

CAG-Bench stores memory as free-text summaries tagged with concept terms. An alternative representation — structured key-value stores, knowledge graph triples, or decision trees — might improve both retrieval precision and model uptake by presenting information in a format the model finds easier to reason over. A comparative study of memory representation formats on the same task sequence would quantify this trade-off. This connects directly to the SlopCodeBench finding that agents produce 2.2× more verbose outputs than human developers: a structured memory format might reduce prompt bloat while improving concept coverage.

### 7.6 Scoring beyond concept presence (medium-term)

CAG-Bench's deterministic scoring measures whether benchmark concepts appear in model outputs, not whether they are used correctly. A model that lists "SQLite, migration_version, soft_delete" as a preamble scores identically to one that integrates those decisions naturally into an implementation plan. This limitation was made explicit by an experiment in this work: a carry-forward prompt that instructed the model to list prior decisions verbatim dramatically improved measured continuity recall while degrading answer actionability as assessed by human review.

A follow-on study would develop an LLM-as-judge scoring layer calibrated against human developer ratings of answer actionability and decision fidelity. The calibration dataset would require human annotation of a sample of task answers across modes, producing human-validated scores that the judge model is trained to replicate. This would establish whether deterministic concept-presence scoring is a reliable proxy for practical usefulness — or systematically diverges from it in identifiable ways.

### 7.7 Longer task sequences and memory decay (longer-term)

Thirty tasks is a starting point. Software projects run for months or years, accumulating hundreds of decisions. A 100-task or 200-task sequence would answer: does CAG's advantage continue to grow, plateau, or eventually invert as the memory store becomes too large to retrieve from effectively? This is the natural boundary condition for any accumulation-based approach. Related work on memory forgetting (MemoryBank, PersistBench) suggests that indiscriminate accumulation eventually becomes a liability; the task sequence length at which this crossover occurs has not been empirically established for software development contexts.

### 7.8 Human baseline (longer-term)

No human developer baseline exists for this benchmark. A developer given the same source documents and task sequence — without the ability to look at prior answers — would produce a natural lower bound. A developer who can review all prior outputs would produce a natural upper bound on what CAG could theoretically achieve. Establishing both would anchor the absolute scale of CAG-Bench scores in human terms and clarify whether the 48.5 composite we report for 7B CAG represents a practically useful assistant or a system still far below the human floor for professional software development.

---

## 8. Conclusion

Current evaluation practice for LLM-assisted software development treats tasks as independent. CAG-Bench demonstrates that this assumption breaks down over the course of a real project: RAG and DAG continuity recall collapses from ~32% in early tasks to ~6% in tasks 21–30, while a model with access to accumulated project memory holds a 34.6% floor at the same stage. The gap is not marginal — at task 30, a final implementation handoff, RAG produces answers with 9.2% continuity recall. CAG produces 46.0%. That is the difference between an assistant that can describe a project and one that cannot.

Three findings from this work warrant particular attention. First, the CAG advantage is robust across model sizes — the same relative ordering appears at 3B and 7B parameters — suggesting it is a property of the retrieval architecture, not of model capability at a specific scale. Second, within the CAG family, a simple deterministic scoring formula for selective retrieval already saturates the oracle ceiling at K=5, which means improving the retrieval formula is unlikely to be productive. The capacity cap and the uptake problem are what matter. Third, and most importantly: this paper introduces `memory_usage_rate` as a diagnostic metric that makes the uptake problem visible for the first time in this context. Without it, a researcher observing that oracle memory does not outperform scoped retrieval on final answer quality would have no way to determine whether the failure is in selection, in delivery, or in the model's use of what was delivered. `memory_usage_rate` answers that question cleanly: the memory was retrieved, it was in the prompt, and the model did not use it. That is a different problem than retrieval failure, and it requires different interventions.

The grounding contract — durable memory derived only from task-defined ground truth, never from raw model output — is both a design choice and a finding. Earlier development of this benchmark confirmed empirically that models will hallucinate plausible-sounding but fabricated architectural decisions (cloud sync, external APIs, incompatible data models) and that without a validation gate, those inventions compound across tasks until the memory store is corrupted. Grounded memory is not a limitation of this benchmark; it is the correct baseline from which model-generated memory formation should be evaluated as a separate, harder problem.

The benchmark, task suite, scoring code, raw outputs, and promotion audit logs are released under AGPL-3.0-or-later at [repository URL]. The central question this work leaves open — and the question we consider most important for the field — is not how to retrieve memory better. It is how to make models use the memory they already have.

---

## Appendix A. Full Metric Grids

![Figure A1. All metrics grid — baseline run (qwen2.5-coder:7b, RAG / DAG / CAG, 30 tasks, 3 trials). Composite score, checklist quality, evidence recall, continuity recall, memory recall, memory precision, and contradiction penalty across all 30 tasks.](results/qwen7b_baseline_smoke/20260501_074401/all_metrics_grid.png)

![Figure A2. All metrics grid — CAG ablation run (qwen2.5-coder:7b, cag / cag_scoped / cag_oracle_memory, 30 tasks, 3 trials). Same metric set across the three CAG variants, showing the scoped/oracle convergence and base CAG token cost.](results/qwen7b_variants_smoke/20260501_130904/all_metrics_grid.png)

---

## Appendix C. Composite Score Weights

```
checklist_quality:      0.34
source_evidence_recall: 0.12
domain_rule_recall:     0.12
continuity_recall:      0.40
token_efficiency:       0.01
latency_efficiency:     0.01
minus contradiction_penalty
```

## Appendix D. Retrieval Formula (`cag_scoped`)

```
score = 3.0 × concept_overlap
      + 1.0 × tag_overlap
      + 0.5 × task_text_overlap (Jaccard)
      + 1.0 × recency_weight (idx / store_size)
```

## Appendix E. Memory Schema

```json
{
  "memory_id": "M01_00003",
  "task_id": "T03",
  "task_title": "Define local persistence",
  "scope": "project",
  "memory_type": "decision",
  "text": "Persistence uses SQLite with UTC timestamps, migration_version, soft_delete, and deterministic IDs.",
  "tags": ["persistence", "storage", "sqlite"],
  "promoted_terms": ["SQLite", "migration_version", "soft_delete", "deterministic IDs"]
}
```

## Appendix F. Run Configuration Reference

| Parameter | Value |
|---|---|
| Temperature | 0.1 |
| Context window | 8192 tokens |
| Source retrieval K | 3 |
| Memory top-K (scoped/oracle) | 5 |
| Memory cap (base CAG) | 25 |
| Ollama server | localhost:11434 |
