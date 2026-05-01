from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .ollama_client import OllamaClient, approx_tokens
from .retrieval import tokenize
from .utils import append_jsonl, coerce_concept_groups, contains_term, ensure_str_list, load_jsonl, normalize
from .vector import OllamaEmbedder, TfidfEmbedder, normalize as vec_normalize

VALID_SCOPES = {
    "global_project",
    "feature",
    "module",
    "decision",
    "test",
    "safety",
    "temporary",
}
VALID_TYPES = {
    "architecture",
    "domain_rule",
    "implementation_decision",
    "testing_rule",
    "safety_rule",
    "naming_convention",
    "known_issue",
    "open_question",
    "task_summary",
}
VALID_STATUSES = {"active", "deprecated", "temporary"}

CAG_VARIANTS = {"cag_naive", "cag_scoped", "cag_briefed", "cag_full"}
SCOPED_VARIANTS = {"cag_scoped", "cag_briefed", "cag_full"}
BRIEFED_VARIANTS = {"cag_briefed", "cag_full"}
COMPRESSION_VARIANTS = {"cag_briefed", "cag_full"}
FULL_VARIANTS = {"cag_full"}


def _extract_json_object(content: str) -> Dict[str, Any]:
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


def _extract_json_array(content: str) -> List[Dict[str, Any]]:
    text = (content or "").strip()
    if not text:
        return []
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        parsed = json.loads(text[start : end + 1])
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [row for row in parsed if isinstance(row, dict)]


def _concept_labels(values: Any) -> List[str]:
    labels: List[str] = []
    for group in coerce_concept_groups(values):
        concept = str(group.get("concept", "")).strip()
        if concept:
            labels.append(concept)
    return labels


def _norm_set(values: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for value in values:
        text = normalize(str(value))
        if text:
            out.add(text)
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def _cosine(vec_a, vec_b) -> float:
    try:
        return float((vec_a @ vec_b))
    except Exception:
        return 0.0


def _first_caps_entities(text: str) -> List[str]:
    # Keep this generic and language-agnostic enough for code-like entities.
    hits = re.findall(r"\b[A-Z][A-Za-z0-9_]{2,}\b", text or "")
    seen = set()
    out: List[str] = []
    for hit in hits:
        if hit not in seen:
            seen.add(hit)
            out.append(hit)
    return out[:24]


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if approx_tokens(text) <= max_tokens:
        return text
    words = (text or "").split()
    if not words:
        return ""
    out_words: List[str] = []
    for word in words:
        out_words.append(word)
        if approx_tokens(" ".join(out_words)) >= max_tokens:
            break
    return " ".join(out_words).strip()


def _type_from_scope(scope: str) -> str:
    if scope == "safety":
        return "safety_rule"
    if scope == "test":
        return "testing_rule"
    if scope == "decision":
        return "implementation_decision"
    if scope == "module":
        return "architecture"
    return "task_summary"


class CAGMemoryEngine:
    def __init__(
        self,
        path: str | Path,
        variant: str,
        *,
        brief_update_every_tasks: int = 5,
        brief_trigger_tokens: int = 1200,
        compression_trigger_tokens: int = 2600,
        max_project_brief_tokens: int = 350,
        max_scoped_memory_tokens: int = 900,
        max_recent_memory_tokens: int = 250,
        max_source_evidence_tokens: int = 1000,
        max_total_context_tokens: int = 2600,
        semantic_backend: str = "tfidf",
        semantic_embed_model: str = "nomic-embed-text",
        brief_log_path: str | Path | None = None,
    ):
        if variant not in CAG_VARIANTS:
            raise ValueError(f"Unknown CAG variant: {variant}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")
        self.variant = variant
        self.brief_update_every_tasks = brief_update_every_tasks
        self.brief_trigger_tokens = brief_trigger_tokens
        self.compression_trigger_tokens = compression_trigger_tokens
        self.max_project_brief_tokens = max_project_brief_tokens
        self.max_scoped_memory_tokens = max_scoped_memory_tokens
        self.max_recent_memory_tokens = max_recent_memory_tokens
        self.max_source_evidence_tokens = max_source_evidence_tokens
        self.max_total_context_tokens = max_total_context_tokens
        self.semantic_backend = semantic_backend
        self.semantic_embed_model = semantic_embed_model
        self._next_id = 1
        self._reuse_counts: Dict[str, int] = {}
        self.brief_log_path = Path(brief_log_path) if brief_log_path else None
        if self.brief_log_path is not None:
            self.brief_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._briefs: Dict[str, str] = {
            "project_brief": "",
            "architecture_brief": "",
            "decision_brief": "",
            "testing_brief": "",
            "open_questions_brief": "",
        }
        self._brief_last_updated_task_index = 0

    def _semantic_similarities(
        self,
        query_text: str,
        doc_texts: List[str],
        client: OllamaClient | None,
    ) -> List[float]:
        if not doc_texts:
            return []
        texts = [query_text] + doc_texts
        try:
            if self.semantic_backend == "ollama" and client is not None:
                embedder = OllamaEmbedder(client, self.semantic_embed_model)
                mat = vec_normalize(embedder.embed(texts))
            else:
                embedder = TfidfEmbedder()
                embedder.fit(texts)
                mat = vec_normalize(embedder.embed(texts))
            q = mat[0]
            sims: List[float] = []
            for i in range(1, mat.shape[0]):
                sims.append(max(0.0, min(1.0, _cosine(q, mat[i]))))
            return sims
        except Exception:
            return [_jaccard(tokenize(query_text), tokenize(text)) for text in doc_texts]

    def rows(self) -> List[Dict[str, Any]]:
        return load_jsonl(self.path)

    def _write_row(self, row: Dict[str, Any]) -> None:
        append_jsonl(self.path, row)

    def _append_update_event(self, event: Dict[str, Any]) -> None:
        payload = {"event": "memory_update", **event}
        self._write_row(payload)

    def _is_memory_item(self, row: Dict[str, Any]) -> bool:
        return bool(row.get("memory_id"))

    def all_items(self) -> List[Dict[str, Any]]:
        return [row for row in self.rows() if self._is_memory_item(row)]

    def active_items(self) -> List[Dict[str, Any]]:
        return [row for row in self.all_items() if row.get("status", "active") == "active"]

    def _ensure_next_id(self) -> None:
        max_n = 0
        for row in self.all_items():
            mid = str(row.get("memory_id", ""))
            if mid.startswith("M"):
                try:
                    max_n = max(max_n, int(mid[1:]))
                except Exception:
                    continue
        self._next_id = max(self._next_id, max_n + 1)

    def _new_memory_id(self) -> str:
        self._ensure_next_id()
        mid = f"M{self._next_id:05d}"
        self._next_id += 1
        return mid

    def infer_task_profile(self, task: Dict[str, Any], current_task_index: int) -> Dict[str, Any]:
        text = f"{task.get('title', '')}\n{task.get('prompt', '')}\n{' '.join(task.get('tags', []))}"
        tokens = tokenize(text)
        entities = _first_caps_entities(text)
        intent = "implementation"
        prompt_l = normalize(str(task.get("prompt", "")))
        if any(k in prompt_l for k in ["test", "validation", "verify"]):
            intent = "testing"
        elif any(k in prompt_l for k in ["safety", "risk", "boundary", "policy"]):
            intent = "safety"
        elif any(k in prompt_l for k in ["roadmap", "handoff", "release", "checklist"]):
            intent = "planning"

        likely_types = ["implementation_decision", "architecture"]
        likely_scope = ["global_project", "decision", "module"]
        specificity_target = 0.6
        if intent == "testing":
            likely_types = ["testing_rule", "known_issue", "implementation_decision"]
            likely_scope = ["test", "module", "decision"]
            specificity_target = 0.7
        if intent == "safety":
            likely_types = ["safety_rule", "domain_rule", "implementation_decision"]
            likely_scope = ["safety", "global_project", "decision"]
            specificity_target = 0.75

        return {
            "task_id": task.get("id"),
            "task_index": current_task_index,
            "task_intent": intent,
            "key_entities": entities,
            "token_query": sorted(tokens),
            "likely_memory_types": likely_types,
            "likely_scope": likely_scope,
            "specificity_target": specificity_target,
        }

    def _memory_text(self, row: Dict[str, Any]) -> str:
        parts = [
            row.get("text", ""),
            " ".join(ensure_str_list(row.get("tags", []))),
            " ".join(ensure_str_list(row.get("entities", []))),
            " ".join(ensure_str_list(row.get("promoted_terms", []))),
            str(row.get("scope", "")),
            str(row.get("type", "")),
        ]
        return " ".join(p for p in parts if p).strip()

    def _score_memory_item(
        self,
        item: Dict[str, Any],
        profile: Dict[str, Any],
        selected_ids: set[str],
        current_task_index: int,
        semantic_similarity: float,
    ) -> Dict[str, Any]:
        query_tokens = set(profile.get("token_query", []))
        item_text = self._memory_text(item)
        item_tokens = tokenize(item_text)
        tag_overlap = _jaccard(query_tokens, _norm_set(ensure_str_list(item.get("tags", []))))
        entity_overlap = _jaccard(
            _norm_set(profile.get("key_entities", [])),
            _norm_set(ensure_str_list(item.get("entities", []))),
        )
        deps = set(ensure_str_list(item.get("depends_on", [])))
        dependency_overlap = 0.0
        if deps:
            dependency_overlap = len(deps & selected_ids) / max(1, len(deps))

        created_idx = int(item.get("created_at_task_index") or 0)
        recency_weight = min(max(created_idx / max(1, current_task_index), 0.0), 1.0)

        mid = str(item.get("memory_id") or "")
        reuse = float(self._reuse_counts.get(mid, int(item.get("reuse_count") or 0)))
        reuse_weight = min(reuse / 6.0, 1.0)

        type_match = 1.0 if str(item.get("type")) in set(profile.get("likely_memory_types", [])) else 0.0

        specificity = float(item.get("specificity") or 0.5)
        specificity_mismatch = abs(float(profile.get("specificity_target", 0.6)) - specificity)

        deprecated_penalty = 1.0 if str(item.get("status", "active")) != "active" else 0.0

        score = (
            1.20 * semantic_similarity
            + 0.55 * tag_overlap
            + 0.70 * entity_overlap
            + 0.45 * dependency_overlap
            + 0.25 * recency_weight
            + 0.20 * reuse_weight
            + 0.45 * type_match
            - 0.25 * specificity_mismatch
            - 0.80 * deprecated_penalty
        )
        return {
            "memory_id": item.get("memory_id"),
            "score": score,
            "semantic_similarity": semantic_similarity,
            "tag_overlap": tag_overlap,
            "entity_overlap": entity_overlap,
            "dependency_overlap": dependency_overlap,
            "recency_weight": recency_weight,
            "reuse_weight": reuse_weight,
            "type_match": type_match,
            "specificity_mismatch": specificity_mismatch,
            "deprecated_penalty": deprecated_penalty,
        }

    def _candidate_contradicts_active(self, candidate: Dict[str, Any], active: List[Dict[str, Any]]) -> bool:
        text = normalize(str(candidate.get("text", "")))
        if not text:
            return False
        # Generic polarity pairs to catch direct memory conflicts.
        conflict_pairs = [
            ("no cloud", "requires cloud"),
            ("local-only", "requires cloud"),
            ("offline", "always online"),
            ("never diagnose", "diagnose"),
            ("no vet needed", "contact a veterinarian"),
            ("positive reinforcement", "shock collar"),
            ("force-free", "dominance theory"),
            ("deterministic id", "random id"),
        ]
        candidate_entities = _norm_set(ensure_str_list(candidate.get("entities", [])))
        for row in active:
            if row.get("status", "active") != "active":
                continue
            row_text = normalize(str(row.get("text", "")))
            row_entities = _norm_set(ensure_str_list(row.get("entities", [])))
            same_entity_family = bool(candidate_entities & row_entities) or not candidate_entities or not row_entities
            if not same_entity_family:
                continue
            for a, b in conflict_pairs:
                if (a in text and b in row_text) or (b in text and a in row_text):
                    return True
        return False

    def retrieve_for_task(
        self,
        task: Dict[str, Any],
        current_task_index: int,
        client: OllamaClient | None = None,
    ) -> Dict[str, Any]:
        profile = self.infer_task_profile(task, current_task_index)
        active = self.active_items()
        for row in active:
            mid = str(row.get("memory_id") or "")
            if not mid:
                continue
            row["reuse_count"] = int(self._reuse_counts.get(mid, int(row.get("reuse_count") or 0)))

        query_text = " ".join(
            [
                str(task.get("title", "")),
                str(task.get("prompt", "")),
                " ".join(task.get("tags", [])),
                " ".join(profile.get("key_entities", [])),
            ]
        )
        memory_texts = [self._memory_text(row) for row in active]
        semantic_sims = self._semantic_similarities(query_text, memory_texts, client)

        if self.variant == "cag_naive":
            continuity_terms: List[str] = []
            for group in coerce_concept_groups(task.get("continuity_terms", [])):
                continuity_terms.extend(ensure_str_list(group.get("accepted_terms", [])))
            query = tokenize(
                " ".join(
                    [
                        str(task.get("title", "")),
                        str(task.get("prompt", "")),
                        " ".join(task.get("tags", [])),
                        " ".join(continuity_terms),
                    ]
                )
            )
            scored: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
            for idx, row in enumerate(active):
                text_toks = tokenize(self._memory_text(row))
                overlap = len(query & text_toks)
                score = (3.0 * overlap) + (idx / max(1, len(active)))
                if str(row.get("scope", "")) in {"global", "global_project"}:
                    score += 0.5
                scored.append(
                    (
                        score,
                        row,
                        {
                            "memory_id": row.get("memory_id"),
                            "score": score,
                            "semantic_similarity": 0.0,
                            "tag_overlap": 0.0,
                            "entity_overlap": 0.0,
                            "dependency_overlap": 0.0,
                            "recency_weight": 0.0,
                            "reuse_weight": 0.0,
                            "type_match": 0.0,
                            "specificity_mismatch": 0.0,
                            "deprecated_penalty": 0.0,
                        },
                    )
                )
            scored.sort(key=lambda x: x[0], reverse=True)
            chosen = [row for _, row, _ in scored]
            for row in chosen:
                mid = str(row.get("memory_id") or "")
                if mid:
                    self._reuse_counts[mid] = int(self._reuse_counts.get(mid, int(row.get("reuse_count") or 0))) + 1
                    row["reuse_count"] = self._reuse_counts[mid]
            return {
                "task_profile": profile,
                "project_brief": "",
                "architecture_brief": "",
                "decision_brief": "",
                "testing_brief": "",
                "open_questions_brief": "",
                "selected_memory": chosen,
                "recent_memory": [],
                "retrieval_scores": [dict(audit, selected=(row in chosen), recent=False) for _, row, audit in scored],
                "brief_tokens_used": 0,
                "memory_tokens_used": sum(approx_tokens(self._memory_text(r)) for r in chosen),
            }

        remaining: List[Tuple[Dict[str, Any], float]] = []
        for idx, row in enumerate(active):
            sim = semantic_sims[idx] if idx < len(semantic_sims) else 0.0
            remaining.append((row, sim))

        selected: List[Dict[str, Any]] = []
        selected_audit: List[Dict[str, Any]] = []
        all_audits: List[Dict[str, Any]] = []
        selected_ids: set[str] = set()
        scoped_tokens = 0

        # Greedy selection with dependency-aware rescoring at each step.
        while remaining:
            rescored: List[Tuple[float, Dict[str, Any], Dict[str, Any], float]] = []
            for row, sim in remaining:
                audit = self._score_memory_item(row, profile, selected_ids, current_task_index, sim)
                rescored.append((audit["score"], row, audit, sim))
            rescored.sort(key=lambda x: x[0], reverse=True)
            best_score, best_row, best_audit, best_sim = rescored[0]
            all_audits.extend([dict(audit, selected=False, recent=False) for _, _, audit, _ in rescored[:20]])
            row_tokens = approx_tokens(self._memory_text(best_row))
            if scoped_tokens + row_tokens <= self.max_scoped_memory_tokens or not selected:
                selected.append(best_row)
                selected_audit.append(dict(best_audit, selected=True, recent=False))
                selected_ids.add(str(best_row.get("memory_id")))
                scoped_tokens += row_tokens
            remaining = [(r, s) for r, s in remaining if r is not best_row]
            if scoped_tokens >= self.max_scoped_memory_tokens:
                break

        # Recent memory gets a separate token lane, but only if likely relevant.
        recent_memory: List[Dict[str, Any]] = []
        recent_ids: set[str] = set()
        recent_tokens = 0
        for row in sorted(active, key=lambda r: int(r.get("created_at_task_index") or 0), reverse=True):
            if row in selected:
                continue
            row_tokens = approx_tokens(self._memory_text(row))
            if recent_tokens + row_tokens > self.max_recent_memory_tokens and recent_memory:
                continue
            rel = _jaccard(set(profile.get("token_query", [])), tokenize(self._memory_text(row)))
            if rel < 0.08:
                continue
            recent_memory.append(row)
            recent_ids.add(str(row.get("memory_id")))
            recent_tokens += row_tokens
            if recent_tokens >= self.max_recent_memory_tokens:
                break

        for row in selected + recent_memory:
            mid = str(row.get("memory_id") or "")
            if mid:
                self._reuse_counts[mid] = int(self._reuse_counts.get(mid, int(row.get("reuse_count") or 0))) + 1
                row["reuse_count"] = self._reuse_counts[mid]

        # Keep audits auditable: top candidate breakdowns plus selection flags.
        final_audits: List[Dict[str, Any]] = []
        seen_audit_ids: set[str] = set()
        for audit in selected_audit + all_audits:
            mid = str(audit.get("memory_id") or "")
            if not mid or mid in seen_audit_ids:
                continue
            seen_audit_ids.add(mid)
            final_audits.append(
                {
                    **audit,
                    "selected": mid in {str(r.get("memory_id")) for r in selected},
                    "recent": mid in recent_ids,
                }
            )
        final_audits.sort(key=lambda a: float(a.get("score", 0.0)), reverse=True)

        brief_tokens = 0
        if self.variant in BRIEFED_VARIANTS:
            for key in self._briefs:
                brief_tokens += approx_tokens(self._briefs.get(key, ""))
            if brief_tokens > self.max_project_brief_tokens:
                # Compress all briefs together into one bounded pass while keeping sections.
                packed = []
                for key in [
                    "project_brief",
                    "architecture_brief",
                    "decision_brief",
                    "testing_brief",
                    "open_questions_brief",
                ]:
                    text = self._briefs.get(key, "")
                    if text:
                        packed.append(f"{key}: {text}")
                merged = "\n".join(packed)
                merged = _truncate_to_tokens(merged, self.max_project_brief_tokens)
                # Keep only project brief in overflow case, but preserve the others as empty.
                self._briefs = {
                    "project_brief": merged,
                    "architecture_brief": "",
                    "decision_brief": "",
                    "testing_brief": "",
                    "open_questions_brief": "",
                }
                brief_tokens = approx_tokens(merged)

        return {
            "task_profile": profile,
            "project_brief": self._briefs.get("project_brief", ""),
            "architecture_brief": self._briefs.get("architecture_brief", ""),
            "decision_brief": self._briefs.get("decision_brief", ""),
            "testing_brief": self._briefs.get("testing_brief", ""),
            "open_questions_brief": self._briefs.get("open_questions_brief", ""),
            "selected_memory": selected,
            "recent_memory": recent_memory,
            "retrieval_scores": final_audits[:80],
            "brief_tokens_used": brief_tokens,
            "memory_tokens_used": scoped_tokens + recent_tokens,
        }

    def _indexer_prompt(
        self,
        task: Dict[str, Any],
        answer: str,
        retrieved_memory: List[Dict[str, Any]],
        source_ids: List[str],
    ) -> List[Dict[str, str]]:
        schema = {
            "memory_candidates": [
                {
                    "scope": "global_project|feature|module|decision|test|safety|temporary",
                    "type": "architecture|domain_rule|implementation_decision|testing_rule|safety_rule|naming_convention|known_issue|open_question|task_summary",
                    "text": "durable memory text",
                    "tags": ["tag"],
                    "entities": ["Entity"],
                    "depends_on": ["M00001"],
                    "supersedes": ["M00002"],
                    "confidence": 0.0,
                    "specificity": 0.0,
                }
            ],
            "task_local_notes": ["detail that should stay local"],
        }
        memory_context = "\n".join(
            [f"- {m.get('memory_id')} | {m.get('scope')} | {m.get('type')}: {m.get('text')}" for m in retrieved_memory[:10]]
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a benchmark memory indexer. Extract durable project memory as JSON. "
                    "Do not include markdown. Avoid vague or procedural-only items."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task ID: {task.get('id')}\n"
                    f"Task title: {task.get('title')}\n"
                    f"Task prompt:\n{task.get('prompt')}\n\n"
                    f"Answer:\n{answer}\n\n"
                    f"Retrieved memory:\n{memory_context or 'none'}\n\n"
                    f"Retrieved source ids: {', '.join(source_ids) if source_ids else 'none'}\n\n"
                    "Extract durable decisions, entities, constraints, dependencies, testing requirements, "
                    "safety/domain rules, known issues, open questions, and task-local details to avoid promotion.\n"
                    f"Return exact JSON shape: {json.dumps(schema)}"
                ),
            },
        ]

    def _dry_index_candidates(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        promote_terms = _concept_labels(task.get("promote_terms", []))
        if not promote_terms:
            promote_terms = _concept_labels(task.get("checklist_terms", []))[:3]
        entities = []
        for group in coerce_concept_groups(task.get("checklist_terms", [])):
            concept = str(group.get("concept", ""))
            if any(ch.isupper() for ch in concept):
                entities.append(concept.split()[0])
        entities = entities[:6]
        for term in promote_terms[:4]:
            scope = "decision"
            if "safety" in normalize(term):
                scope = "safety"
            if "test" in normalize(term):
                scope = "test"
            candidate = {
                "scope": scope,
                "type": _type_from_scope(scope),
                "text": f"{term} is an accepted project decision and should be reused for later tasks.",
                "tags": ensure_str_list(task.get("tags", []))[:4],
                "entities": entities,
                "depends_on": [],
                "supersedes": [],
                "confidence": 0.72,
                "specificity": 0.68,
            }
            candidates.append(candidate)
        candidates.append(
            {
                "scope": "temporary",
                "type": "task_summary",
                "text": f"Task-local output summary for {task.get('id')}",
                "tags": ensure_str_list(task.get("tags", []))[:2],
                "entities": [],
                "depends_on": [],
                "supersedes": [],
                "confidence": 0.4,
                "specificity": 0.35,
            }
        )
        return candidates

    def _normalize_candidate(
        self,
        candidate: Dict[str, Any],
        task: Dict[str, Any],
        created_at_task_index: int,
    ) -> Dict[str, Any] | None:
        text = str(candidate.get("text", "")).strip()
        if not text:
            return None

        scope = str(candidate.get("scope", "decision")).strip()
        if scope not in VALID_SCOPES:
            scope = "decision"

        ctype = str(candidate.get("type", "implementation_decision")).strip()
        if ctype not in VALID_TYPES:
            ctype = _type_from_scope(scope)

        depends_on = [mid for mid in ensure_str_list(candidate.get("depends_on", [])) if mid.startswith("M")]
        supersedes = [mid for mid in ensure_str_list(candidate.get("supersedes", [])) if mid.startswith("M")]

        try:
            confidence = float(candidate.get("confidence", 0.6))
        except Exception:
            confidence = 0.6
        confidence = max(0.0, min(1.0, confidence))

        try:
            specificity = float(candidate.get("specificity", 0.5))
        except Exception:
            specificity = 0.5
        specificity = max(0.0, min(1.0, specificity))

        tags = ensure_str_list(candidate.get("tags", []))
        if not tags:
            tags = ensure_str_list(task.get("tags", []))[:4]
        entities = ensure_str_list(candidate.get("entities", []))
        if not entities:
            entities = _first_caps_entities(text)[:8]

        return {
            "memory_id": self._new_memory_id(),
            "source_task_id": str(task.get("id", "")),
            "created_at_task_index": int(created_at_task_index),
            "status": "active",
            "scope": scope,
            "type": ctype,
            "text": text,
            "tags": tags[:12],
            "entities": entities[:16],
            "depends_on": depends_on[:16],
            "supersedes": supersedes[:16],
            "confidence": confidence,
            "specificity": specificity,
            "reuse_count": 0,
        }

    def _is_vague(self, text: str) -> bool:
        low = normalize(text)
        if len(low) < 35:
            return True
        vague_markers = [
            "we should",
            "maybe",
            "consider",
            "in general",
            "it is important",
            "task completed",
        ]
        return any(marker in low for marker in vague_markers)

    def _is_procedural_only(self, text: str) -> bool:
        low = normalize(text)
        procedural_markers = ["step", "run command", "click", "execute", "open file"]
        if any(marker in low for marker in procedural_markers):
            durable_markers = ["must", "rule", "constraint", "schema", "interface", "entity", "policy"]
            return not any(marker in low for marker in durable_markers)
        return False

    def _find_duplicate(self, candidate: Dict[str, Any], active: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        c_text = tokenize(candidate.get("text", ""))
        best = None
        best_score = 0.0
        for row in active:
            sim = _jaccard(c_text, tokenize(row.get("text", "")))
            if sim > best_score:
                best_score = sim
                best = row
        if best is not None and best_score >= 0.78:
            return best
        return None

    def _classify_candidate(self, candidate: Dict[str, Any], active: List[Dict[str, Any]]) -> Tuple[str, str, Dict[str, Any] | None]:
        text = str(candidate.get("text", "")).strip()
        if self._is_vague(text):
            return "discard", "too_vague", None
        if self._is_procedural_only(text):
            return "discard", "procedural_only", None
        if str(candidate.get("scope")) == "temporary" or str(candidate.get("type")) == "task_summary":
            return "keep_task_local", "task_local_only", None
        if self._candidate_contradicts_active(candidate, active):
            return "discard", "contradicted_by_active_memory", None

        dup = self._find_duplicate(candidate, active)
        if dup is not None:
            # If candidate explicitly indicates replacement, allow supersede.
            cand_l = normalize(text)
            if any(k in cand_l for k in ["supersede", "replace", "deprecate", "new standard"]):
                candidate["supersedes"] = [str(dup.get("memory_id"))]
                return "supersede_existing", "explicit_replace_signal", dup
            if float(candidate.get("confidence", 0.0)) >= float(dup.get("confidence", 0.0)):
                return "update_existing", "duplicate_update", dup
            return "discard", "duplicate_lower_confidence", dup

        supersedes = ensure_str_list(candidate.get("supersedes", []))
        if supersedes:
            return "supersede_existing", "explicit_supersede", None

        return "promote", "durable", None

    def _update_existing(self, target: Dict[str, Any], candidate: Dict[str, Any], reason: str) -> Dict[str, Any]:
        updated = dict(target)
        updated["text"] = str(candidate.get("text", target.get("text", ""))).strip()
        updated["tags"] = sorted(set(ensure_str_list(updated.get("tags", [])) + ensure_str_list(candidate.get("tags", []))))[:12]
        updated["entities"] = sorted(set(ensure_str_list(updated.get("entities", [])) + ensure_str_list(candidate.get("entities", []))))[:16]
        updated["depends_on"] = sorted(
            set(ensure_str_list(updated.get("depends_on", [])) + ensure_str_list(candidate.get("depends_on", [])))
        )[:16]
        updated["confidence"] = max(float(updated.get("confidence", 0.0)), float(candidate.get("confidence", 0.0)))
        updated["specificity"] = max(float(updated.get("specificity", 0.0)), float(candidate.get("specificity", 0.0)))
        updated["updated_by"] = candidate.get("source_task_id")
        self._append_update_event(
            {
                "action": "update_existing",
                "reason": reason,
                "target_memory_id": target.get("memory_id"),
                "updated_fields": ["text", "tags", "entities", "depends_on", "confidence", "specificity"],
            }
        )
        return updated

    def _deprecate_memory_ids(self, memory_ids: List[str], by_memory_id: str) -> None:
        if not memory_ids:
            return
        self._append_update_event(
            {
                "action": "deprecate_existing",
                "target_memory_ids": memory_ids,
                "by_memory_id": by_memory_id,
            }
        )

    def _rewrite_full_memory(self, items: List[Dict[str, Any]]) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    def index_and_update(
        self,
        *,
        client: OllamaClient,
        model: str,
        task: Dict[str, Any],
        answer: str,
        source_ids: List[str],
        retrieved_memory: List[Dict[str, Any]],
        dry_run: bool,
        num_ctx: int,
        temperature: float,
        current_task_index: int,
    ) -> Dict[str, Any]:
        if self.variant == "cag_naive":
            text = str(task.get("promote_summary", "")).strip()
            scope = str(task.get("scope", "project")).strip() or "project"
            row = {
                "memory_id": self._new_memory_id(),
                "source_task_id": str(task.get("id")),
                "created_at_task_index": current_task_index,
                "status": "active",
                "scope": scope,
                "type": "task_summary",
                "text": text,
                "tags": ensure_str_list(task.get("tags", []))[:8],
                "entities": _first_caps_entities(text)[:8],
                "promoted_terms": _concept_labels(task.get("promote_terms", [])),
                "depends_on": [],
                "supersedes": [],
                "confidence": 0.60,
                "specificity": 0.50,
                "reuse_count": 0,
            }
            self._write_row(row)
            return {
                "index_candidates": [row],
                "promotion_decisions": [
                    {
                        "memory_id": row.get("memory_id"),
                        "decision": "promote",
                        "reason": "naive_append",
                    }
                ],
                "added_count": 1,
                "updated_count": 0,
                "discarded_count": 0,
                "task_local_count": 0,
            }

        candidates_raw: List[Dict[str, Any]] = []
        if dry_run:
            candidates_raw = self._dry_index_candidates(task)
        else:
            messages = self._indexer_prompt(task, answer, retrieved_memory, source_ids)
            result = client.chat(model=model, messages=messages, temperature=temperature, num_ctx=num_ctx)
            parsed_obj = _extract_json_object(result.get("content", ""))
            if isinstance(parsed_obj.get("memory_candidates"), list):
                candidates_raw = [c for c in parsed_obj.get("memory_candidates", []) if isinstance(c, dict)]
            elif not candidates_raw:
                candidates_raw = _extract_json_array(result.get("content", ""))

        active = self.active_items()
        normalized: List[Dict[str, Any]] = []
        for cand in candidates_raw:
            norm = self._normalize_candidate(cand, task, current_task_index)
            if norm is not None:
                normalized.append(norm)

        decisions: List[Dict[str, Any]] = []
        keep: List[Dict[str, Any]] = []
        updates: Dict[str, Dict[str, Any]] = {}
        supersede_actions: List[Tuple[Dict[str, Any], List[str]]] = []

        discarded = 0
        task_local_count = 0

        for cand in normalized:
            decision, reason, dup_target = self._classify_candidate(cand, active)
            entry = {
                "memory_id": cand.get("memory_id"),
                "decision": decision,
                "reason": reason,
                "candidate_scope": cand.get("scope"),
                "candidate_type": cand.get("type"),
            }
            if dup_target is not None:
                entry["target_memory_id"] = dup_target.get("memory_id")
            decisions.append(entry)

            if decision == "promote":
                keep.append(cand)
            elif decision == "update_existing" and dup_target is not None:
                updates[str(dup_target.get("memory_id"))] = self._update_existing(dup_target, cand, reason)
            elif decision == "supersede_existing":
                supersede_actions.append((cand, ensure_str_list(cand.get("supersedes", []))))
                keep.append(cand)
            elif decision == "keep_task_local":
                task_local_count += 1
            else:
                discarded += 1

        if updates or supersede_actions:
            all_items = self.all_items()
            updated_items = []
            for row in all_items:
                mid = str(row.get("memory_id"))
                if mid in updates:
                    updated_items.append(updates[mid])
                else:
                    updated_items.append(row)

            if supersede_actions:
                superseded_ids = set()
                for cand, targets in supersede_actions:
                    for target in targets:
                        superseded_ids.add(target)
                for row in updated_items:
                    if str(row.get("memory_id")) in superseded_ids:
                        row["status"] = "deprecated"
                for cand, targets in supersede_actions:
                    cand["supersedes"] = targets
                    self._deprecate_memory_ids(targets, str(cand.get("memory_id")))

            self._rewrite_full_memory(updated_items)

        for row in keep:
            self._write_row(row)

        return {
            "index_candidates": normalized,
            "promotion_decisions": decisions,
            "added_count": len(keep),
            "updated_count": len(updates),
            "discarded_count": discarded,
            "task_local_count": task_local_count,
        }

    def _brief_section(self, title: str, items: List[Dict[str, Any]], max_tokens: int) -> str:
        if not items:
            return ""
        lines = [f"{title}:"]
        for row in items:
            tags = ", ".join(ensure_str_list(row.get("tags", []))[:3])
            text = str(row.get("text", "")).strip()
            lines.append(f"- {text} ({tags})" if tags else f"- {text}")
        return _truncate_to_tokens("\n".join(lines), max_tokens)

    def maybe_refresh_briefs(self, current_task_index: int) -> Dict[str, Any]:
        if self.variant not in BRIEFED_VARIANTS:
            return {"briefs_updated": False, "brief_reason": "variant_without_briefs"}

        active = self.active_items()
        token_load = sum(approx_tokens(self._memory_text(row)) for row in active)
        need_refresh = False
        reason = ""
        if current_task_index - self._brief_last_updated_task_index >= self.brief_update_every_tasks:
            need_refresh = True
            reason = "interval"
        if token_load >= self.brief_trigger_tokens:
            need_refresh = True
            reason = "token_threshold"

        if not need_refresh:
            return {"briefs_updated": False, "brief_reason": "not_due"}

        project_items = active[:18]
        architecture_items = [r for r in active if r.get("type") == "architecture"][:10]
        decision_items = [r for r in active if r.get("type") == "implementation_decision"][:12]
        testing_items = [r for r in active if r.get("type") == "testing_rule"][:10]
        open_items = [r for r in active if r.get("type") == "open_question"][:8]

        slice_tokens = max(40, self.max_project_brief_tokens // 5)
        self._briefs = {
            "project_brief": self._brief_section("Project brief", project_items, slice_tokens),
            "architecture_brief": self._brief_section("Architecture brief", architecture_items, slice_tokens),
            "decision_brief": self._brief_section("Decision brief", decision_items, slice_tokens),
            "testing_brief": self._brief_section("Testing brief", testing_items, slice_tokens),
            "open_questions_brief": self._brief_section("Open questions brief", open_items, slice_tokens),
        }
        self._brief_last_updated_task_index = current_task_index
        if self.brief_log_path is not None:
            append_jsonl(
                self.brief_log_path,
                {
                    "task_index": current_task_index,
                    "reason": reason,
                    "briefs": dict(self._briefs),
                    "brief_token_load": sum(approx_tokens(v) for v in self._briefs.values()),
                },
            )
        return {
            "briefs_updated": True,
            "brief_reason": reason,
            "brief_token_load": sum(approx_tokens(v) for v in self._briefs.values()),
        }

    def maybe_compress_memory(self, current_task_index: int) -> Dict[str, Any]:
        if self.variant not in COMPRESSION_VARIANTS:
            return {"compression_ran": False, "compression_reason": "variant_without_compression"}

        active = self.active_items()
        token_load = sum(approx_tokens(self._memory_text(row)) for row in active)
        if token_load < self.compression_trigger_tokens:
            return {"compression_ran": False, "compression_reason": "below_threshold"}

        groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        for row in active:
            entity_key = ""
            ents = ensure_str_list(row.get("entities", []))
            if ents:
                entity_key = ents[0]
            key = (str(row.get("scope", "")), str(row.get("type", "")), entity_key)
            groups.setdefault(key, []).append(row)

        all_items = self.all_items()
        items_by_id = {str(r.get("memory_id")): dict(r) for r in all_items}
        merged_count = 0
        deprecated_ids: List[str] = []

        for (_, _, _), rows in groups.items():
            if len(rows) < 2:
                continue
            rows_sorted = sorted(rows, key=lambda r: int(r.get("created_at_task_index") or 0))
            anchor = rows_sorted[-1]
            anchor_tokens = tokenize(anchor.get("text", ""))
            to_merge = [anchor]
            for row in rows_sorted[:-1]:
                sim = _jaccard(anchor_tokens, tokenize(row.get("text", "")))
                if sim >= 0.48:
                    to_merge.append(row)
            if len(to_merge) < 2:
                continue

            merged_count += len(to_merge) - 1
            merged_text_parts = []
            merged_tags: set[str] = set()
            merged_entities: set[str] = set()
            merged_supersedes: List[str] = []
            for row in to_merge:
                merged_text_parts.append(str(row.get("text", "")).strip())
                merged_tags.update(ensure_str_list(row.get("tags", [])))
                merged_entities.update(ensure_str_list(row.get("entities", [])))
                merged_supersedes.append(str(row.get("memory_id")))

            consolidated = dict(anchor)
            consolidated["memory_id"] = self._new_memory_id()
            consolidated["text"] = _truncate_to_tokens(" ".join(dict.fromkeys(merged_text_parts)), 120)
            consolidated["tags"] = sorted(merged_tags)[:12]
            consolidated["entities"] = sorted(merged_entities)[:16]
            consolidated["supersedes"] = merged_supersedes
            consolidated["created_at_task_index"] = current_task_index
            consolidated["status"] = "active"
            consolidated["confidence"] = max(float(r.get("confidence", 0.0)) for r in to_merge)
            consolidated["specificity"] = max(float(r.get("specificity", 0.0)) for r in to_merge)
            items_by_id[str(consolidated.get("memory_id"))] = consolidated

            for row in to_merge:
                rid = str(row.get("memory_id"))
                if rid in items_by_id:
                    items_by_id[rid]["status"] = "deprecated"
                    deprecated_ids.append(rid)

        if merged_count <= 0:
            return {"compression_ran": False, "compression_reason": "no_merge_candidates"}

        rewritten = list(items_by_id.values())
        rewritten.sort(key=lambda r: (int(r.get("created_at_task_index") or 0), str(r.get("memory_id"))))
        self._rewrite_full_memory(rewritten)
        for row in rewritten:
            mid = str(row.get("memory_id") or "")
            if mid:
                self._reuse_counts[mid] = int(row.get("reuse_count") or self._reuse_counts.get(mid, 0))
        self._append_update_event(
            {
                "action": "compression",
                "merged_items": merged_count,
                "deprecated_ids": sorted(set(deprecated_ids)),
            }
        )
        return {
            "compression_ran": True,
            "compression_reason": "threshold",
            "merged_items": merged_count,
            "deprecated_items": len(set(deprecated_ids)),
        }

    def contradiction_check(
        self,
        *,
        client: OllamaClient,
        model: str,
        task: Dict[str, Any],
        answer: str,
        selected_memory: List[Dict[str, Any]],
        dry_run: bool,
        num_ctx: int,
    ) -> Dict[str, Any]:
        if self.variant not in FULL_VARIANTS:
            return {
                "checked": False,
                "contradictions_before_revision": 0,
                "contradictions_after_revision": 0,
                "revision_used": False,
                "answer": answer,
                "check_notes": [],
            }

        high_priority = []
        for row in selected_memory:
            if row.get("status", "active") != "active":
                continue
            if row.get("scope") in {"global_project", "decision", "safety"}:
                high_priority.append(row)
        high_priority = high_priority[:10]

        def _rule_based_count(ans: str) -> int:
            ans_l = normalize(ans)
            total = 0
            for group in task.get("contradiction_terms", []) or []:
                bad_terms = ensure_str_list(group.get("bad_terms", [])) if isinstance(group, dict) else []
                if any(normalize(term) in ans_l for term in bad_terms):
                    total += 1
            # Lightweight rename drift check for selected entity names.
            entity_hits = 0
            entity_total = 0
            for row in high_priority:
                for ent in ensure_str_list(row.get("entities", []))[:2]:
                    entity_total += 1
                    if normalize(ent) in ans_l:
                        entity_hits += 1
            if entity_total > 0 and entity_hits == 0:
                total += 1
            return total

        def _run_llm_check(ans: str) -> Tuple[int, Dict[str, Any]]:
            memory_text = "\n".join(
                [
                    f"- {m.get('memory_id')} | {m.get('scope')} | {m.get('type')} | entities={','.join(ensure_str_list(m.get('entities', [])))}\n  {m.get('text')}"
                    for m in high_priority
                ]
            )
            schema = {
                "contradictions": [
                    {
                        "memory_id": "M00001",
                        "reason": "conflict detail",
                        "severity": "low|medium|high",
                    }
                ],
                "rename_conflicts": ["entity rename issue"],
                "constraint_omissions": ["missing required constraint"],
            }
            check_messages = [
                {
                    "role": "system",
                    "content": "You are a contradiction checker. Return JSON only with specific contradictions against memory.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Task: {task.get('title')}\n"
                        f"Task prompt: {task.get('prompt')}\n\n"
                        f"High priority active memory:\n{memory_text or 'none'}\n\n"
                        f"Answer:\n{ans}\n\n"
                        "Check whether answer contradicts active decisions, renames accepted entities without reason, "
                        "ignores required constraints, or conflicts with architecture/safety rules.\n"
                        f"Return JSON: {json.dumps(schema)}"
                    ),
                },
            ]
            check_result = client.chat(model=model, messages=check_messages, temperature=0.0, num_ctx=num_ctx)
            parsed = _extract_json_object(check_result.get("content", ""))
            contradictions = parsed.get("contradictions", []) if isinstance(parsed, dict) else []
            rename_conflicts = parsed.get("rename_conflicts", []) if isinstance(parsed, dict) else []
            constraint_omissions = parsed.get("constraint_omissions", []) if isinstance(parsed, dict) else []
            llm_count = len(contradictions) + len(rename_conflicts) + len(constraint_omissions)
            return llm_count, {
                "contradictions": contradictions,
                "rename_conflicts": rename_conflicts,
                "constraint_omissions": constraint_omissions,
            }

        if dry_run:
            contradictions = []
            answer_l = normalize(answer)
            for row in high_priority:
                if str(row.get("scope")) == "safety":
                    text = normalize(str(row.get("text", "")))
                    if "never" in text and "always" in answer_l and "always" not in text:
                        contradictions.append({"memory_id": row.get("memory_id"), "reason": "possible polarity conflict"})
            before = len(contradictions) + _rule_based_count(answer)
            after = before
            return {
                "checked": True,
                "contradictions_before_revision": before,
                "contradictions_after_revision": after,
                "revision_used": False,
                "answer": answer,
                "check_notes": contradictions,
            }

        llm_before, parsed_notes = _run_llm_check(answer)
        before = max(llm_before, _rule_based_count(answer))
        if before <= 0:
            return {
                "checked": True,
                "contradictions_before_revision": 0,
                "contradictions_after_revision": 0,
                "revision_used": False,
                "answer": answer,
                "check_notes": parsed_notes,
            }

        memory_text = "\n".join(
            [
                f"- {m.get('memory_id')} | {m.get('scope')} | {m.get('type')} | entities={','.join(ensure_str_list(m.get('entities', [])))}\n  {m.get('text')}"
                for m in high_priority
            ]
        )
        revise_messages = [
            {
                "role": "system",
                "content": "You revise answers to resolve contradictions with accepted project memory while preserving task intent.",
            },
            {
                "role": "user",
                "content": (
                    f"Original answer:\n{answer}\n\n"
                    f"High priority memory:\n{memory_text}\n\n"
                    f"Conflicts found:\n{json.dumps(parsed_notes, ensure_ascii=False)}\n\n"
                    "Rewrite the answer to resolve conflicts and keep it concrete."
                ),
            },
        ]
        revised = client.chat(model=model, messages=revise_messages, temperature=0.0, num_ctx=num_ctx)
        revised_answer = revised.get("content", "") or answer

        llm_after, after_notes = _run_llm_check(revised_answer)
        after = max(llm_after, _rule_based_count(revised_answer))
        return {
            "checked": True,
            "contradictions_before_revision": before,
            "contradictions_after_revision": after,
            "revision_used": True,
            "answer": revised_answer,
            "check_notes": {
                "before": parsed_notes,
                "after": after_notes,
            },
        }

    def build_context_bundle(
        self,
        task: Dict[str, Any],
        sources: List[Dict[str, Any]],
        retrieval: Dict[str, Any],
    ) -> Dict[str, Any]:
        source_text = "\n\n".join([f"### Source: {s.get('title','')}\n{s.get('text','').strip()}" for s in sources])
        source_text = _truncate_to_tokens(source_text, self.max_source_evidence_tokens)
        source_tokens = approx_tokens(source_text)

        memory_lines = []
        for row in retrieval.get("selected_memory", []):
            memory_lines.append(
                f"- [{row.get('memory_id')}] [{row.get('scope')}/{row.get('type')}] entities={','.join(ensure_str_list(row.get('entities', [])))} tags={','.join(ensure_str_list(row.get('tags', [])))}\n  {row.get('text','').strip()}"
            )
        for row in retrieval.get("recent_memory", []):
            memory_lines.append(
                f"- [recent {row.get('memory_id')}] [{row.get('scope')}/{row.get('type')}] {row.get('text','').strip()}"
            )
        memory_text = "\n".join(memory_lines)
        memory_text = _truncate_to_tokens(memory_text, self.max_scoped_memory_tokens + self.max_recent_memory_tokens)
        memory_tokens = approx_tokens(memory_text)

        brief_sections = []
        if self.variant in BRIEFED_VARIANTS:
            for key in [
                "project_brief",
                "architecture_brief",
                "decision_brief",
                "testing_brief",
                "open_questions_brief",
            ]:
                text = retrieval.get(key, "")
                if text:
                    brief_sections.append(text)
        brief_text = "\n\n".join(brief_sections)
        brief_text = _truncate_to_tokens(brief_text, self.max_project_brief_tokens)
        brief_tokens = approx_tokens(brief_text) if brief_text else 0

        known_constraints = []
        for row in retrieval.get("selected_memory", []):
            if row.get("scope") in {"safety", "decision"}:
                known_constraints.append(f"- {row.get('text','').strip()}")
        constraints_text = "\n".join(known_constraints[:8])

        open_questions = []
        for row in self.active_items():
            if row.get("type") == "open_question" and row.get("status") == "active":
                open_questions.append(f"- {row.get('text','').strip()}")
        open_questions_text = "\n".join(open_questions[:8])

        total = brief_tokens + memory_tokens + source_tokens + approx_tokens(constraints_text + open_questions_text)
        if total > self.max_total_context_tokens:
            # Tighten memory first, then sources.
            overflow = total - self.max_total_context_tokens
            memory_trim = max(memory_tokens - overflow, 120)
            memory_text = _truncate_to_tokens(memory_text, memory_trim)
            memory_tokens = approx_tokens(memory_text)
            total = brief_tokens + memory_tokens + source_tokens + approx_tokens(constraints_text + open_questions_text)
            if total > self.max_total_context_tokens:
                source_trim = max(self.max_source_evidence_tokens - (total - self.max_total_context_tokens), 180)
                source_text = _truncate_to_tokens(source_text, source_trim)
                source_tokens = approx_tokens(source_text)

        user = (
            f"Mode: {self.variant}.\n\n"
            f"Current task:\n{task.get('prompt','')}\n\n"
            f"Project brief:\n{brief_text or 'none'}\n\n"
            f"Relevant durable decisions and scoped memory:\n{memory_text or 'none'}\n\n"
            f"Known constraints:\n{constraints_text or 'none'}\n\n"
            f"Open questions:\n{open_questions_text or 'none'}\n\n"
            f"Fresh retrieved evidence:\n{source_text or 'none'}\n\n"
            "Return: implementation plan, concrete decisions, risks, and tests. Preserve accepted terms and entities."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful local coding assistant. Use provided project memory and evidence. "
                    "Do not invent conflicting architecture."
                ),
            },
            {"role": "user", "content": user},
        ]

        return {
            "messages": messages,
            "memory_tokens_used": memory_tokens,
            "brief_tokens_used": brief_tokens,
            "source_tokens_used": source_tokens,
            "total_context_tokens": approx_tokens(user),
        }


def expected_prior_concept_groups(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    explicit = ensure_str_list(task.get("expected_prior_concepts", []))
    if explicit:
        for concept in explicit:
            groups.append({"concept": concept, "accepted_terms": [concept]})
    continuity_groups = coerce_concept_groups(task.get("continuity_terms", []))
    if continuity_groups:
        by_concept: Dict[str, set[str]] = {}
        for group in continuity_groups:
            concept = str(group.get("concept", "")).strip()
            if not concept:
                continue
            if concept not in by_concept:
                by_concept[concept] = set()
            for term in ensure_str_list(group.get("accepted_terms", [])):
                by_concept[concept].add(term)
        for concept, terms in by_concept.items():
            groups.append({"concept": concept, "accepted_terms": sorted(terms) or [concept]})
    # Fallback for legacy plain string continuity lists.
    if not groups:
        legacy = ensure_str_list(task.get("continuity_terms", []))
        for concept in legacy:
            groups.append({"concept": concept, "accepted_terms": [concept]})
    return groups


def expected_prior_concepts(task: Dict[str, Any]) -> List[str]:
    return [str(g.get("concept", "")).strip() for g in expected_prior_concept_groups(task) if str(g.get("concept", "")).strip()]


def memory_precision_recall(
    selected_memory: List[Dict[str, Any]],
    expected_concepts: List[str],
    expected_groups: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    groups = expected_groups or [{"concept": c, "accepted_terms": [c]} for c in expected_concepts]
    if not expected_concepts:
        return {
            "memory_precision": 1.0,
            "memory_recall": 1.0,
            "relevant_hits": 0,
            "selected_total": len(selected_memory),
            "concept_hits": {},
            "irrelevant_memory_used": 0,
        }

    concept_hit: Dict[str, bool] = {str(group.get("concept", "")): False for group in groups if str(group.get("concept", ""))}
    relevant = 0
    deprecated_used = 0

    for item in selected_memory:
        text = " ".join(
            [
                str(item.get("text", "")),
                " ".join(ensure_str_list(item.get("tags", []))),
                " ".join(ensure_str_list(item.get("entities", []))),
            ]
        )
        item_rel = False
        for group in groups:
            concept = str(group.get("concept", "")).strip()
            accepted_terms = ensure_str_list(group.get("accepted_terms", [])) or [concept]
            if any(contains_term(text, term) for term in accepted_terms):
                if concept:
                    concept_hit[concept] = True
                item_rel = True
        if item_rel:
            relevant += 1
        if str(item.get("status", "active")) != "active":
            deprecated_used += 1

    selected_total = len(selected_memory)
    precision = 1.0 if selected_total == 0 else (relevant / selected_total)
    recall = sum(1 for v in concept_hit.values() if v) / max(1, len(expected_concepts))
    return {
        "memory_precision": precision,
        "memory_recall": recall,
        "relevant_hits": relevant,
        "selected_total": selected_total,
        "concept_hits": concept_hit,
        "irrelevant_memory_used": max(0, selected_total - relevant),
        "deprecated_memory_used": deprecated_used,
    }
