
from pathlib import Path
from typing import Dict, Any, List
from .retrieval import tokenize
from .utils import append_jsonl, load_jsonl, extract_terms, coerce_concept_groups, contains_term

class ProjectMemory:
    def __init__(self, path: str | Path, max_chars: int | None = None, trial: int | None = None):
        self.path = Path(path)
        self.max_chars = max_chars
        self.trial = trial
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists(): self.path.write_text('', encoding='utf-8')
        self.promotions_path = self.path.with_name(self.path.stem + '_promotions.jsonl')
        if not self.promotions_path.exists():
            self.promotions_path.write_text('', encoding='utf-8')

    @staticmethod
    def _norm_set(values: list[str]) -> set[str]:
        return {str(v).strip().lower() for v in values if str(v).strip()}

    def rows(self) -> List[Dict[str, Any]]:
        return load_jsonl(self.path)

    def add(self, task: Dict[str, Any], accepted_summary: str) -> None:
        candidate_text = (accepted_summary or '').strip()
        if not candidate_text:
            append_jsonl(
                self.promotions_path,
                {
                    "task_id": task.get('id'),
                    "task_index": task.get('index'),
                    "candidate_text": "",
                    "decision": "task_local",
                    "reason": "no_promote_summary",
                    "memory_id": None,
                },
            )
            return

        next_idx = len(self.rows()) + 1
        trial_num = int(self.trial or 0)
        memory_id = f"M{trial_num:02d}_{next_idx:05d}"
        append_jsonl(
            self.path,
            {
                "memory_id": memory_id,
                "task_id": task['id'],
                "task_title": task['title'],
                "scope": task.get('scope', 'project'),
                "memory_type": task.get('memory_type', 'decision'),
                "text": candidate_text,
                "tags": task.get('tags', []),
                "promoted_terms": task.get('promote_terms', []),
                "source": "promote_summary",
            },
        )
        append_jsonl(
            self.promotions_path,
            {
                "task_id": task.get('id'),
                "task_index": task.get('index'),
                "candidate_text": candidate_text,
                "decision": "promote",
                "reason": "supported_by_promote_summary",
                "memory_id": memory_id,
            },
        )

    def retrieve(self, task: Dict[str, Any], k: int | None = None) -> List[Dict[str, Any]]:
        rows = self.rows()
        if not rows: return []
        continuity_terms = extract_terms(task.get('continuity_terms', []))
        qtext = ' '.join([task.get('title',''), task.get('prompt',''), ' '.join(task.get('tags',[])), ' '.join(continuity_terms)])
        q = tokenize(qtext); scored = []
        for idx,row in enumerate(rows):
            rtext = ' '.join([row.get('text',''), row.get('scope',''), ' '.join(row.get('tags',[])), ' '.join(row.get('promoted_terms',[]))])
            score = len(q & tokenize(rtext))*3.0 + (idx/max(1,len(rows)))
            if row.get('scope') == 'global': score += 0.5
            scored.append((score,row))
        scored.sort(key=lambda x:x[0], reverse=True)
        chosen=[]; total=0
        limit = len(scored) if k is None else k
        for score,row in scored[:limit]:
            n=len(row.get('text',''))
            if self.max_chars is not None and total+n > self.max_chars and chosen: continue
            chosen.append(row); total += n
        return chosen

    def retrieve_scoped(
        self,
        task: Dict[str, Any],
        k: int | None = None,
        return_scores: bool = False,
    ) -> List[Dict[str, Any]] | tuple[List[Dict[str, Any]], list[dict[str, Any]]]:
        rows = self.rows()
        if not rows:
            return ([], []) if return_scores else []

        continuity_groups = coerce_concept_groups(task.get('continuity_terms', []))
        task_tags = self._norm_set(task.get('tags', []))
        continuity_terms = extract_terms(task.get('continuity_terms', []))
        qtext = ' '.join([task.get('title', ''), task.get('prompt', ''), ' '.join(task.get('tags', [])), ' '.join(continuity_terms)])
        q_tokens = tokenize(qtext)
        scored: list[dict[str, Any]] = []
        rows_count = len(rows)

        for idx, row in enumerate(rows, start=1):
            row_text = str(row.get('text', '') or '')
            row_tags = self._norm_set(row.get('tags', []))
            row_promoted_terms = self._norm_set(row.get('promoted_terms', []))

            concept_overlap = 0
            for group in continuity_groups:
                accepted_terms = self._norm_set(group.get('accepted_terms', []))
                matched = any(contains_term(row_text, term) for term in accepted_terms) or bool(accepted_terms & row_promoted_terms)
                if matched:
                    concept_overlap += 1

            tag_overlap = len(task_tags & row_tags)
            rtext = ' '.join([row_text, str(row.get('scope', '') or ''), ' '.join(row.get('tags', [])), ' '.join(row.get('promoted_terms', []))])
            r_tokens = tokenize(rtext)
            union = q_tokens | r_tokens
            task_text_overlap = (len(q_tokens & r_tokens) / len(union)) if union else 0.0
            recency_weight = idx / max(1, rows_count)

            score = (
                3.0 * concept_overlap
                + 1.0 * tag_overlap
                + 0.5 * task_text_overlap
                + 1.0 * recency_weight
            )

            scored.append(
                {
                    "row": row,
                    "score": score,
                    "concept_overlap": concept_overlap,
                    "tag_overlap": tag_overlap,
                    "task_text_overlap": task_text_overlap,
                    "recency_weight": recency_weight,
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        limit = len(scored) if k is None else k
        chosen: list[dict[str, Any]] = []
        chosen_ids: set[str] = set()
        total = 0
        for item in scored[:limit]:
            row = item["row"]
            n = len(str(row.get('text', '') or ''))
            if self.max_chars is not None and total + n > self.max_chars and chosen:
                continue
            chosen.append(row)
            chosen_ids.add(str(row.get("memory_id", "")))
            total += n

        score_rows = []
        for item in scored:
            row = item["row"]
            memory_id = str(row.get("memory_id", ""))
            score_rows.append(
                {
                    "memory_id": memory_id,
                    "score": item["score"],
                    "concept_overlap": item["concept_overlap"],
                    "tag_overlap": item["tag_overlap"],
                    "task_text_overlap": item["task_text_overlap"],
                    "recency_weight": item["recency_weight"],
                    "selected": memory_id in chosen_ids,
                }
            )
        if return_scores:
            return chosen, score_rows
        return chosen

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    def retrieve_scoped_promptonly(
        self,
        task: Dict[str, Any],
        sources: List[Any],
        k: int | None = None,
        return_scores: bool = False,
    ) -> List[Dict[str, Any]] | tuple[List[Dict[str, Any]], list[dict[str, Any]]]:
        rows = self.rows()
        if not rows:
            return ([], []) if return_scores else []

        task_prompt_tokens = tokenize(
            " ".join([task.get("title", ""), task.get("prompt", "")])
        )
        source_text = " ".join(
            [f"{getattr(s, 'title', '')} {getattr(s, 'text', '')}" for s in sources]
        )
        source_tokens = tokenize(source_text)
        task_tags = self._norm_set(task.get("tags", []))
        rows_count = len(rows)

        scored: list[dict[str, Any]] = []
        for idx, row in enumerate(rows, start=1):
            row_text = str(row.get("text", "") or "")
            row_promoted = " ".join([str(v) for v in row.get("promoted_terms", [])])
            row_prompt_tokens = tokenize(f"{row_text} {row_promoted}")
            row_text_tokens = tokenize(row_text)
            row_tags = self._norm_set(row.get("tags", []))

            task_prompt_overlap = self._jaccard(task_prompt_tokens, row_prompt_tokens)
            source_doc_overlap = self._jaccard(source_tokens, row_text_tokens)
            tag_overlap = len(task_tags & row_tags)
            recency_weight = idx / max(1, rows_count)

            score = (
                2.0 * task_prompt_overlap
                + 1.0 * source_doc_overlap
                + 1.0 * tag_overlap
                + 0.5 * recency_weight
            )

            scored.append(
                {
                    "row": row,
                    "score": score,
                    "task_prompt_overlap": task_prompt_overlap,
                    "source_doc_overlap": source_doc_overlap,
                    "tag_overlap": tag_overlap,
                    "recency_weight": recency_weight,
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        limit = len(scored) if k is None else k
        chosen: list[dict[str, Any]] = []
        chosen_ids: set[str] = set()
        total = 0
        for item in scored[:limit]:
            row = item["row"]
            n = len(str(row.get("text", "") or ""))
            if self.max_chars is not None and total + n > self.max_chars and chosen:
                continue
            chosen.append(row)
            chosen_ids.add(str(row.get("memory_id", "")))
            total += n

        score_rows = []
        for item in scored:
            row = item["row"]
            memory_id = str(row.get("memory_id", ""))
            score_rows.append(
                {
                    "memory_id": memory_id,
                    "score": item["score"],
                    "task_prompt_overlap": item["task_prompt_overlap"],
                    "source_doc_overlap": item["source_doc_overlap"],
                    "tag_overlap": item["tag_overlap"],
                    "recency_weight": item["recency_weight"],
                    "selected": memory_id in chosen_ids,
                }
            )
        if return_scores:
            return chosen, score_rows
        return chosen
