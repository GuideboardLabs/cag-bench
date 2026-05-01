from typing import Dict, Any
from .utils import coerce_concept_groups, coerce_contradiction_groups, contains_term

BASE_WEIGHTS = {
    "checklist_quality": 0.34,
    "source_evidence_recall": 0.12,
    "domain_rule_recall": 0.12,
    "continuity_recall": 0.40,
    "token_efficiency": 0.01,
    "latency_efficiency": 0.01,
}

def _safe_pct(hit_count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return hit_count / total

def _hit_concepts(answer: str, groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        matched_term = None
        for term in group.get("accepted_terms", []):
            if contains_term(answer, term):
                matched_term = term
                break
        rows.append(
            {
                "concept": group.get("concept", ""),
                "hit": matched_term is not None,
                "matched_term": matched_term,
            }
        )
    return rows

def _hit_contradictions(answer: str, groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        matched_term = None
        for term in group.get("bad_terms", []):
            if contains_term(answer, term):
                matched_term = term
                break
        if matched_term is None:
            continue
        rows.append(
            {
                "concept": group.get("concept", ""),
                "matched_term": matched_term,
                "penalty": float(group.get("penalty", 0.0)),
            }
        )
    return rows

def _rows_to_pct(rows: list[dict[str, Any]]) -> tuple[float | None, int, int]:
    hit_count = sum(1 for row in rows if row.get("hit"))
    total = len(rows)
    pct = _safe_pct(hit_count, total)
    return (None if pct is None else pct * 100.0), hit_count, total

def _resolve_evidence_groups(task: Dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if "source_evidence_terms" in task or "domain_rule_terms" in task:
        source_groups = coerce_concept_groups(task.get("source_evidence_terms", []))
        domain_groups = coerce_concept_groups(task.get("domain_rule_terms", []))
        return source_groups, domain_groups
    # Backward compatibility: legacy evidence_terms map to source_evidence_recall.
    return coerce_concept_groups(task.get("evidence_terms", [])), []

def metric_scores(
    answer: str,
    task: Dict[str, Any],
    prompt_tokens: int,
    latency_seconds: float,
    token_budget: int = 4500,
    latency_budget: float = 90.0,
) -> Dict[str, Any]:
    checklist_rows = _hit_concepts(answer, coerce_concept_groups(task.get("checklist_terms", [])))
    source_groups, domain_groups = _resolve_evidence_groups(task)
    source_rows = _hit_concepts(answer, source_groups)
    domain_rows = _hit_concepts(answer, domain_groups)
    continuity_rows = _hit_concepts(answer, coerce_concept_groups(task.get("continuity_terms", [])))
    contradiction_rows = _hit_contradictions(answer, coerce_contradiction_groups(task.get("contradiction_terms", [])))

    checklist_quality, checklist_hit_count, checklist_total = _rows_to_pct(checklist_rows)
    source_evidence_recall, source_hit_count, source_total = _rows_to_pct(source_rows)
    domain_rule_recall, domain_hit_count, domain_total = _rows_to_pct(domain_rows)
    continuity_recall, continuity_hit_count, continuity_total = _rows_to_pct(continuity_rows)

    # If one evidence side is absent, mirror the non-empty side so split weights stay stable.
    if source_total == 0 and domain_total > 0:
        source_evidence_recall = domain_rule_recall
        source_hit_count, source_total = domain_hit_count, domain_total
    elif domain_total == 0 and source_total > 0:
        domain_rule_recall = source_evidence_recall
        domain_hit_count, domain_total = source_hit_count, source_total

    if source_total > 0 and domain_total > 0:
        evidence_recall = (float(source_evidence_recall) + float(domain_rule_recall)) / 2.0
    elif source_total > 0:
        evidence_recall = source_evidence_recall
    elif domain_total > 0:
        evidence_recall = domain_rule_recall
    else:
        evidence_recall = None

    contradiction_penalty = sum(row["penalty"] for row in contradiction_rows)
    token_efficiency = max(0.0, min(1.0, 1.0 - (prompt_tokens / token_budget))) * 100.0
    latency_efficiency = max(0.0, min(1.0, 1.0 - (latency_seconds / latency_budget))) * 100.0

    return {
        "checklist_quality": checklist_quality,
        "source_evidence_recall": source_evidence_recall,
        "domain_rule_recall": domain_rule_recall,
        "evidence_recall": evidence_recall,
        "continuity_recall": continuity_recall,
        "token_efficiency": token_efficiency,
        "latency_efficiency": latency_efficiency,
        "contradiction_penalty": contradiction_penalty,
        "checklist_hit_count": checklist_hit_count,
        "checklist_total": checklist_total,
        "source_evidence_hit_count": source_hit_count,
        "source_evidence_total": source_total,
        "domain_rule_hit_count": domain_hit_count,
        "domain_rule_total": domain_total,
        "evidence_hit_count": source_hit_count + domain_hit_count,
        "evidence_total": source_total + domain_total,
        "continuity_hit_count": continuity_hit_count,
        "continuity_total": continuity_total,
        "contradiction_hit_count": len(contradiction_rows),
        "concept_hits": {
            "checklist_quality": checklist_rows,
            "source_evidence_recall": source_rows,
            "domain_rule_recall": domain_rows,
            "continuity_recall": continuity_rows,
        },
        "contradiction_hits": contradiction_rows,
    }

def composite_score(metrics: Dict[str, float], weights: Dict[str, float] = BASE_WEIGHTS) -> float:
    weighted_sum = 0.0
    weight_total = 0.0
    for metric, weight in weights.items():
        value = metrics.get(metric)
        if value is None:
            continue
        weighted_sum += float(value) * weight
        weight_total += weight
    normalized = (weighted_sum / weight_total) if weight_total > 0 else 0.0
    penalty = float(metrics.get("contradiction_penalty", 0.0))
    return max(0.0, normalized - penalty)

def term_breakdown(answer: str, task: Dict[str, Any]) -> Dict[str, Any]:
    scored = metric_scores(answer, task, prompt_tokens=0, latency_seconds=0)
    return {
        "checklist_hit_count": scored["checklist_hit_count"],
        "checklist_total": scored["checklist_total"],
        "source_evidence_hit_count": scored["source_evidence_hit_count"],
        "source_evidence_total": scored["source_evidence_total"],
        "domain_rule_hit_count": scored["domain_rule_hit_count"],
        "domain_rule_total": scored["domain_rule_total"],
        "evidence_hit_count": scored["evidence_hit_count"],
        "evidence_total": scored["evidence_total"],
        "continuity_hit_count": scored["continuity_hit_count"],
        "continuity_total": scored["continuity_total"],
        "contradiction_hit_count": scored["contradiction_hit_count"],
        "contradiction_penalty": scored["contradiction_penalty"],
        "concept_hits": scored["concept_hits"],
        "contradiction_hits": scored["contradiction_hits"],
    }
