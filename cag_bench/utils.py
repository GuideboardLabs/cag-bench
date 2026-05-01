
import json, re
from pathlib import Path
from typing import Iterable, Dict, Any, List

def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower()).strip()

def contains_term(text: str, term: str) -> bool:
    return normalize(term) in normalize(text)

def count_terms(text: str, terms: list[str]) -> int:
    return sum(1 for term in terms if contains_term(text, term))

def pct(part: int, whole: int) -> float:
    return 1.0 if whole <= 0 else part / whole

def ensure_str_list(values: Any) -> list[str]:
    if not values:
        return []
    out = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            out.append(text)
    return out

def coerce_concept_groups(values: Any) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    if not values:
        return groups
    for value in values:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            groups.append({"concept": text, "accepted_terms": [text]})
            continue
        if isinstance(value, dict):
            concept = str(value.get("concept") or "").strip()
            accepted_terms = ensure_str_list(value.get("accepted_terms", []))
            if not concept and accepted_terms:
                concept = accepted_terms[0]
            if concept and not accepted_terms:
                accepted_terms = [concept]
            if concept:
                groups.append({"concept": concept, "accepted_terms": accepted_terms})
    return groups

def coerce_contradiction_groups(values: Any) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    if not values:
        return groups
    for value in values:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            groups.append({"concept": text, "bad_terms": [text], "penalty": 0.0})
            continue
        if isinstance(value, dict):
            concept = str(value.get("concept") or "").strip()
            bad_terms = ensure_str_list(value.get("bad_terms", []))
            if not concept and bad_terms:
                concept = bad_terms[0]
            if concept and not bad_terms:
                bad_terms = [concept]
            if not concept:
                continue
            try:
                penalty = float(value.get("penalty", 0.0))
            except Exception:
                penalty = 0.0
            groups.append({"concept": concept, "bad_terms": bad_terms, "penalty": penalty})
    return groups

def extract_terms(values: Any) -> list[str]:
    out: list[str] = []
    for group in coerce_concept_groups(values):
        out.extend(group["accepted_terms"])
    return out
