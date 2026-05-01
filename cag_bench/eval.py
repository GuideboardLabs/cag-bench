from __future__ import annotations

import math
from typing import Any

import numpy as np

from .vector import Embedder, tokenize


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an == 0 or bn == 0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def semantic_coverage(answer: str, checklist: list[str], embedder: Embedder, threshold: float = 0.42) -> tuple[float, list[dict[str, Any]]]:
    if not checklist:
        return 1.0, []
    texts = [answer] + checklist
    if hasattr(embedder, 'fit'):
        embedder.fit(texts)  # type: ignore[attr-defined]
    vecs = embedder.embed(texts)
    ans = vecs[0]
    details = []
    hits = 0
    answer_tokens = set(tokenize(answer))
    for i, item in enumerate(checklist, start=1):
        sim = _cos(ans, vecs[i])
        item_tokens = [t for t in tokenize(item) if len(t) > 3]
        lexical = 0.0
        if item_tokens:
            lexical = sum(1 for t in item_tokens if t in answer_tokens) / len(item_tokens)
        passed = sim >= threshold or lexical >= 0.45
        hits += int(passed)
        details.append({'item': item, 'similarity': sim, 'lexical': lexical, 'passed': passed})
    return hits / len(checklist), details


def id_recall(selected_ids: list[str], expected_ids: list[str]) -> float:
    if not expected_ids:
        return 1.0
    s = set(selected_ids)
    e = set(expected_ids)
    return len(s & e) / len(e)


def safe_norm_cost(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return min(max((value - low) / (high - low), 0.0), 1.0)


def composite_score(
    *,
    task_quality: float,
    evidence_recall: float,
    memory_recall: float,
    prompt_tokens: int,
    seconds: float,
    token_low: int = 500,
    token_high: int = 12000,
    seconds_low: float = 1.0,
    seconds_high: float = 120.0,
) -> float:
    # v3 continuity-trap composite: correctness remains primary, but continuity gets enough
    # weight to expose whether a method carries prior decisions across related tasks.
    token_penalty = safe_norm_cost(prompt_tokens, token_low, token_high)
    time_penalty = safe_norm_cost(seconds, seconds_low, seconds_high)
    raw = (
        0.40 * task_quality
        + 0.20 * evidence_recall
        + 0.25 * memory_recall
        + 0.10 * (1.0 - token_penalty)
        + 0.05 * (1.0 - time_penalty)
    )
    return round(100 * raw, 2)
