from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .ollama_client import OllamaClient

TOKEN_RE = re.compile(r"[a-zA-Z0-9_./:-]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class OllamaEmbedder:
    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.client.embed(self.model, texts), dtype=np.float32)


class TfidfEmbedder:
    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        docs = [set(tokenize(t)) for t in texts]
        vocab = sorted(set().union(*docs)) if docs else []
        self.vocab = {term: i for i, term in enumerate(vocab)}
        n = max(len(docs), 1)
        self.idf = {}
        for term in vocab:
            df = sum(1 for d in docs if term in d)
            self.idf[term] = math.log((1 + n) / (1 + df)) + 1
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        mat = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row, text in enumerate(texts):
            counts = Counter(tokenize(text))
            for term, count in counts.items():
                idx = self.vocab.get(term)
                if idx is not None:
                    mat[row, idx] = count * self.idf.get(term, 1.0)
        return normalize(mat)


def normalize(mat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return mat / norm


@dataclass
class SearchHit:
    id: str
    text: str
    score: float
    meta: dict


class VectorIndex:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.metas: list[dict] = []
        self.vecs: np.ndarray | None = None

    def build(self, docs: list[dict]) -> None:
        self.ids = [str(d['id']) for d in docs]
        self.texts = [str(d['text']) for d in docs]
        self.metas = [dict(d) for d in docs]
        if hasattr(self.embedder, 'fit'):
            self.embedder.fit(self.texts)  # type: ignore[attr-defined]
        self.vecs = normalize(self.embedder.embed(self.texts))

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        if self.vecs is None or not self.ids:
            return []
        q = normalize(self.embedder.embed([query]))[0]
        scores = self.vecs @ q
        order = np.argsort(-scores)[:k]
        return [SearchHit(self.ids[i], self.texts[i], float(scores[i]), self.metas[i]) for i in order]
