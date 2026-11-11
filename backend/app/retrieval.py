"""
Retrieval: dense, lexical, and hybrid fusion.

Dense retrieval alone is a poor fit for code and for any corpus with exact
identifiers. A query for `get_user_session` should match that literal symbol,
but an embedding places it near every other session-ish function and the exact
match is not guaranteed to rank first. BM25 has the opposite failure: it
cannot match "how do I authenticate" to a function called `verify_token`
because they share no terms.

Fusing both covers each other's blind spot. This module implements all three
so the eval harness can report what the fusion is actually worth on a given
corpus, rather than assuming it helps.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .chunking import Chunk
from .embedding import Embedder, tokenize


@dataclass
class Hit:
    chunk: Chunk
    score: float
    # Per-retriever contributions, kept for explainability in the UI.
    detail: dict = None

    def __post_init__(self):
        if self.detail is None:
            self.detail = {}


class BM25:
    """
    Okapi BM25.

    k1 controls how quickly term frequency saturates; b controls how strongly
    long documents are penalised. The defaults are the standard ones and are
    not tuned here, because tuning them per corpus without a held-out set is
    how you fool yourself.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[list[str]] = []
        self.doc_freqs: list[Counter] = []
        self.idf: dict[str, float] = {}
        self.avg_len: float = 0.0

    def fit(self, texts: list[str]) -> "BM25":
        self.doc_tokens = [tokenize(t) for t in texts]
        self.doc_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        lengths = [len(t) for t in self.doc_tokens]
        self.avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0

        n = len(texts)
        document_frequency: Counter = Counter()
        for tokens in self.doc_tokens:
            document_frequency.update(set(tokens))

        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }
        return self

    def scores(self, query: str) -> np.ndarray:
        query_tokens = tokenize(query)
        out = np.zeros(len(self.doc_freqs), dtype=np.float32)
        for i, freqs in enumerate(self.doc_freqs):
            length = len(self.doc_tokens[i])
            total = 0.0
            for term in query_tokens:
                tf = freqs.get(term)
                if not tf:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = tf + self.k1 * (
                    1 - self.b + self.b * (length / (self.avg_len or 1))
                )
                total += idf * (tf * (self.k1 + 1)) / denom
            out[i] = total
        return out


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = 60, weights: list[float] | None = None
) -> dict[int, float]:
    """
    Combine ranked ID lists by Reciprocal Rank Fusion.

    RRF sums 1/(k + rank) across rankings. It is used here instead of
    normalising and adding raw scores because BM25 and cosine live on
    different, corpus-dependent scales — min-max normalising them makes the
    fusion sensitive to whichever retriever happened to produce an outlier.
    RRF only looks at positions, so it is scale-free.
    """
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank + 1)
    return fused


class Retriever:
    """
    Indexes chunks and answers queries in dense, lexical, or hybrid mode.

    The index is in-process NumPy. For the corpus sizes this harness targets
    (tens of thousands of chunks) a brute-force cosine is a few milliseconds,
    and it removes a network dependency from every experiment. `store.py`
    carries the Qdrant adapter for when persistence is wanted.
    """

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.bm25 = BM25()

    def index(self, chunks: list[Chunk]) -> "Retriever":
        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks]
        if not texts:
            self.matrix = np.zeros((0, max(self.embedder.dim, 1)), dtype=np.float32)
            return self
        self.embedder.fit(texts)
        self.matrix = self.embedder.encode(texts)
        self.bm25.fit(texts)
        return self

    def _dense_scores(self, query: str) -> np.ndarray:
        if not len(self.chunks):
            return np.zeros(0, dtype=np.float32)
        # Both sides are unit vectors, so the dot product is cosine.
        return self.matrix @ self.embedder.encode_one(query)

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        if not self.chunks:
            return []
        if mode not in ("dense", "bm25", "hybrid"):
            raise ValueError(f"unknown mode {mode!r}")

        dense = self._dense_scores(query) if mode in ("dense", "hybrid") else None
        lexical = self.bm25.scores(query) if mode in ("bm25", "hybrid") else None

        if mode == "dense":
            order = np.argsort(-dense)[:k]
            return [
                Hit(self.chunks[i], float(dense[i]), {"dense": float(dense[i])})
                for i in order
            ]
        if mode == "bm25":
            order = np.argsort(-lexical)[:k]
            return [
                Hit(self.chunks[i], float(lexical[i]), {"bm25": float(lexical[i])})
                for i in order
            ]

        # Hybrid: fuse the two rankings, then take the top k.
        depth = min(len(self.chunks), max(k * 10, 50))
        dense_rank = list(np.argsort(-dense)[:depth])
        lexical_rank = list(np.argsort(-lexical)[:depth])
        fused = reciprocal_rank_fusion([dense_rank, lexical_rank])

        order = sorted(fused, key=lambda i: -fused[i])[:k]
        return [
            Hit(
                self.chunks[i],
                float(fused[i]),
                {
                    "rrf": float(fused[i]),
                    "dense": float(dense[i]),
                    "bm25": float(lexical[i]),
                },
            )
            for i in order
        ]
