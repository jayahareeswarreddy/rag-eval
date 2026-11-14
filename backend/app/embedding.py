"""
Embedding providers.

Two implementations, chosen deliberately:

* `tfidf` — a lexical baseline built from the corpus itself. Pure NumPy, no
  model download, no network, no API key. It always works, and it is the
  control the semantic provider has to beat. A RAG write-up that reports
  "recall@5 was 0.84" tells you nothing without a baseline next to it.
* `fastembed` — real sentence embeddings via ONNX (no PyTorch, ~90MB model),
  so it fits in a 512MB free-tier container.

The point of keeping both is not redundancy. Measuring the gap between them
on a specific corpus is the finding: on prose the semantic model wins
comfortably, on code with exact identifiers the lexical baseline is
surprisingly competitive, which is the argument for hybrid retrieval.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod

import numpy as np


# Function words carry no retrieval signal but do carry term frequency.
# On a large corpus IDF suppresses them automatically; on a small one -- an
# uploaded handful of documents -- there is not enough signal, and a query
# like "can I deploy on a friday" gets dominated by "on" matching "On-call".
# Filtering them explicitly makes retrieval behave the same at both scales.
STOPWORDS = frozenset("""
a an the and or but if then else for of to in on at by with from into over
under again further is are was were be been being am do does did doing have
has had having i me my we our you your he him his she her it its they them
their what which who whom this that these those as not no nor only own same
so than too very can will just should now how when where why all any both
each few more most other some such about against between during before after
above below up down out off through
""".split())

_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """
    Lowercase word tokens, splitting snake_case and camelCase.

    Identifiers matter for code search: a query for "get user session" should
    reach `get_user_session`, and `getUserSession` should produce the same
    three tokens as its snake_case twin.
    """
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    tokens: list[str] = []
    for raw in _TOKEN.findall(text.lower()):
        if raw in STOPWORDS:
            continue
        tokens.append(raw)
        if "_" in raw:
            tokens.extend(
                part for part in raw.split("_") if part and part not in STOPWORDS
            )
    return tokens


class Embedder(ABC):
    """Maps text to a dense vector. Vectors are L2-normalised."""

    name: str
    dim: int

    @abstractmethod
    def fit(self, corpus: list[str]) -> "Embedder":
        """Learn any corpus statistics. A no-op for pretrained models."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of unit vectors."""

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero vector stays zero rather than producing NaN.
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class TfidfEmbedder(Embedder):
    """
    TF-IDF over the corpus vocabulary, L2-normalised.

    Cosine similarity between these vectors is equivalent to classic TF-IDF
    ranking, which makes it a fair lexical control rather than a strawman.
    """

    name = "tfidf"

    def __init__(self, max_features: int = 20000) -> None:
        self.max_features = max_features
        self.vocabulary: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0, dtype=np.float32)
        self.dim = 0
        self._postings: dict[int, list[tuple[int, float]]] = {}
        self._rows = 0

    def fit(self, corpus: list[str]) -> "TfidfEmbedder":
        document_frequency: dict[str, int] = {}
        for document in corpus:
            for token in set(tokenize(document)):
                document_frequency[token] = document_frequency.get(token, 0) + 1

        # Keep the most frequent terms; rare one-off tokens mostly add noise.
        ranked = sorted(document_frequency.items(), key=lambda kv: (-kv[1], kv[0]))
        ranked = ranked[: self.max_features]

        self.vocabulary = {term: i for i, (term, _) in enumerate(ranked)}
        self.dim = len(self.vocabulary)
        n = max(len(corpus), 1)
        self.idf = np.zeros(self.dim, dtype=np.float32)
        for term, index in self.vocabulary.items():
            # Smoothed IDF, so a term in every document still has weight > 0.
            self.idf[index] = math.log((1 + n) / (1 + document_frequency[term])) + 1.0
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self.vocabulary:
            raise RuntimeError("TfidfEmbedder.fit must be called before encode")
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, int] = {}
            for token in tokenize(text):
                index = self.vocabulary.get(token)
                if index is not None:
                    counts[index] = counts.get(index, 0) + 1
            for index, count in counts.items():
                # Sublinear term frequency: the tenth occurrence of a word
                # says much less than the second.
                matrix[row, index] = (1.0 + math.log(count)) * self.idf[index]
        return _normalise(matrix)


    # ---------------------------------------------------------------- sparse

    def build_sparse(self, texts: list[str]) -> None:
        """
        Build an inverted index: term -> [(row, weight), ...].

        The dense alternative is a chunks x vocabulary matrix, which for 7,437
        chunks over a 20,000-term vocabulary is 595MB of mostly zeros and will
        not fit in a small container. Each chunk actually carries on the order
        of eighty distinct terms, so the same information as postings is a few
        megabytes. Scoring is exact cosine either way, because both sides stay
        L2-normalised; only the storage changes.
        """
        self._postings = {}
        self._rows = len(texts)
        for row, text in enumerate(texts):
            weights = self._weights(text)
            if not weights:
                continue
            for index, weight in weights.items():
                self._postings.setdefault(index, []).append((row, weight))

    def _weights(self, text: str) -> dict:
        """Normalised TF-IDF weights for one document, as {term_index: w}."""
        counts = {}
        for token in tokenize(text):
            index = self.vocabulary.get(token)
            if index is not None:
                counts[index] = counts.get(index, 0) + 1
        if not counts:
            return {}
        weights = {
            index: (1.0 + math.log(count)) * self.idf[index]
            for index, count in counts.items()
        }
        norm = math.sqrt(sum(w * w for w in weights.values()))
        if norm == 0:
            return {}
        return {index: w / norm for index, w in weights.items()}

    def score_all(self, query: str) -> np.ndarray:
        """
        Cosine of the query against every indexed document.

        Only documents sharing a term with the query are touched, so cost is
        proportional to the postings for the query's terms rather than to the
        size of the corpus.
        """
        scores = np.zeros(self._rows, dtype=np.float32)
        for index, weight in self._weights(query).items():
            for row, doc_weight in self._postings.get(index, ()):
                scores[row] += weight * doc_weight
        return scores


class FastEmbedEmbedder(Embedder):
    """
    Sentence embeddings via fastembed (ONNX runtime, no PyTorch).

    Optional: if the package or model is unavailable, construction raises and
    the caller falls back to TF-IDF rather than failing the run.
    """

    name = "fastembed"

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "fastembed is not installed; pip install fastembed"
            ) from exc
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def fit(self, corpus: list[str]) -> "FastEmbedEmbedder":
        return self  # pretrained

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = np.array(list(self._model.embed(texts)), dtype=np.float32)
        return _normalise(vectors)


def get_embedder(name: str = "auto", **kwargs) -> Embedder:
    """
    Build an embedder by name.

    "auto" prefers fastembed and degrades to TF-IDF, so the harness runs on a
    machine with no model cache and no network.
    """
    if name in ("fastembed", "auto"):
        try:
            return FastEmbedEmbedder(**kwargs)
        except Exception as exc:
            if name == "fastembed":
                raise
            print(f"[embedding] fastembed unavailable ({exc}); using tfidf")
    return TfidfEmbedder()
