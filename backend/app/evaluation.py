"""
Retrieval and answer metrics, and the experiment grid that uses them.

Everything here answers one question: *how do you know your RAG works?*

The metrics are deliberately boring and standard — recall@k, MRR, nDCG,
precision@k — because inventing a bespoke score is how a project becomes
unfalsifiable. The value is in running them across a grid of configurations
and reporting what actually changed, including when a change made things
worse.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Callable, Iterable

from .chunking import Chunk, chunk as chunk_text, describe
from .embedding import get_embedder
from .retrieval import Retriever


@dataclass
class Question:
    """
    One labelled example.

    `relevant_doc_ids` is the ground truth: the documents that genuinely
    answer the question. Judging at document level rather than chunk level is
    deliberate — chunk ids change whenever the chunking strategy changes, so
    chunk-level labels could not be reused across the very comparison this
    harness exists to run.
    """

    id: str
    question: str
    relevant_doc_ids: list[str]
    # Optional: a phrase that must appear in a correct answer.
    expected_substring: str | None = None
    tags: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Ranking metrics
# --------------------------------------------------------------------------

def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of relevant documents that appear in the top k."""
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & set(relevant)) / len(set(relevant))


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of the top k that are relevant."""
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    relevant_set = set(relevant)
    return sum(1 for doc in top if doc in relevant_set) / len(top)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """1/rank of the first relevant hit; 0 if none was retrieved."""
    relevant_set = set(relevant)
    for i, doc in enumerate(retrieved):
        if doc in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Normalised discounted cumulative gain, binary relevance.

    Unlike recall, nDCG cares where in the list a hit lands — useful when the
    generator only sees the top few chunks, so rank 1 is worth more than
    rank 5 even though recall treats them identically.
    """
    relevant_set = set(relevant)
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, doc in enumerate(retrieved[:k])
        if doc in relevant_set
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_set), k)))
    return dcg / ideal if ideal else 0.0


# --------------------------------------------------------------------------
# Running an evaluation
# --------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Aggregate metrics for one configuration over the whole question set."""

    config: dict
    questions: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    mrr: float
    ndcg_at_5: float
    chunk_stats: dict
    index_seconds: float
    query_ms_mean: float
    # Questions where nothing relevant was retrieved at all, by id. These are
    # the only rows worth reading by hand.
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_retrieval(
    retriever: Retriever,
    questions: Iterable[Question],
    config: dict,
    chunk_stats: dict,
    index_seconds: float,
    mode: str = "hybrid",
    max_k: int = 10,
) -> RetrievalResult:
    """Run every question against an indexed retriever and aggregate."""
    questions = list(questions)
    recalls = {1: [], 3: [], 5: [], 10: []}
    precisions, rrs, ndcgs, latencies = [], [], [], []
    failures: list[str] = []

    for question in questions:
        started = time.perf_counter()
        hits = retriever.search(question.question, k=max_k, mode=mode)
        latencies.append((time.perf_counter() - started) * 1000)

        # Collapse chunk hits to document ids, preserving rank order and
        # keeping only the first appearance of each document.
        seen: set[str] = set()
        retrieved_docs: list[str] = []
        for hit in hits:
            if hit.chunk.doc_id not in seen:
                seen.add(hit.chunk.doc_id)
                retrieved_docs.append(hit.chunk.doc_id)

        for k in recalls:
            recalls[k].append(recall_at_k(retrieved_docs, question.relevant_doc_ids, k))
        precisions.append(precision_at_k(retrieved_docs, question.relevant_doc_ids, 5))
        rr = reciprocal_rank(retrieved_docs, question.relevant_doc_ids)
        rrs.append(rr)
        ndcgs.append(ndcg_at_k(retrieved_docs, question.relevant_doc_ids, 5))
        if rr == 0.0:
            failures.append(question.id)

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return RetrievalResult(
        config=config,
        questions=len(questions),
        recall_at_1=mean(recalls[1]),
        recall_at_3=mean(recalls[3]),
        recall_at_5=mean(recalls[5]),
        recall_at_10=mean(recalls[10]),
        precision_at_5=mean(precisions),
        mrr=mean(rrs),
        ndcg_at_5=mean(ndcgs),
        chunk_stats=chunk_stats,
        index_seconds=round(index_seconds, 3),
        query_ms_mean=round(mean(latencies), 2),
        failures=failures,
    )


# --------------------------------------------------------------------------
# The experiment grid
# --------------------------------------------------------------------------

def run_grid(
    documents: dict[str, str],
    questions: list[Question],
    *,
    strategies: list[str] | None = None,
    sizes: list[int] | None = None,
    modes: list[str] | None = None,
    embedders: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[RetrievalResult]:
    """
    Evaluate the cartesian product of chunking, size, retrieval mode and
    embedder, returning one result per configuration sorted by recall@5.

    Each configuration is indexed from scratch. That is slower than reusing an
    index, but reusing one would silently share chunking across runs, which is
    exactly the variable under test.
    """
    strategies = strategies or ["fixed", "recursive", "auto"]
    sizes = sizes or [256, 512, 1024]
    modes = modes or ["dense", "bm25", "hybrid"]
    embedders = embedders or ["tfidf"]

    results: list[RetrievalResult] = []

    for strategy, size, embedder_name in product(strategies, sizes, embedders):
        label = f"{strategy}/{size}/{embedder_name}"
        if progress:
            progress(f"indexing {label}")

        chunks: list[Chunk] = []
        for doc_id, text in documents.items():
            if strategy == "auto":
                from .chunking import chunk_auto

                chunks.extend(chunk_auto(text, doc_id, size=size))
            elif strategy == "fixed":
                chunks.extend(
                    chunk_text(text, doc_id, "fixed", size=size, overlap=size // 8)
                )
            else:
                chunks.extend(chunk_text(text, doc_id, strategy, size=size))

        if not chunks:
            continue

        started = time.perf_counter()
        retriever = Retriever(get_embedder(embedder_name)).index(chunks)
        index_seconds = time.perf_counter() - started
        stats = describe(chunks)

        for mode in modes:
            if progress:
                progress(f"evaluating {label}/{mode}")
            results.append(
                evaluate_retrieval(
                    retriever,
                    questions,
                    config={
                        "chunking": strategy,
                        "chunk_size": size,
                        "retrieval": mode,
                        "embedder": embedder_name,
                    },
                    chunk_stats=stats,
                    index_seconds=index_seconds,
                    mode=mode,
                )
            )

    results.sort(key=lambda r: (-r.recall_at_5, -r.mrr))
    return results


def format_table(results: list[RetrievalResult], limit: int = 20) -> str:
    """Render results as a fixed-width table for the terminal and the README."""
    header = (
        f"{'chunking':<10} {'size':>5} {'retrieval':<9} {'embed':<9} "
        f"{'chunks':>7} {'R@1':>6} {'R@5':>6} {'MRR':>6} {'nDCG@5':>7} {'ms':>7}"
    )
    lines = [header, "-" * len(header)]
    for r in results[:limit]:
        c = r.config
        lines.append(
            f"{c['chunking']:<10} {c['chunk_size']:>5} {c['retrieval']:<9} "
            f"{c['embedder']:<9} {r.chunk_stats['count']:>7} "
            f"{r.recall_at_1:>6.3f} {r.recall_at_5:>6.3f} {r.mrr:>6.3f} "
            f"{r.ndcg_at_5:>7.3f} {r.query_ms_mean:>7.2f}"
        )
    return "\n".join(lines)
