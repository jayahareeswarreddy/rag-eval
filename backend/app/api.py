"""
FastAPI surface over the pipeline.

The index is built at startup and held in memory. For a corpus of this size
that is a few seconds and a few hundred megabytes, and it removes a database
from the deployment. `RAG_CORPUS_DIR` swaps in any local directory.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .chunking import chunk_auto, describe
from .corpus import (
    Corpus,
    build_questions_from_docstrings,
    load_directory,
    load_stdlib,
    split_questions,
)
from .embedding import get_embedder
from .evaluation import evaluate_retrieval, run_grid
from .generation import answer_question, get_generator
from .retrieval import Retriever

STATE: dict = {
    "corpus": None,
    "retriever": None,
    "chunk_stats": {},
    "index_seconds": 0.0,
    "questions": [],
    "config": {},
}


def build_index() -> None:
    """Load the corpus, chunk it, and build the retrieval index."""
    corpus_dir = os.getenv("RAG_CORPUS_DIR", "").strip()
    full = os.getenv("RAG_STDLIB_FULL", "0") == "1"
    strategy = os.getenv("RAG_CHUNKING", "auto")
    size = int(os.getenv("RAG_CHUNK_SIZE", "1024"))
    embedder_name = os.getenv("RAG_EMBEDDER", "auto")

    corpus: Corpus = (
        load_directory(corpus_dir) if corpus_dir else load_stdlib(full=full)
    )

    chunks = []
    for doc_id, text in corpus.documents.items():
        chunks.extend(chunk_auto(text, doc_id, size=size))

    started = time.perf_counter()
    retriever = Retriever(get_embedder(embedder_name)).index(chunks)
    elapsed = time.perf_counter() - started

    STATE.update(
        corpus=corpus,
        retriever=retriever,
        chunk_stats=describe(chunks),
        index_seconds=round(elapsed, 2),
        questions=build_questions_from_docstrings(corpus, per_document=2),
        config={
            "chunking": strategy,
            "chunk_size": size,
            "embedder": retriever.embedder.name,
        },
    )
    print(
        f"[index] {corpus.summary()['documents']} docs, "
        f"{STATE['chunk_stats']['count']} chunks in {elapsed:.1f}s "
        f"using {retriever.embedder.name}"
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    build_index()
    yield


app = FastAPI(
    title="RAG Evaluation Harness",
    description=(
        "A retrieval pipeline with the measurement apparatus attached: "
        "chunking strategies, hybrid retrieval, and citation-verified answers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_origin_regex=r"https://.*\.(github\.io|onrender\.com|vercel\.app)",
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="hybrid", pattern="^(dense|bm25|hybrid)$")


class AskRequest(SearchRequest):
    # Exposed because the right value is corpus-dependent; see generation.py.
    floor: float = Field(default=0.45, ge=0.0, le=1.0)


def _retriever() -> Retriever:
    retriever = STATE.get("retriever")
    if retriever is None:
        raise HTTPException(status_code=503, detail="index is still building")
    return retriever


@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
async def root():
    return {"service": "rag-eval", "docs": "/docs", "status": "running"}


@app.get("/health")
async def health():
    retriever = STATE.get("retriever")
    corpus = STATE.get("corpus")
    try:
        generator = get_generator()
        generator_name = generator.name
    except Exception:
        generator_name = "unavailable"

    return {
        "status": "healthy" if retriever is not None else "indexing",
        "corpus": corpus.summary() if corpus else None,
        "chunks": STATE["chunk_stats"],
        "index_seconds": STATE["index_seconds"],
        "config": STATE["config"],
        "generator": generator_name,
        "labelled_questions": len(STATE["questions"]),
    }


@app.post("/search")
async def search(request: SearchRequest):
    """Retrieve chunks. Returns per-retriever scores for explainability."""
    hits = _retriever().search(request.query, k=request.k, mode=request.mode)
    return {
        "query": request.query,
        "mode": request.mode,
        "hits": [
            {
                "rank": i,
                "doc_id": h.chunk.doc_id,
                "chunk_id": h.chunk.chunk_id,
                "score": round(h.score, 4),
                "detail": {k: round(v, 4) for k, v in h.detail.items()},
                "text": h.chunk.text[:1500],
                "meta": h.chunk.meta,
            }
            for i, h in enumerate(hits, start=1)
        ],
    }


@app.post("/ask")
async def ask(request: AskRequest):
    """Retrieve, generate, and verify every claim against its cited source."""
    hits = _retriever().search(request.query, k=request.k, mode=request.mode)
    answer = answer_question(request.query, hits, floor=request.floor)
    return answer.as_dict()


@app.get("/questions")
async def questions(limit: int = 20):
    """A sample of the labelled set, so the benchmark is inspectable."""
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=422, detail="limit must be 1..200")
    return {
        "total": len(STATE["questions"]),
        "questions": [
            {"id": q.id, "question": q.question, "relevant": q.relevant_doc_ids}
            for q in STATE["questions"][:limit]
        ],
    }


@app.post("/evaluate")
async def evaluate(mode: str = "hybrid", sample: int = 50):
    """Score the live index against the labelled set."""
    if mode not in ("dense", "bm25", "hybrid"):
        raise HTTPException(status_code=422, detail="mode must be dense|bm25|hybrid")
    if sample < 5 or sample > 500:
        raise HTTPException(status_code=422, detail="sample must be 5..500")

    dev, _ = split_questions(STATE["questions"])
    result = evaluate_retrieval(
        _retriever(),
        dev[:sample],
        config={**STATE["config"], "retrieval": mode},
        chunk_stats=STATE["chunk_stats"],
        index_seconds=STATE["index_seconds"],
        mode=mode,
    )
    return result.as_dict()


@app.get("/experiments")
async def experiments():
    """
    The stored grid results.

    Served from a file rather than recomputed: a full grid is minutes of CPU,
    and a web request is the wrong place to spend it.
    """
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "data" / "grid_results.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="no stored results; run scripts/run_experiments.py first",
        )
    return {"results": json.loads(path.read_text(encoding="utf-8"))}
