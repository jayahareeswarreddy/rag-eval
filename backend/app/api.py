"""
FastAPI surface over the pipeline.

The index is built at startup and held in memory. For a corpus of this size
that is a few seconds and a few hundred megabytes, and it removes a database
from the deployment. `RAG_CORPUS_DIR` swaps in any local directory.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .chunking import chunk_auto, describe
from .corpus import (
    Corpus,
    build_questions_from_docstrings,
    load_directory,
    load_stdlib,
    split_questions,
)
from .embedding import get_embedder, tokenize
from .evaluation import evaluate_retrieval, run_grid
from .generation import answer_question, get_generator
from .retrieval import Retriever
from .sessions import MAX_FILES, UploadRejected, store

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
    # TF-IDF by default, not "auto". "auto" prefers fastembed, and embedding
    # thousands of chunks through ONNX on CPU takes minutes where TF-IDF takes
    # seconds -- as a startup default that reads as a hung deploy. Real
    # embeddings are opt-in via RAG_EMBEDDER=fastembed, and the experiment
    # script can use them without blocking a web process.
    embedder_name = os.getenv("RAG_EMBEDDER", "tfidf")

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
    # When set, search the visitor's uploaded documents instead of the demo
    # corpus. An unknown or expired id is an error rather than a silent
    # fallback, so nobody is told their own files answered when they did not.
    session_id: str | None = Field(default=None, max_length=64)


class AskRequest(SearchRequest):
    # Exposed because the right value is corpus-dependent; see generation.py.
    floor: float = Field(default=0.45, ge=0.0, le=1.0)


def _retriever(session_id: str | None = None) -> Retriever:
    if session_id:
        session = store.get(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail="That upload session has expired. Upload the files again.",
            )
        if session.retriever is None:
            raise HTTPException(status_code=400, detail="no documents in that session")
        return session.retriever

    retriever = STATE.get("retriever")
    if retriever is None:
        raise HTTPException(status_code=503, detail="index is still building")
    return retriever


STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.get("/", include_in_schema=False)
async def index():
    """The UI and the API are one service: one deploy, no CORS in production."""
    return FileResponse(STATIC_DIR / "index.html")


@app.head("/", include_in_schema=False)
async def index_head():
    # Platform probes and uptime monitors default to HEAD.
    return FileResponse(STATIC_DIR / "index.html")


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
    hits = _retriever(request.session_id).search(
        request.query, k=request.k, mode=request.mode
    )

    # Retrieval always returns k results, whether or not any of them are
    # relevant. Without a signal for that, a visitor reads noise as an answer:
    # a query about something the corpus never mentions still produces five
    # confident-looking passages. Coverage is the share of the question's
    # content words that actually appear in a passage, which is interpretable
    # and does not depend on a score scale that shifts per corpus.
    query_tokens = {t for t in tokenize(request.query) if len(t) > 2}

    def coverage(text: str) -> float:
        if not query_tokens:
            return 0.0
        return len(query_tokens & set(tokenize(text))) / len(query_tokens)

    coverages = [coverage(h.chunk.text) for h in hits]
    best_coverage = max(coverages, default=0.0)

    return {
        "query": request.query,
        "scope": "uploaded" if request.session_id else "corpus",
        "best_coverage": round(best_coverage, 3),
        # Under a third of the question's terms found anywhere in the top
        # results means these are the least-bad matches, not answers.
        "weak_match": best_coverage < 0.34,
        "mode": request.mode,
        "hits": [
            {
                "rank": i,
                "coverage": round(c, 3),
                "doc_id": h.chunk.doc_id,
                "chunk_id": h.chunk.chunk_id,
                "score": round(h.score, 4),
                "detail": {k: round(v, 4) for k, v in h.detail.items()},
                "text": h.chunk.text[:1500],
                "meta": h.chunk.meta,
            }
            for i, (h, c) in enumerate(zip(hits, coverages), start=1)
        ],
    }


@app.post("/ask")
async def ask(request: AskRequest):
    """Retrieve, generate, and verify every claim against its cited source."""
    hits = _retriever(request.session_id).search(
        request.query, k=request.k, mode=request.mode
    )
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


@app.post("/upload")
async def upload(
    files: list[UploadFile] = File(...),
    session_id: str | None = None,
):
    """
    Index a visitor's own documents into a temporary session.

    Text and source files only. Everything is held in memory, capped, and
    expires; nothing is written to disk. The response reports which files were
    skipped and why, so a partial upload is visible rather than silent.
    """
    if not files:
        raise HTTPException(status_code=422, detail="no files were sent")
    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=422, detail=f"at most {MAX_FILES} files per upload"
        )

    payload: list[tuple[str, bytes]] = []
    for item in files:
        payload.append((item.filename or "untitled", await item.read()))

    try:
        session, skipped = store.add(payload, session_id=session_id)
    except UploadRejected as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not session.documents:
        raise HTTPException(
            status_code=422,
            detail={"message": "nothing could be indexed", "skipped": skipped},
        )

    return {**session.summary(), "skipped": skipped}


@app.get("/sessions/{session_id}")
async def session_info(session_id: str):
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found or expired")
    return session.summary()


@app.delete("/sessions/{session_id}")
async def session_delete(session_id: str):
    """Let a visitor remove their documents before the TTL does."""
    if not store.drop(session_id):
        raise HTTPException(status_code=404, detail="session not found or expired")
    return {"deleted": session_id}


@app.get("/upload-limits")
async def upload_limits():
    """The caps, so the UI can state them instead of discovering them."""
    return store.stats()


# Named explicitly rather than mounted, so only these files are reachable
# and a new asset cannot be served by accident.
_ASSETS = {
    "app.css": "text/css",
    "app.js": "text/javascript",
    "upload.js": "text/javascript",
}


@app.get("/{asset}", include_in_schema=False)
async def asset(asset: str):
    media_type = _ASSETS.get(asset)
    if media_type is None:
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(STATIC_DIR / asset, media_type=media_type)
