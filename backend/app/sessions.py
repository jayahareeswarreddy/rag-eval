"""
Per-visitor document sessions.

A fixed demo corpus only shows that retrieval works on documents the author
picked. Letting a visitor bring their own turns a demonstration into a tool,
and it is the more honest test: their documents are ones nothing here was
tuned against.

Scope is deliberately narrow: UTF-8 text and source files. Decoding text and
indexing it is a small, well-understood operation. Supporting document formats
would mean pulling in parsers for untrusted binary input, which is a much
larger surface for very little gain in a retrieval demo.

The limits below are the feature rather than paperwork around it:

* File count, per-file size and total characters are capped, so one visitor
  cannot exhaust the process.
* Sessions expire, and the oldest is evicted at capacity, because indexes are
  held in memory and memory is the binding constraint on a small container.
* Nothing is written to disk and nothing survives a restart.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import Lock

from .chunking import chunk_auto, describe
from .embedding import get_embedder
from .retrieval import Retriever

MAX_FILES = 20
MAX_FILE_BYTES = 1_000_000
MAX_SESSION_CHARS = 3_000_000
MAX_SESSIONS = 25
SESSION_TTL_SECONDS = 3600
MIN_DOC_CHARS = 40

ALLOWED_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".log",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rb", ".rs",
    ".c", ".h", ".cpp", ".cs", ".sh", ".sql",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".tsv",
}


class UploadRejected(ValueError):
    """The upload cannot be indexed. The message is safe to show a visitor."""


def suffix_of(filename: str) -> str:
    name = (filename or "").strip().lower()
    dot = name.rfind(".")
    return name[dot:] if dot != -1 else ""


def safe_name(filename: str) -> str:
    """
    Reduce an uploaded name to a plain label.

    The name is only ever shown back to the visitor and used as a dictionary
    key, never as a path, so stripping directory separators keeps it from
    reading like one.
    """
    name = (filename or "untitled").replace("\\", "/").split("/")[-1]
    cleaned = "".join(c for c in name if c.isalnum() or c in "._- ").strip()
    return (cleaned or "untitled")[:120]


def decode(filename: str, data: bytes) -> str:
    """Decode an uploaded file as text, or explain why it cannot be."""
    if len(data) > MAX_FILE_BYTES:
        raise UploadRejected(
            f"{safe_name(filename)} is over {MAX_FILE_BYTES // 1000}KB"
        )

    suffix = suffix_of(filename)
    if suffix not in ALLOWED_SUFFIXES:
        raise UploadRejected(
            f"{safe_name(filename)}: {suffix or 'no extension'} is not supported. "
            "Plain text, Markdown and source files only."
        )

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise UploadRejected(
            f"{safe_name(filename)} is not valid UTF-8 text"
        ) from None

    # A NUL byte means this is binary content wearing a text extension.
    if "\x00" in text:
        raise UploadRejected(f"{safe_name(filename)} looks like binary data")

    return text


@dataclass
class Session:
    id: str
    created: float = field(default_factory=time.time)
    touched: float = field(default_factory=time.time)
    documents: dict[str, str] = field(default_factory=dict)
    retriever: Retriever | None = None
    chunk_stats: dict = field(default_factory=dict)
    index_seconds: float = 0.0

    @property
    def total_chars(self) -> int:
        return sum(len(t) for t in self.documents.values())

    def summary(self) -> dict:
        return {
            "session_id": self.id,
            "documents": len(self.documents),
            "filenames": list(self.documents),
            "total_chars": self.total_chars,
            "chunks": self.chunk_stats,
            "index_seconds": round(self.index_seconds, 3),
            "expires_in_seconds": max(
                0, int(SESSION_TTL_SECONDS - (time.time() - self.touched))
            ),
        }


class SessionStore:
    """In-memory sessions with a TTL and a hard capacity."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = Lock()

    def _evict(self) -> None:
        """Drop expired sessions, then the oldest if still at capacity."""
        now = time.time()
        for sid in [
            sid
            for sid, s in self._sessions.items()
            if now - s.touched > SESSION_TTL_SECONDS
        ]:
            self._sessions.pop(sid, None)

        while len(self._sessions) >= MAX_SESSIONS:
            oldest = min(self._sessions, key=lambda sid: self._sessions[sid].touched)
            self._sessions.pop(oldest, None)

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if time.time() - session.touched > SESSION_TTL_SECONDS:
                self._sessions.pop(session_id, None)
                return None
            session.touched = time.time()
            return session

    def drop(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def add(self, files: list[tuple[str, bytes]], session_id: str | None = None):
        """
        Decode, index, and return (session, skipped).

        `skipped` lists files that could not be used and why, so a partial
        upload reports what happened instead of failing silently.
        """
        with self._lock:
            session = None
            if session_id:
                session = self._sessions.get(session_id)
            if session is None:
                self._evict()
                session = Session(id=uuid.uuid4().hex[:16])
                self._sessions[session.id] = session

        skipped: list[dict] = []

        for filename, data in files:
            if len(session.documents) >= MAX_FILES:
                skipped.append(
                    {"file": safe_name(filename), "reason": f"over {MAX_FILES} files"}
                )
                continue
            try:
                text = decode(filename, data)
            except UploadRejected as exc:
                skipped.append({"file": safe_name(filename), "reason": str(exc)})
                continue

            if len(text.strip()) < MIN_DOC_CHARS:
                skipped.append(
                    {"file": safe_name(filename), "reason": "too short to index"}
                )
                continue
            if session.total_chars + len(text) > MAX_SESSION_CHARS:
                skipped.append(
                    {"file": safe_name(filename), "reason": "session size limit reached"}
                )
                continue

            name = safe_name(filename)
            # Keep both files when two uploads share a name.
            if name in session.documents:
                name = f"{len(session.documents)}-{name}"
            session.documents[name] = text

        if session.documents:
            self._reindex(session)

        return session, skipped

    def _reindex(self, session: Session) -> None:
        """Rebuild the session index from scratch after an upload."""
        chunks = []
        for doc_id, text in session.documents.items():
            chunks.extend(chunk_auto(text, doc_id, size=1024))

        started = time.perf_counter()
        # A fresh embedder per session: TF-IDF statistics are learned from the
        # corpus, and reusing the demo corpus vocabulary would score these
        # documents against the wrong term distribution.
        session.retriever = Retriever(get_embedder("tfidf")).index(chunks)
        session.index_seconds = time.perf_counter() - started
        session.chunk_stats = describe(chunks)
        session.touched = time.time()

    def stats(self) -> dict:
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "max_sessions": MAX_SESSIONS,
                "ttl_seconds": SESSION_TTL_SECONDS,
                "limits": {
                    "max_files": MAX_FILES,
                    "max_file_bytes": MAX_FILE_BYTES,
                    "max_session_chars": MAX_SESSION_CHARS,
                    "allowed_suffixes": sorted(ALLOWED_SUFFIXES),
                },
            }


store = SessionStore()
