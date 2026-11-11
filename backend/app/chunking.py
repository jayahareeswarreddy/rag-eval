"""
Chunking strategies.

Chunking is the decision that most determines whether a RAG system works, and
it is usually made by accident — a fixed character count copied from a
tutorial. This module implements four strategies so they can be measured
against each other instead of assumed.

The interesting result is that the strategies trade off against each other:
small chunks retrieve precisely but give the generator too little context,
large chunks carry context but dilute the embedding so recall drops. There is
no globally best setting, only a best setting for a corpus and a question
shape, which is the argument for measuring rather than guessing.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable


@dataclass
class Chunk:
    """A retrievable unit of text plus where it came from."""

    text: str
    doc_id: str
    chunk_id: str
    # Ordinal position within the document, for reassembling context.
    position: int = 0
    # Free-form provenance: line numbers, symbol name, heading path.
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)


# --------------------------------------------------------------------------
# Strategy 1: fixed-size character windows
# --------------------------------------------------------------------------

def chunk_fixed(
    text: str, doc_id: str, *, size: int = 512, overlap: int = 64
) -> list[Chunk]:
    """
    Slice every `size` characters with `overlap` carried forward.

    The naive baseline. It splits mid-sentence and mid-word, so a chunk often
    begins halfway through the clause that gave it meaning. Included because
    it is what most tutorials teach, and because the other strategies are only
    interesting relative to it.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    chunks: list[Chunk] = []
    step = size - overlap
    for position, start in enumerate(range(0, max(len(text), 1), step)):
        window = text[start : start + size]
        if not window.strip():
            continue
        chunks.append(
            Chunk(
                text=window,
                doc_id=doc_id,
                chunk_id=f"{doc_id}::fixed::{position}",
                position=position,
                meta={"strategy": "fixed", "start": start, "end": start + len(window)},
            )
        )
        if start + size >= len(text):
            break
    return chunks


# --------------------------------------------------------------------------
# Strategy 2: recursive splitting on natural boundaries
# --------------------------------------------------------------------------

# Tried in order, coarsest first: paragraphs, then lines, then sentences,
# then words. Each level is only used when the level above leaves a piece
# that is still too large.
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_recursive(text: str, size: int, separators: list[str]) -> list[str]:
    if len(text) <= size:
        return [text] if text.strip() else []
    if not separators:
        # No boundary left to respect; fall back to a hard cut.
        return [text[i : i + size] for i in range(0, len(text), size)]

    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)

    out: list[str] = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer}{sep}{part}" if buffer else part
        if len(candidate) <= size:
            buffer = candidate
            continue
        if buffer:
            out.append(buffer)
        # The part alone may still exceed the budget: recurse on a finer
        # separator rather than cutting blindly.
        if len(part) > size:
            out.extend(_split_recursive(part, size, rest))
            buffer = ""
        else:
            buffer = part
    if buffer:
        out.append(buffer)
    return [piece for piece in out if piece.strip()]


def chunk_recursive(text: str, doc_id: str, *, size: int = 512) -> list[Chunk]:
    """
    Split on the coarsest natural boundary that fits the size budget.

    Respects paragraph, line and sentence boundaries, so a chunk is far more
    likely to be a complete thought than with fixed windows.
    """
    pieces = _split_recursive(text, size, _SEPARATORS)
    return [
        Chunk(
            text=piece,
            doc_id=doc_id,
            chunk_id=f"{doc_id}::recursive::{i}",
            position=i,
            meta={"strategy": "recursive"},
        )
        for i, piece in enumerate(pieces)
    ]


# --------------------------------------------------------------------------
# Strategy 3: structure-aware splitting for prose (Markdown headings)
# --------------------------------------------------------------------------

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.M)


def chunk_markdown(text: str, doc_id: str, *, size: int = 1024) -> list[Chunk]:
    """
    Split on heading boundaries and prefix each chunk with its heading path.

    The prefix matters more than the split. A section about "retry policy"
    under "## Queue" is ambiguous alone; embedding it as
    "Queue > retry policy: ..." puts the parent topic inside the vector, so a
    query naming the parent can still reach it.
    """
    matches = list(_HEADING.finditer(text))
    if not matches:
        return chunk_recursive(text, doc_id, size=size)

    chunks: list[Chunk] = []
    path: list[str] = []

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        path = path[: level - 1]
        path.append(title)
        heading_path = " > ".join(path)

        if not body:
            continue

        # A long section is split further, but every piece keeps the path.
        for j, piece in enumerate(_split_recursive(body, size, _SEPARATORS)):
            chunks.append(
                Chunk(
                    text=f"{heading_path}: {piece}",
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}::md::{i}::{j}",
                    position=len(chunks),
                    meta={
                        "strategy": "markdown",
                        "heading": title,
                        "heading_path": heading_path,
                    },
                )
            )
    return chunks


# --------------------------------------------------------------------------
# Strategy 4: AST-aware splitting for source code
# --------------------------------------------------------------------------

def chunk_python_ast(text: str, doc_id: str, *, size: int = 2000) -> list[Chunk]:
    """
    Split Python source at function and class boundaries.

    Fixed-size chunking is particularly destructive on code: it cuts a
    function in half, so neither piece is runnable, readable, or retrievable
    by a question about what that function does. Splitting on the AST keeps
    each definition whole and attaches the context a reader would need —
    module path, enclosing class, signature, docstring.

    Falls back to recursive splitting when the file does not parse, so a
    syntax error in one file cannot fail an ingest.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return chunk_recursive(text, doc_id, size=size)

    lines = text.splitlines()
    chunks: list[Chunk] = []

    def emit(node: ast.AST, qualname: str) -> None:
        start = node.lineno - 1
        end = getattr(node, "end_lineno", node.lineno)
        body = "\n".join(lines[start:end])
        if not body.strip():
            return

        docstring = ast.get_docstring(node) or ""
        kind = type(node).__name__.replace("Def", "").lower()
        # The header is what makes an isolated function findable: without it
        # the embedding has no idea which module or class it belongs to.
        header = f"{doc_id} {kind} {qualname}"
        if docstring:
            header += f" — {docstring.splitlines()[0]}"

        for j, piece in enumerate(_split_recursive(body, size, _SEPARATORS)):
            chunks.append(
                Chunk(
                    text=f"{header}\n{piece}",
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}::ast::{qualname}::{j}",
                    position=len(chunks),
                    meta={
                        "strategy": "ast",
                        "symbol": qualname,
                        "kind": kind,
                        "lineno": node.lineno,
                        "end_lineno": end,
                        "docstring": docstring[:200],
                    },
                )
            )

    def walk(node: ast.AST, prefix: str = "") -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                emit(child, f"{prefix}{child.name}")
            elif isinstance(child, ast.ClassDef):
                emit(child, f"{prefix}{child.name}")
                # Methods are emitted separately too: a question usually
                # targets one method, not a 400-line class.
                walk(child, prefix=f"{prefix}{child.name}.")

    walk(tree)

    # Module-level code outside any definition still needs to be searchable.
    covered = set()
    for chunk in chunks:
        meta = chunk.meta
        if "lineno" in meta:
            covered.update(range(meta["lineno"] - 1, meta["end_lineno"]))
    leftover = "\n".join(
        line for i, line in enumerate(lines) if i not in covered and line.strip()
    )
    if len(leftover.strip()) > 40:
        chunks.append(
            Chunk(
                text=f"{doc_id} module level\n{leftover[:size]}",
                doc_id=doc_id,
                chunk_id=f"{doc_id}::ast::__module__",
                position=len(chunks),
                meta={"strategy": "ast", "symbol": "__module__", "kind": "module"},
            )
        )

    return chunks or chunk_recursive(text, doc_id, size=size)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

Strategy = Callable[..., list[Chunk]]

STRATEGIES: dict[str, Strategy] = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
    "markdown": chunk_markdown,
    "ast": chunk_python_ast,
}


def chunk(text: str, doc_id: str, strategy: str = "recursive", **kwargs) -> list[Chunk]:
    """Chunk `text` with a named strategy."""
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; expected one of {sorted(STRATEGIES)}"
        )
    return STRATEGIES[strategy](text, doc_id, **kwargs)


def chunk_auto(text: str, doc_id: str, **kwargs) -> list[Chunk]:
    """Pick a structure-aware strategy from the document's extension."""
    if doc_id.endswith(".py"):
        return chunk_python_ast(text, doc_id, **kwargs)
    if doc_id.endswith((".md", ".markdown")):
        return chunk_markdown(text, doc_id, **kwargs)
    return chunk_recursive(text, doc_id, **kwargs)


def describe(chunks: Iterable[Chunk]) -> dict:
    """Summary statistics, reported alongside retrieval metrics."""
    sizes = [len(c) for c in chunks]
    if not sizes:
        return {"count": 0, "mean_chars": 0, "min_chars": 0, "max_chars": 0}
    return {
        "count": len(sizes),
        "mean_chars": round(sum(sizes) / len(sizes), 1),
        "min_chars": min(sizes),
        "max_chars": max(sizes),
    }
