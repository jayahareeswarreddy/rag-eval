"""
Corpora and labelled question sets.

A benchmark is only as honest as the data under it. Two things went wrong in
early versions of this harness and both are guarded against here:

1. **A corpus too small to discriminate.** With three documents every
   configuration scored a perfect 1.000, which says nothing about which is
   better. A useful corpus needs enough near-miss documents that retrieval can
   actually fail.
2. **Label leakage.** If a question is derived from a docstring that names its
   own module, any lexical retriever wins by copying the name back. Questions
   whose text leaks the answer are filtered out, which is why the generated set
   is smaller than the number of docstrings available.

The default corpus is the Python standard library, already on disk for anyone
running this. No download, no API key, and large enough to be hard.
"""

from __future__ import annotations

import ast
import random
import re
import sysconfig
from dataclasses import dataclass
from pathlib import Path

from .evaluation import Question

# Modules that are large, pure-Python and thematically distinct enough that
# questions about them have a defensible single right answer.
DEFAULT_MODULES = [
    "argparse", "ast", "base64", "bisect", "calendar", "cmd", "codeop",
    "colorsys", "configparser", "copy", "csv", "dataclasses", "datetime",
    "difflib", "dis", "enum", "filecmp", "fnmatch", "fractions", "ftplib",
    "getopt", "getpass", "gettext", "glob", "gzip", "hashlib", "heapq",
    "hmac", "imaplib", "inspect", "ipaddress", "json/decoder", "json/encoder",
    "keyword", "linecache", "locale", "mailbox", "mimetypes", "modulefinder",
    "netrc", "ntpath", "nturl2path", "numbers", "operator", "optparse",
    "pathlib", "pickle", "pickletools", "pipes", "pkgutil", "platform",
    "plistlib", "poplib", "posixpath", "pprint", "profile", "pstats", "pty",
    "queue", "quopri", "random", "reprlib", "sched", "secrets", "selectors",
    "shelve", "shlex", "shutil", "signal", "smtplib", "socket", "socketserver",
    "sndhdr", "statistics", "string", "stringprep", "struct", "subprocess",
    "symtable", "tabnanny", "tarfile", "telnetlib", "tempfile", "textwrap",
    "threading", "timeit", "token", "tokenize", "trace", "traceback",
    "tracemalloc", "tty", "types", "typing", "uu", "uuid", "warnings", "wave",
    "weakref", "webbrowser", "zipfile",
]


@dataclass
class Corpus:
    name: str
    documents: dict[str, str]

    @property
    def total_chars(self) -> int:
        return sum(len(t) for t in self.documents.values())

    def summary(self) -> dict:
        return {
            "name": self.name,
            "documents": len(self.documents),
            "total_chars": self.total_chars,
            "mean_doc_chars": (
                round(self.total_chars / len(self.documents))
                if self.documents
                else 0
            ),
        }


def load_stdlib(
    modules: list[str] | None = None,
    max_chars: int = 200_000,
    full: bool = False,
) -> Corpus:
    """
    Read pure-Python stdlib modules from the running interpreter.

    `full=True` walks the whole stdlib tree instead of the curated list. That
    is the setting worth using: with only ~90 documents recall@5 saturates
    near 1.0 and every configuration looks identical. Several hundred
    documents, many of them thematically adjacent, is what makes the retrieval
    task discriminating.
    """
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    documents: dict[str, str] = {}
    skip_parts = {"test", "tests", "idlelib", "lib2to3", "turtledemo", "__pycache__"}

    if full:
        for path in sorted(stdlib.rglob("*.py")):
            if any(part in skip_parts for part in path.parts):
                continue
            if path.name.startswith("test_"):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if 500 <= len(text) <= max_chars:
                key = str(path.relative_to(stdlib)).replace("\\", "/")
                documents[key] = text
        return Corpus(name="python-stdlib-full", documents=documents)

    for name in modules or DEFAULT_MODULES:
        path = stdlib / f"{name}.py"
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if len(text) > max_chars or len(text) < 500:
            continue
        documents[f"{name}.py"] = text

    return Corpus(name="python-stdlib", documents=documents)


def load_directory(
    root: str | Path,
    patterns: tuple[str, ...] = ("*.py", "*.md"),
    max_files: int = 400,
    max_chars: int = 200_000,
) -> Corpus:
    """Load a local directory — point this at any repository."""
    root = Path(root)
    documents: dict[str, str] = {}
    skip = {".git", "node_modules", "__pycache__", ".venv", "venv", "build", "dist"}

    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if any(part in skip for part in path.parts):
                continue
            if len(documents) >= max_files:
                break
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if 200 < len(text) <= max_chars:
                documents[str(path.relative_to(root)).replace("\\", "/")] = text

    return Corpus(name=f"dir:{root.name}", documents=documents)


# --------------------------------------------------------------------------
# Building a labelled question set
# --------------------------------------------------------------------------

_ARTICLE = re.compile(r"^(return|returns|a|an|the)\b\s*", re.I)


def _looks_usable(summary: str) -> bool:
    """Reject docstring lines that cannot serve as a question."""
    if not (25 <= len(summary) <= 160):
        return False
    words = summary.split()
    if len(words) < 5:
        return False
    # Doctest fragments, headers and code are not natural questions.
    if any(token in summary for token in (">>>", "::", "http", "--", "()")):
        return False
    return summary[0].isalpha()


def _leaks_answer(summary: str, doc_id: str) -> bool:
    """
    True when the question text contains the module name.

    "Convert a datetime to a string" as a question about datetime.py is not a
    retrieval test, it is a string match. Dropping these is what keeps the
    benchmark from flattering lexical retrieval.
    """
    stem = doc_id.rsplit("/", 1)[-1].removesuffix(".py").lower()
    haystack = summary.lower()
    if stem in haystack:
        return True
    # Also catch the singular/plural and underscore-free forms.
    return stem.rstrip("s") in haystack.replace("_", "")


def build_questions_from_docstrings(
    corpus: Corpus,
    *,
    per_document: int = 3,
    seed: int = 7,
    min_symbol_len: int = 4,
) -> list[Question]:
    """
    Derive questions from function and class docstrings.

    This is the CodeSearchNet construction: a docstring summary is a natural
    language description of code written by someone who was not thinking about
    retrieval, and the file it came from is the ground truth. It scales to
    hundreds of labelled examples without hand-annotation, and it is
    reproducible by anyone who runs the script.

    Its limitation, stated plainly: a docstring is a *description*, not a real
    user question. Scores here are not directly comparable to a hand-written
    set, and `data/questions.json` carries a smaller human-written set for
    exactly that reason.
    """
    rng = random.Random(seed)
    questions: list[Question] = []

    for doc_id, text in corpus.documents.items():
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        candidates: list[tuple[str, str]] = []
        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            if node.name.startswith("_") or len(node.name) < min_symbol_len:
                continue
            docstring = ast.get_docstring(node)
            if not docstring:
                continue

            summary = docstring.strip().splitlines()[0].strip()
            summary = _ARTICLE.sub("", summary).strip().rstrip(".")
            if not _looks_usable(summary):
                continue
            if _leaks_answer(summary, doc_id):
                continue
            candidates.append((node.name, summary))

        rng.shuffle(candidates)
        for symbol, summary in candidates[:per_document]:
            questions.append(
                Question(
                    id=f"{doc_id}::{symbol}",
                    question=summary,
                    relevant_doc_ids=[doc_id],
                    tags=["docstring", "code"],
                )
            )

    return questions


def split_questions(
    questions: list[Question], holdout: float = 0.3, seed: int = 11
) -> tuple[list[Question], list[Question]]:
    """
    Split into development and held-out sets.

    Any threshold or weight tuned while watching a score must be tuned on the
    development half only. Reporting a number from the same data used to choose
    the configuration is the most common way a retrieval benchmark ends up
    meaningless.
    """
    shuffled = list(questions)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1 - holdout))
    return shuffled[:cut], shuffled[cut:]
