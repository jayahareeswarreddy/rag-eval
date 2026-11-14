#!/usr/bin/env python
"""
Run the experiment grid and write data/grid_results.json.

Kept out of the request path deliberately: a full grid is minutes of CPU, and
a web request is the wrong place to spend it. The API serves the stored file.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# app/ and data/ live under backend/, not at the repository root.
BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

from app.corpus import (  # noqa: E402
    build_questions_from_docstrings,
    load_directory,
    load_stdlib,
    split_questions,
)
from app.evaluation import format_table, run_grid  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", help="index a local directory instead of the stdlib")
    parser.add_argument("--full", action="store_true", help="walk the whole stdlib (~875 docs)")
    parser.add_argument("--questions", type=int, default=300, help="cap the dev set")
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2000])
    parser.add_argument("--strategies", nargs="+", default=["fixed", "recursive", "auto"])
    parser.add_argument("--modes", nargs="+", default=["dense", "bm25", "hybrid"])
    parser.add_argument("--embedders", nargs="+", default=["tfidf"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    corpus = (
        load_directory(args.corpus_dir) if args.corpus_dir else load_stdlib(full=args.full)
    )
    questions = build_questions_from_docstrings(corpus, per_document=2)
    dev, held = split_questions(questions)
    dev = dev[: args.questions]

    print(f"corpus    : {corpus.summary()}")
    print(f"questions : {len(dev)} dev, {len(held)} held out\n")

    started = time.perf_counter()
    results = run_grid(
        corpus.documents,
        dev,
        strategies=args.strategies,
        sizes=args.sizes,
        modes=args.modes,
        embedders=args.embedders,
        progress=lambda m: print(f"  {m}", file=sys.stderr, flush=True),
    )
    print(f"\n{len(results)} configurations in {time.perf_counter() - started:.0f}s\n")
    print(format_table(results, limit=40))

    out = Path(args.out) if args.out else BACKEND / "data" / "grid_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([r.as_dict() for r in results], indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
