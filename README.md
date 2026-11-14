# RAG Eval

[![CI](https://github.com/jayahareeswarreddy/rag-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/jayahareeswarreddy/rag-eval/actions/workflows/ci.yml)

A retrieval pipeline with the measurement apparatus attached.

Most RAG systems are built by copying a chunk size from a tutorial, picking a
retriever, and eyeballing a dozen answers. This one measures those decisions
instead: four chunking strategies and three retrievers, scored against
labelled questions, with every generated sentence verified against the
passage it cites. Upload your own documents and query them from the browser.

**Live demo → _(deploying)_**

---

## Why measure retrieval at all

An AI document assistant does two things: it searches your documents for
relevant passages, then writes an answer from them. When the answer is wrong,
the blame lands on the writing step. Usually the search step never found the
right passage, so the model had nothing to work from — and no prompt can fix a
model that was never shown the answer.

Almost nobody checks the search step, because checking it needs a labelled
test set and metrics. So the question *"how do you know your RAG works?"* is
usually answered with *"it seemed fine."* This project is the other answer.

## What was found

875 Python standard-library modules (14M characters), 930 questions derived
from docstrings, 27 configurations scored on a 150-question development split.
Full table in `backend/data/grid_results.json`; the **Results** tab in the UI
renders it sortable.

| chunking | size | retrieval | chunks | top-1 | MRR | nDCG@5 | ms p50 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| **ast** | 512 | bm25 | 54,459 | **0.960** | 0.973 | 0.977 | 38 |
| **ast** | 1024 | bm25 | 34,478 | **0.960** | 0.972 | 0.976 | 28 |
| ast | 2000 | bm25 | 26,384 | 0.953 | 0.970 | 0.976 | 26 |
| ast | 512 | hybrid | 54,459 | 0.947 | 0.966 | 0.971 | 170 |
| recursive | 512 | bm25 | 36,605 | 0.893 | 0.935 | 0.949 | 48 |
| fixed | 512 | bm25 | 31,692 | 0.860 | 0.916 | 0.932 | 22 |
| fixed | 2000 | dense | 8,333 | 0.673 | 0.789 | 0.829 | 18 |
| recursive | 2000 | dense | 8,635 | 0.587 | 0.745 | 0.798 | 19 |

Three findings, including one that reversed and one that contradicts the
standard advice.

**1. Structure-aware chunking won, but only once the corpus was realistic.**
Splitting Python at function and class boundaries (`ast`) takes the top three
rows: top-1 of 0.960 against 0.860 for fixed-size windows at the same chunk
size. On an earlier 93-document corpus it *lost* to fixed windows. The result
reversed when the corpus grew to 875 documents with many thematically adjacent
modules, which is the setting that actually resembles production. A benchmark
too small to discriminate produces confident wrong conclusions.

**2. Pure keyword search beat hybrid retrieval everywhere — on this data.**
Every top row is BM25 alone. Fusing in dense retrieval cost 0.013 of top-1 and
quadrupled latency. The reason is in how the questions were made: they come
from docstrings, which share vocabulary with the code they describe, so a
lexical retriever is already near-perfect and fusion only dilutes it. Real
user questions use different words from the documents, and that is where dense
retrieval earns its keep. The honest conclusion is not "hybrid is worse" but
"hybrid is worth nothing until your questions stop sharing vocabulary with
your corpus" — which an eval set of real questions would show directly.

**3. Larger chunks hurt dense retrieval badly.** Top-1 falls from 0.913 to
0.587 as chunks grow from 512 to 2000 characters. A 2000-character chunk's
vector averages several unrelated functions into one point, and nothing is
near it. Keyword search is far less sensitive to this because a term either
appears or it does not.

### A measurement bug worth recording

The first run of this grid reported **89,910 ms** per query for one
configuration — six thousand times its neighbours. Re-measured, it was 15 ms.
The grid had run overnight, the machine suspended, and `perf_counter()` counts
suspend time, so one suspended sample destroyed the mean. The harness now
reports p50/p95/max and never a mean. Latency should always be a percentile.

## How it works

```
documents ──► chunk ──► embed ──► index ──┐
                                           ├──► retrieve ──► generate ──► verify
question ─────────────────────────────────┘        │              │          │
                                               dense/bm25     citations   per-sentence
                                               /hybrid        enforced    grounding
```

**Chunking** (`backend/app/chunking.py`) — fixed windows, recursive splitting
on natural boundaries, Markdown with the heading path prefixed into each
chunk, and Python AST splitting at function and class boundaries with the
module path, enclosing class and docstring attached. The AST header is what
makes an isolated method findable: without it the embedding has no idea which
module it came from.

**Retrieval** (`backend/app/retrieval.py`) — dense cosine, Okapi BM25, and a
hybrid that fuses both with reciprocal rank fusion. RRF is used rather than
normalising and adding scores because the two live on different scales
(roughly 0.25 against 12 on the same hit); normalising makes the fusion
hostage to whichever retriever produced an outlier.

**Embeddings** (`backend/app/embedding.py`) — a TF-IDF baseline in pure NumPy
and real sentence embeddings via `fastembed` (ONNX, no PyTorch). The baseline
is the point: an accuracy figure means nothing without a control beside it.
The tokenizer splits identifiers (`getUserSession` → `get`, `user`, `session`)
and drops stopwords explicitly — on four uploaded documents there is not
enough IDF signal to suppress "on", and *"can I deploy on a friday"* was
matching the **On**-call section.

**Generation** (`backend/app/generation.py`) — every sentence must carry a
citation, and each citation is checked mechanically by token overlap against
the chunk it points at. Not by asking the model whether it was truthful: a
model that invented a fact will confirm that it did not. With no API key the
generator is extractive — it composes only from retrieved spans and filters
them for relevance to the question, so it is never ungrounded and never pads
an answer with on-topic-looking noise.

**Evaluation** (`backend/app/evaluation.py`) — recall@k, precision@k, MRR,
nDCG, run across the full cartesian product of configurations. Relevance is
judged at document level because chunk ids change whenever chunking changes,
and the whole point is to compare chunking.

**Labelled questions** (`backend/app/corpus.py`) — derived from docstrings,
the CodeSearchNet construction. Any question whose text contains its own
module name is discarded: without that filter a lexical retriever wins by
copying the name back and the benchmark flatters BM25.

**Upload** (`backend/app/sessions.py`) — text and source files only, held in
memory, capped at 20 files / 1 MB each / 3 M characters, expiring after an
hour. An unknown session is a 404, not a quiet fall-back to the demo corpus.
Search reports a **coverage** figure — the share of the question's terms found
in a passage — and flags a weak match when nothing clears a third, because
retrieval returns k results whether or not any of them are relevant.

## Running it

```bash
cd backend
pip install -r requirements.txt
uvicorn app.api:app --reload --port 8020      # http://localhost:8020
```

No API key, no database, no network needed. Optional:

| Variable | Effect |
| --- | --- |
| `GROQ_API_KEY` / `OPENAI_API_KEY` | LLM answers instead of extractive |
| `RAG_CORPUS_DIR=../some-repo` | index a local directory instead of the stdlib |
| `RAG_STDLIB_FULL=1` | the full ~875-module corpus rather than ~93 |
| `RAG_EMBEDDER=fastembed` | real embeddings — minutes to index, not seconds |

The default embedder is TF-IDF deliberately. Embedding 7,437 chunks through
ONNX on CPU takes over seven minutes; as a server default that reads as a hung
deploy. Real embeddings are for the experiment script, not the web process.

```bash
python scripts/run_experiments.py --full          # ~5 min, writes grid_results.json
pytest backend/tests -q                           # 47 tests
```

## Stack

Python 3.11+ · FastAPI · NumPy · fastembed (optional) · vanilla JS · GitHub Actions

No framework on the frontend: the page is three views over a handful of
endpoints, and a build step would cost more than it saves. The UI and API are
one service, so there is one thing to deploy and no CORS in production.

## Limits, stated plainly

- The question set is docstring-derived, not real user questions. Scores are
  not comparable to a human-written set, and finding 2 above would likely
  change with one.
- The in-process index is for comparing configurations, not serving traffic.
  Adding a vector database would not change any measured result — same
  vectors, same cosine — but it is what you would do next for scale.
- Grounding by token overlap is a weak check. It cannot catch a sentence that
  reuses the source's words to say something the source does not.
