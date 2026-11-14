"""
Tests for the retrieval pipeline and the metrics.

The metric tests matter more than usual here. If recall@k is subtly wrong,
every experiment in the README is wrong too, and nothing downstream would
reveal it — a broken metric still produces plausible numbers. So the metrics
are checked against hand-computed values rather than against themselves.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chunking import (  # noqa: E402
    chunk,
    chunk_auto,
    chunk_fixed,
    chunk_markdown,
    chunk_python_ast,
    chunk_recursive,
    describe,
)
from app.embedding import TfidfEmbedder, get_embedder, tokenize  # noqa: E402
from app.evaluation import (  # noqa: E402
    Question,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.generation import ExtractiveGenerator, answer_question, verify  # noqa: E402
from app.retrieval import BM25, Retriever, reciprocal_rank_fusion  # noqa: E402

SAMPLE_MD = """# Queue

## Leases
A worker leases a task for thirty seconds and heartbeats while it runs.
If the worker dies the lease expires and another worker claims the task.

## Retries
Failed tasks retry with exponential backoff, then move to the dead letter queue.
"""

SAMPLE_PY = '''
"""Module docstring."""

def build_index(documents):
    """Build an inverted index from a list of documents."""
    index = {}
    for doc in documents:
        index[doc] = len(doc)
    return index


class Searcher:
    """Queries an inverted index."""

    def query(self, term):
        """Return the documents containing a term."""
        return self.index.get(term, [])
'''


# --------------------------------------------------------------- chunking

def test_fixed_chunks_respect_size_and_overlap():
    text = "x" * 1000
    chunks = chunk_fixed(text, "d", size=200, overlap=50)
    assert all(len(c) <= 200 for c in chunks)
    # Overlap means consecutive windows share their boundary region.
    assert chunks[0].text[-50:] == chunks[1].text[:50]


def test_fixed_rejects_overlap_larger_than_size():
    with pytest.raises(ValueError):
        chunk_fixed("abc", "d", size=10, overlap=10)


def test_recursive_prefers_paragraph_boundaries():
    text = "para one is here.\n\npara two is here.\n\npara three is here."
    chunks = chunk_recursive(text, "d", size=40)
    # No chunk should begin mid-word.
    assert all(not c.text.startswith(" ") for c in chunks)
    assert len(chunks) >= 2


def test_markdown_prefixes_the_heading_path():
    chunks = chunk_markdown(SAMPLE_MD, "queue.md", size=400)
    leases = [c for c in chunks if "lease" in c.text.lower()][0]
    # The parent heading must be inside the text, not only the metadata,
    # because only the text gets embedded.
    assert leases.text.startswith("Queue > Leases:")
    assert leases.meta["heading_path"] == "Queue > Leases"


def test_ast_chunking_keeps_functions_whole():
    chunks = chunk_python_ast(SAMPLE_PY, "search.py")
    symbols = {c.meta.get("symbol") for c in chunks}
    assert "build_index" in symbols
    assert "Searcher" in symbols
    assert "Searcher.query" in symbols

    body = next(c for c in chunks if c.meta.get("symbol") == "build_index")
    # The whole function survived in one piece.
    assert "def build_index" in body.text
    assert "return index" in body.text
    # And it carries enough context to be found on its own.
    assert "search.py" in body.text


def test_ast_falls_back_on_a_syntax_error():
    chunks = chunk_python_ast("def broken(:\n  pass", "bad.py")
    assert chunks  # degraded, not empty


def test_chunk_auto_dispatches_on_extension():
    assert chunk_auto(SAMPLE_PY, "a.py")[0].meta["strategy"] == "ast"
    assert chunk_auto(SAMPLE_MD, "a.md")[0].meta["strategy"] == "markdown"
    assert chunk_auto("plain text here", "a.txt")[0].meta["strategy"] == "recursive"


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        chunk("text", "d", "nonsense")


def test_describe_handles_an_empty_corpus():
    assert describe([])["count"] == 0


# -------------------------------------------------------------- embedding

def test_tokenize_splits_identifiers():
    tokens = tokenize("getUserSession and get_user_session")
    assert "user" in tokens and "session" in tokens
    assert "get_user_session" in tokens  # the whole identifier is kept too


def test_tfidf_vectors_are_unit_length():
    embedder = TfidfEmbedder().fit(["alpha beta", "beta gamma", "gamma delta"])
    vectors = embedder.encode(["alpha beta"])
    assert math.isclose(float((vectors[0] ** 2).sum()), 1.0, rel_tol=1e-5)


def test_tfidf_requires_fit_first():
    with pytest.raises(RuntimeError):
        TfidfEmbedder().encode(["anything"])


def test_unseen_text_does_not_produce_nan():
    embedder = TfidfEmbedder().fit(["alpha beta"])
    vector = embedder.encode(["completely unrelated vocabulary"])[0]
    assert not bool((vector != vector).any())  # no NaN


# -------------------------------------------------------------- retrieval

def test_bm25_ranks_the_matching_document_first():
    bm25 = BM25().fit(
        ["the cat sat on the mat", "database indexing strategies", "cat grooming tips"]
    )
    scores = bm25.scores("database indexing")
    assert scores.argmax() == 1


def test_rrf_rewards_agreement_between_rankings():
    # Document 5 is ranked first by both retrievers; 9 is first by one only.
    fused = reciprocal_rank_fusion([[5, 9, 1], [5, 1, 9]])
    assert max(fused, key=fused.get) == 5


@pytest.fixture
def retriever():
    chunks = chunk_markdown(SAMPLE_MD, "queue.md", size=400) + chunk_python_ast(
        SAMPLE_PY, "search.py"
    )
    return Retriever(get_embedder("tfidf")).index(chunks)


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_search_modes_find_the_right_document(retriever, mode):
    hits = retriever.search("what happens when a worker dies", k=3, mode=mode)
    assert hits
    assert hits[0].chunk.doc_id == "queue.md"


def test_search_rejects_an_unknown_mode(retriever):
    with pytest.raises(ValueError):
        retriever.search("anything", mode="telepathy")


def test_empty_index_returns_no_hits():
    assert Retriever(get_embedder("tfidf")).index([]).search("q") == []


def test_hybrid_reports_both_component_scores(retriever):
    hit = retriever.search("inverted index", k=1, mode="hybrid")[0]
    assert {"rrf", "dense", "bm25"} <= set(hit.detail)


# ---------------------------------------------------------------- metrics

def test_recall_at_k_is_computed_over_relevant_not_retrieved():
    # One of two relevant documents found in the top 2.
    assert recall_at_k(["a", "b", "c"], ["a", "z"], 2) == 0.5
    assert recall_at_k(["a", "b"], ["a", "b"], 2) == 1.0
    assert recall_at_k(["x"], ["a"], 5) == 0.0


def test_recall_with_no_relevant_documents_is_zero_not_an_error():
    assert recall_at_k(["a"], [], 5) == 0.0


def test_precision_at_k_is_over_retrieved():
    assert precision_at_k(["a", "x", "y", "z"], ["a"], 4) == 0.25
    assert precision_at_k([], ["a"], 5) == 0.0


def test_reciprocal_rank_uses_the_first_hit():
    assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
    assert reciprocal_rank(["a", "x"], ["a"]) == 1.0
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_ndcg_prefers_a_hit_at_rank_one():
    early = ndcg_at_k(["a", "x", "y"], ["a"], 3)
    late = ndcg_at_k(["x", "y", "a"], ["a"], 3)
    assert early == 1.0
    assert early > late > 0


def test_ndcg_matches_a_hand_computed_value():
    # Relevant at ranks 2 and 3: DCG = 1/log2(3) + 1/log2(4)
    # Ideal (both at ranks 1 and 2) = 1/log2(2) + 1/log2(3)
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert math.isclose(ndcg_at_k(["x", "a", "b"], ["a", "b"], 3), dcg / ideal, rel_tol=1e-9)


# ------------------------------------------------------------- generation

def test_verify_flags_a_sentence_with_no_citation(retriever):
    hits = retriever.search("leases", k=2)
    claims, grounded = verify("Workers lease tasks for thirty seconds.", hits)
    assert not grounded
    assert claims[0].reason == "no citation"


def test_verify_flags_a_citation_that_does_not_support_the_claim(retriever):
    hits = retriever.search("leases", k=2)
    claims, grounded = verify(
        "The system is written in Haskell and deploys to mainframes [1].", hits
    )
    assert not grounded
    assert "below" in claims[0].reason


def test_verify_accepts_a_grounded_sentence(retriever):
    hits = retriever.search("what happens when a worker dies", k=2)
    source = hits[0].chunk.text
    # Take words from the first sentence only. Slicing blindly can cross a
    # sentence boundary, and verify would then correctly flag the uncited
    # first half -- which tests the splitter, not the grounding check.
    first_sentence = source.replace("\n", " ").split(". ")[0]
    sentence = " ".join(first_sentence.split()[:14])
    claims, grounded = verify(f"{sentence} [1].", hits)
    assert grounded, claims


def test_verify_requires_every_sentence_to_cite(retriever):
    """One uncited sentence invalidates the answer, even beside a good one."""
    hits = retriever.search("what happens when a worker dies", k=2)
    source = hits[0].chunk.text.replace("\n", " ").split(". ")[0]
    cited = " ".join(source.split()[:14])
    claims, grounded = verify(f"{cited} [1]. An uncited assertion follows.", hits)
    assert not grounded
    assert [c.grounded for c in claims] == [True, False]


def test_extractive_answers_are_always_grounded(retriever):
    """The control: an answer built only from retrieved text must verify."""
    for query in ["what happens when a worker dies", "how is an index built"]:
        hits = retriever.search(query, k=3)
        answer = answer_question(query, hits, generator=ExtractiveGenerator())
        assert answer.claims
        assert answer.fully_grounded, answer.as_dict()


def test_answer_with_no_hits_warns_rather_than_crashing():
    answer = answer_question("anything", [], generator=ExtractiveGenerator())
    assert "nothing was retrieved" in " ".join(answer.warnings)


# ------------------------------------------------------- evaluation wiring

def test_evaluate_retrieval_reports_failures(retriever):
    from app.evaluation import evaluate_retrieval

    questions = [
        Question("ok", "what happens when a worker dies", ["queue.md"]),
        Question("bad", "quantum chromodynamics", ["nonexistent.md"]),
    ]
    result = evaluate_retrieval(
        retriever, questions, config={}, chunk_stats={}, index_seconds=0.0
    )
    assert result.questions == 2
    assert "bad" in result.failures
    assert "ok" not in result.failures


# ------------------------------------------------------ stopword handling

def test_stopwords_are_dropped_from_tokens():
    assert tokenize("can I deploy on a friday") == ["deploy", "friday"]


def test_identifier_parts_survive_stopword_filtering():
    """Splitting an identifier must not be undone by the stopword list."""
    tokens = tokenize("get_user_session")
    assert "get_user_session" in tokens
    assert {"user", "session"} <= set(tokens)


def test_stopwords_do_not_dominate_a_small_corpus():
    """
    The bug this guards: on a handful of documents there is not enough IDF
    signal to suppress function words, so "can I deploy on a friday" matched
    the "On-call" section through the word "on" alone.
    """
    from app.chunking import chunk_markdown

    handbook = (
        "# Handbook\n\n"
        "## On-call rotation\n"
        "On-call runs Monday to Monday. The primary responder acknowledges a page "
        "within five minutes, and on-call escalates on no acknowledgement.\n\n"
        "## Deployment windows\n"
        "Deploys are allowed Monday through Thursday. Friday deploys require "
        "written approval from a staff engineer.\n"
    )
    chunks = chunk_markdown(handbook, "handbook.md", size=400)
    retriever = Retriever(get_embedder("tfidf")).index(chunks)

    for mode in ("dense", "bm25", "hybrid"):
        top = retriever.search("can I deploy on a friday", k=1, mode=mode)[0]
        assert "Deployment" in top.chunk.text, f"{mode} matched the wrong section"


# ------------------------------------------------------- upload sessions

@pytest.fixture
def store():
    from app.sessions import SessionStore

    return SessionStore()


def test_upload_indexes_and_is_queryable(store):
    session, skipped = store.add(
        [
            ("handbook.md", b"# Handbook\n\n## Deploys\nFriday deploys need approval from a staff engineer.\n"),
            ("notes.txt", b"Receipts must be submitted within thirty days of the expense date.\n"),
        ]
    )
    assert not skipped
    assert len(session.documents) == 2
    hits = session.retriever.search("can I deploy on friday", k=1)
    assert "Friday" in hits[0].chunk.text


@pytest.mark.parametrize(
    "name,data,reason",
    [
        ("thing.exe", b"MZ\x90\x00binary", "not supported"),
        ("notes.md", b"\xff\xfe\x00bad utf8", "UTF-8"),
        ("tiny.txt", b"short", "too short"),
    ],
)
def test_unusable_uploads_are_reported_not_silently_dropped(store, name, data, reason):
    session, skipped = store.add([(name, data)])
    assert len(skipped) == 1
    assert reason.lower() in skipped[0]["reason"].lower()


def test_a_text_extension_hiding_binary_is_rejected(store):
    """A NUL byte means binary content wearing a text extension."""
    _, skipped = store.add([("payload.txt", b"text\x00\x01\x02more text here padding")])
    assert skipped and "binary" in skipped[0]["reason"].lower()


def test_filenames_are_reduced_to_plain_labels():
    from app.sessions import safe_name

    # The name is a label and a dict key, never a path.
    assert "/" not in safe_name("../../etc/passwd")
    assert chr(92) not in safe_name(".." + chr(92) + "windows" + chr(92) + "x.txt")
    assert safe_name("") == "untitled"
    assert len(safe_name("x" * 400)) <= 120


def test_session_character_cap_is_enforced(store):
    from app.sessions import MAX_SESSION_CHARS

    big = ("word " * 40_000).encode()          # ~200KB each
    files = [(f"doc{i}.txt", big) for i in range(30)]
    session, skipped = store.add(files)
    assert session.total_chars <= MAX_SESSION_CHARS
    assert skipped


def test_store_evicts_once_full(store):
    from app.sessions import MAX_SESSIONS

    for i in range(MAX_SESSIONS + 6):
        store.add([(f"d{i}.txt", b"a document with enough characters to index properly")])
    assert store.stats()["active_sessions"] <= MAX_SESSIONS


def test_dropping_a_session_makes_it_unreachable(store):
    session, _ = store.add([("d.txt", b"a document with enough characters to be indexed")])
    assert store.get(session.id) is not None
    assert store.drop(session.id)
    assert store.get(session.id) is None


# -------------------------------------------------- weak-match detection

def test_coverage_separates_a_real_match_from_noise():
    """
    Retrieval returns k results whether or not any are relevant, so a query
    the corpus never mentions still yields confident-looking passages.
    Coverage is what distinguishes them.
    """
    from app.embedding import tokenize

    passage = (
        "Every mutating endpoint accepts an idempotency key header. "
        "Retrying with the same key must not create a second resource."
    )

    def coverage(question: str) -> float:
        q = {t for t in tokenize(question) if len(t) > 2}
        return len(q & set(tokenize(passage))) / len(q)

    assert coverage("idempotency key header") == 1.0
    assert coverage("what is the parental leave policy") < 0.34
