"""
Answer generation with enforced citations.

A RAG system that returns fluent prose with no way to check it is worse than
no RAG system, because it launders a guess into something that looks sourced.
The rule enforced here is simple: every sentence in an answer must carry a
citation to a retrieved chunk, and any sentence that does not is flagged
before the user sees it.

Grounding is then verified *mechanically* rather than by asking the model
whether it was truthful. A model that hallucinated a fact will happily assert
that it did not, so self-assessment is not evidence. Token overlap against the
cited chunk is weak, but it is a real measurement and it cannot be talked
around.

Works with no API key: `ExtractiveGenerator` composes an answer purely from
retrieved spans. It is blunt, but it is never ungrounded, and it keeps the
demo honest when there is no model behind it.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .embedding import tokenize
from .retrieval import Hit

CITATION = re.compile(r"\[(\d+)\]")
SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Claim:
    """One sentence of an answer, with its grounding verdict."""

    text: str
    citations: list[int]
    grounded: bool
    overlap: float
    reason: str = ""


@dataclass
class Answer:
    text: str
    claims: list[Claim]
    sources: list[dict]
    model: str
    # True only when every claim cites a source and clears the overlap floor.
    fully_grounded: bool = False
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "model": self.model,
            "fully_grounded": self.fully_grounded,
            "claims": [vars(c) for c in self.claims],
            "sources": self.sources,
            "warnings": self.warnings,
        }


def _overlap(sentence: str, source: str) -> float:
    """
    Fraction of a sentence's content tokens that appear in its source.

    Stopword-ish short tokens are dropped so that "the", "a" and "of" cannot
    carry a sentence over the threshold on their own.
    """
    sentence_tokens = {t for t in tokenize(sentence) if len(t) > 3}
    if not sentence_tokens:
        return 1.0  # nothing substantive asserted
    source_tokens = set(tokenize(source))
    return len(sentence_tokens & source_tokens) / len(sentence_tokens)


def verify(text: str, hits: list[Hit], floor: float = 0.45) -> tuple[list[Claim], bool]:
    """
    Split an answer into sentences and check each one against what it cites.

    `floor` is a judgement call, not a derived constant: too high and correct
    paraphrase gets flagged, too low and the check waves through invention.
    0.45 was chosen on the development set and is exposed so it can be tuned
    per corpus rather than being buried.
    """
    claims: list[Claim] = []
    for sentence in (s.strip() for s in SENTENCE.split(text) if s.strip()):
        cited = [int(n) for n in CITATION.findall(sentence)]
        valid = [n for n in cited if 1 <= n <= len(hits)]

        if not valid:
            claims.append(
                Claim(sentence, [], False, 0.0, "no citation")
            )
            continue

        bare = CITATION.sub("", sentence)
        best = max(_overlap(bare, hits[n - 1].chunk.text) for n in valid)
        grounded = best >= floor
        claims.append(
            Claim(
                sentence,
                valid,
                grounded,
                round(best, 3),
                "" if grounded else f"overlap {best:.2f} below {floor}",
            )
        )

    return claims, bool(claims) and all(c.grounded for c in claims)


class Generator(ABC):
    name: str

    @abstractmethod
    def generate(self, question: str, hits: list[Hit]) -> str:
        """Return answer text with [n] citations into `hits`."""


class ExtractiveGenerator(Generator):
    """
    Builds an answer by selecting sentences from retrieved chunks.

    No model, no key, no network. It cannot hallucinate because it only
    concatenates text that was retrieved, which makes it the control that
    proves the grounding check is measuring something real: it should always
    pass verification.
    """

    name = "extractive"

    def generate(self, question: str, hits: list[Hit]) -> str:
        if not hits:
            return "No relevant passage was retrieved for this question."

        query_tokens = {t for t in tokenize(question) if len(t) > 3}
        picked: list[str] = []

        for i, hit in enumerate(hits[:3], start=1):
            sentences = [
                s.strip()
                for s in SENTENCE.split(hit.chunk.text.replace("\n", " "))
                if len(s.strip()) > 30
            ]
            if not sentences:
                continue
            best = max(
                sentences,
                key=lambda s: len(query_tokens & set(tokenize(s))),
            )
            picked.append(f"{best.rstrip('.')} [{i}].")

        return " ".join(picked) or "No relevant passage was retrieved."


class LLMGenerator(Generator):
    """
    Groq or OpenAI, instructed to cite and refuse when the context is thin.

    The prompt forbids outside knowledge. That instruction is not trusted —
    `verify` checks the output regardless — but it measurably reduces how
    often the check has to fire.
    """

    name = "llm"

    PROMPT = (
        "Answer the question using ONLY the numbered sources below.\n"
        "Rules:\n"
        "1. Every sentence must end with a citation like [1] or [2].\n"
        "2. Use only facts stated in the sources. Do not add outside knowledge.\n"
        "3. If the sources do not answer the question, reply exactly: "
        "'The retrieved sources do not answer this question.'\n"
        "4. Be concise: at most four sentences.\n"
    )

    def __init__(self, provider: str = "auto", model: str | None = None) -> None:
        self.provider, self.key, self.base_url, self.model = _resolve_provider(
            provider, model
        )
        self.name = f"llm:{self.provider}:{self.model}"

    def generate(self, question: str, hits: list[Hit]) -> str:
        from openai import OpenAI

        sources = "\n\n".join(
            f"[{i}] ({h.chunk.doc_id}) {h.chunk.text[:1200]}"
            for i, h in enumerate(hits, start=1)
        )
        kwargs = {"api_key": self.key, "timeout": 30}
        if self.base_url:
            kwargs["base_url"] = self.base_url

        response = OpenAI(**kwargs).chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.PROMPT},
                {
                    "role": "user",
                    "content": f"Sources:\n{sources}\n\nQuestion: {question}",
                },
            ],
            max_tokens=400,
            temperature=0.1,
        )
        return (response.choices[0].message.content or "").strip()


def _resolve_provider(provider: str, model: str | None):
    groq = os.getenv("GROQ_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if provider in ("groq", "auto") and groq:
        return "groq", groq, "https://api.groq.com/openai/v1", model or _groq_model(groq)
    if provider in ("openai", "auto") and openai_key:
        return "openai", openai_key, None, model or "gpt-4o-mini"
    raise RuntimeError("no API key: set GROQ_API_KEY or OPENAI_API_KEY")


def _groq_model(key: str) -> str:
    """
    Pick a model Groq currently serves.

    Pinning a name is fragile — providers retire them, and a retired id turns
    every call into a silent failure.
    """
    preferred = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    ]
    try:
        from openai import OpenAI

        available = {
            m.id
            for m in OpenAI(
                api_key=key, base_url="https://api.groq.com/openai/v1", timeout=10
            )
            .models.list()
            .data
        }
        for candidate in preferred:
            if candidate in available:
                return candidate
    except Exception as exc:
        print(f"[generation] could not list Groq models ({exc})")
    return preferred[0]


def get_generator(name: str = "auto") -> Generator:
    """Prefer an LLM, fall back to extractive so the demo always answers."""
    if name == "extractive":
        return ExtractiveGenerator()
    try:
        return LLMGenerator(provider="auto" if name == "auto" else name)
    except Exception as exc:
        if name not in ("auto", "llm"):
            raise
        print(f"[generation] LLM unavailable ({exc}); using extractive")
        return ExtractiveGenerator()


def answer_question(
    question: str, hits: list[Hit], generator: Generator | None = None, floor: float = 0.45
) -> Answer:
    """Generate an answer and verify every claim against its cited source."""
    generator = generator or get_generator()
    text = generator.generate(question, hits)
    claims, fully_grounded = verify(text, hits, floor=floor)

    warnings: list[str] = []
    ungrounded = [c for c in claims if not c.grounded]
    if ungrounded:
        warnings.append(
            f"{len(ungrounded)} of {len(claims)} sentences are not supported "
            "by their cited source"
        )
    if not hits:
        warnings.append("nothing was retrieved for this question")

    return Answer(
        text=text,
        claims=claims,
        sources=[
            {
                "n": i,
                "doc_id": h.chunk.doc_id,
                "chunk_id": h.chunk.chunk_id,
                "score": round(h.score, 4),
                "preview": h.chunk.text[:300],
                "meta": h.chunk.meta,
            }
            for i, h in enumerate(hits, start=1)
        ],
        model=generator.name,
        fully_grounded=fully_grounded,
        warnings=warnings,
    )
