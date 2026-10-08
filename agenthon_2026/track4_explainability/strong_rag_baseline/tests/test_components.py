"""Component tests: indexer offsets, BM25 embargo/determinism, span finding."""
from __future__ import annotations

from pathlib import Path

from baselines.strong_rag_baseline.indexer import Chunk, build_index
from baselines.strong_rag_baseline.retriever import BM25Index
from baselines.strong_rag_baseline.span_finder import find_span

EXAMPLE_UNIT = Path(__file__).resolve().parents[3] / "units" / "t4-EXAMPLE-eps-beat"


def test_chunk_offsets_resolve_in_joined_text():
    corpus = build_index(EXAMPLE_UNIT / "corpus")
    assert corpus.chunks, "example corpus produced no chunks"
    for chunk in corpus.chunks:
        doc_text = corpus.doc_texts[chunk.doc_id]
        assert doc_text[chunk.span_start : chunk.span_end] == chunk.text


def make_chunk(doc_id: str, text: str, date: str | None = "2024-01-01") -> Chunk:
    return Chunk(doc_id=doc_id, doc_date=date, span_start=0, span_end=len(text), text=text)


def test_bm25_drops_post_cutoff_and_undated_docs():
    chunks = [
        make_chunk("ok", "services revenue grew strongly"),
        make_chunk("late", "services revenue grew strongly", date="2024-06-01"),
        make_chunk("undated", "services revenue grew strongly", date=None),
    ]
    index = BM25Index(chunks, cutoff_date="2024-03-15")
    hits = index.search("services revenue", top_k=10)
    assert [h.chunk.doc_id for h in hits] == ["ok"]


def test_bm25_ranking_is_deterministic_and_relevant():
    chunks = [
        make_chunk("a", "iphone demand stable in international markets"),
        make_chunk("b", "services revenue record services margin services"),
        make_chunk("c", "unrelated filing boilerplate text"),
    ]
    index = BM25Index(chunks, cutoff_date="2024-03-15")
    first = index.search("services revenue", top_k=3)
    second = index.search("services revenue", top_k=3)
    assert [h.chunk.doc_id for h in first] == [h.chunk.doc_id for h in second]
    assert first[0].chunk.doc_id == "b"


def test_find_span_exact_and_normalized():
    doc = "Management said the company’s outlook is “stable” for 2024."
    exact = find_span(doc, "outlook is")
    assert exact is not None and doc[exact[0] : exact[1]] == "outlook is"

    normalized = find_span(doc, "the company's outlook")
    assert normalized is not None
    start, end = normalized
    assert doc[start:end] == "the company’s outlook"  # offsets are original text


def test_find_span_missing_returns_none():
    assert find_span("short text", "not present anywhere") is None
    assert find_span("short text", "   ") is None
