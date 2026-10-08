"""Embargo-aware lexical retriever for the minimal Track 4 baseline.

Scores each indexed document against a query by token overlap (a stand-in for the
BM25 + dense hybrid in the full baseline spec) and returns the best pre-cutoff
span. Documents whose ``doc_date`` exceeds the task ``cutoff_date`` are dropped
BEFORE scoring, so the agent never cites stale evidence (Family 5 trap).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .indexer import IndexedDoc

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 2]


@dataclass
class Retrieved:
    doc_id: str
    span_start: int
    span_end: int
    text: str
    score: float


def retrieve(
    query: str,
    docs: list[IndexedDoc],
    cutoff_date: str,
    max_span_chars: int = 240,
) -> Retrieved | None:
    """Return the best embargo-compliant span for ``query``, or None if none."""
    q = set(_tokens(query))
    if not q:
        return None
    best: Retrieved | None = None
    for doc in docs:
        if doc.doc_date is not None and doc.doc_date > cutoff_date:
            continue  # embargo: skip post-cutoff documents
        d_tokens = _tokens(doc.text)
        if not d_tokens:
            continue
        overlap = sum(1 for t in d_tokens if t in q)
        score = overlap / (len(d_tokens) ** 0.5)
        if best is None or score > best.score:
            span_end = min(len(doc.text), max_span_chars)
            best = Retrieved(
                doc_id=doc.doc_id,
                span_start=0,
                span_end=span_end,
                text=doc.text[:span_end],
                score=score,
            )
    return best
