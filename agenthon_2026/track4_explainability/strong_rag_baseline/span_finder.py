"""Locate model-quoted evidence as exact character spans in the corpus.

Model-supplied offsets are never trusted: the model returns a *quote*, and this
module finds that quote as an exact substring of the cited document's joined
text, emitting real ``(span_start, span_end)`` offsets. If the quote cannot be
located exactly (after a length-preserving punctuation normalization), the
caller falls back to the retrieved chunk's known-good offsets or drops the
claim — a fabricated span must never reach the answer.

Normalization is strictly one-to-one per character (curly quotes/dashes to
ASCII), so offsets found in normalized text are valid in the original.
"""
from __future__ import annotations

_CHAR_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        " ": " ",
    }
)


def _normalize(text: str) -> str:
    return text.translate(_CHAR_MAP)


def find_span(doc_text: str, quote: str) -> tuple[int, int] | None:
    """Return (start, end) of ``quote`` in ``doc_text``, or None if not found."""
    quote = quote.strip()
    if not quote:
        return None
    start = doc_text.find(quote)
    if start < 0:
        start = _normalize(doc_text).find(_normalize(quote))
    if start < 0:
        return None
    return start, start + len(quote)
