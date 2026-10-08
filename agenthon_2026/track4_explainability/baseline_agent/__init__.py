"""Track 4 minimal runnable RAG baseline.

A dependency-light reference agent for Explainability. It
indexes the frozen corpus, retrieves the most relevant spans per entity by lexical
overlap (BM25-style scoring with no external deps), applies a simple rule-based
classifier, emits a 90% prediction interval, and writes a schema-valid
``answer.json`` with grounded citations.

This is intentionally simple: it is the floor a real agent should beat, and it
exists so the public quick-start commands actually run. It is NOT the strong RAG
baseline described in ``baselines/README.md`` (BM25+dense retrieval over an
open-weights LLM with a learned calibration head). That one IS shipped, as a
runnable scaffold in ``baselines/strong_rag_baseline/``; only its learned
calibration head and dense index are unreleased. See the release-status table
in ``baselines/README.md``, which is authoritative. The entry point is ``baseline_agent.cli:main``
(installed/invoked as ``python -m baseline_agent.cli`` or via ``baseline_agent.py``).
"""

from .cli import main

__all__ = ["main"]
