"""NeMo Guardrails action wrapping the pure-stdlib citation rail.

Registered by ``nemoguardrails`` at config load. The action expects the draft
answer as a JSON string (the bot message) and reads the unit location from the
``T4_UNIT_DIR`` environment variable so the rail knows which frozen corpus and
cutoff_date to check against.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

try:
    from nemoguardrails.actions import action
except ImportError:  # pragma: no cover - illustrative wiring only
    def action(*_a, **_k):  # type: ignore[misc]
        def deco(fn):
            return fn

        return deco

from ..citation_rail import check_answer, load_corpus


@action(name="check_citations")
async def check_citations(answer: str) -> list[str]:
    """Return one message per rail finding (empty list = draft is clean)."""
    unit_dir = Path(os.environ.get("T4_UNIT_DIR", "units/t4-EXAMPLE-eps-beat"))
    task = json.loads((unit_dir / "task.json").read_text(encoding="utf-8"))
    corpus = load_corpus(unit_dir / "corpus")
    draft = json.loads(answer)
    findings = check_answer(draft, corpus, task["cutoff_date"])
    return [str(f) for f in findings]
