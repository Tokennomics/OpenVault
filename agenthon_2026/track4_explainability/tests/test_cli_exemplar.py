"""End-to-end test: the minimal baseline produces a schema-valid answer.json
for the public worked exemplar (t4-EXAMPLE-eps-beat)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BASELINES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BASELINES))

from baseline_agent.cli import run  # noqa: E402

_UNIT = _BASELINES.parent / "units" / "t4-EXAMPLE-eps-beat"


def test_baseline_writes_valid_answer(tmp_path: Path) -> None:
    out = tmp_path / "answer.json"
    answer = run(_UNIT / "task.json", _UNIT / "corpus", out)

    assert out.exists()
    assert answer["task_id"] == "t4-EXAMPLE-eps-beat"
    eps = answer["entity_predictions"]
    assert len(eps) == 1
    pred = eps[0]
    assert pred["entity_id"] == "AAPL"
    assert pred["label"] in {"beat", "miss", "inline"}
    assert pred["interval"]["level"] == 0.90
    assert pred["interval"]["lo"] <= pred["interval"]["hi"]
    assert len(pred["claims"]) >= 1
    for claim in pred["claims"]:
        assert claim["doc_id"]
        assert 0 <= claim["span_start"] <= claim["span_end"]


def test_baseline_respects_embargo(tmp_path: Path) -> None:
    """No cited doc may post-date the task cutoff."""
    out = tmp_path / "answer.json"
    answer = run(_UNIT / "task.json", _UNIT / "corpus", out)
    task = json.loads((_UNIT / "task.json").read_text())
    cutoff = task["cutoff_date"]
    corpus = _UNIT / "corpus"
    for pred in answer["entity_predictions"]:
        for claim in pred["claims"]:
            doc_path = corpus / f"{claim['doc_id']}.json"
            if doc_path.exists():
                doc = json.loads(doc_path.read_text())
                assert doc.get("doc_date", "0000-00-00") <= cutoff


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
