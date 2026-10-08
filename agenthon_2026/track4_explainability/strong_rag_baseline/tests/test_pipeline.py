"""End-to-end pipeline test on the example unit with a mock model client."""
from __future__ import annotations

import json
from pathlib import Path

from baselines.strong_rag_baseline.cli import run
from baselines.strong_rag_baseline.client import MockModelClient
from baselines.strong_rag_baseline.indexer import build_index

EXAMPLE_UNIT = Path(__file__).resolve().parents[3] / "units" / "t4-EXAMPLE-eps-beat"
DOC_ID = "EDGAR_0000320193_8K_20240201"


def make_mock_reply() -> str:
    corpus = build_index(EXAMPLE_UNIT / "corpus")
    doc_text = corpus.doc_texts[DOC_ID]
    quote = doc_text[10:150]
    return json.dumps(
        {
            "label": "beat",
            "point_forecast": 1.58,
            "interval": {"level": 0.90, "lo": 1.45, "hi": 1.72},
            "evidence": [
                {  # groundable: verbatim corpus substring
                    "doc_id": DOC_ID,
                    "quote": quote,
                    "claim": "Apple announced quarterly results in its press release.",
                },
                {  # ungroundable: fabricated quote, must be dropped or chunk-mapped
                    "doc_id": DOC_ID,
                    "quote": "zzz this text appears nowhere zzz",
                    "claim": "A fabricated statement.",
                },
                {  # unknown doc: must be dropped
                    "doc_id": "NOT_A_DOC",
                    "quote": quote,
                    "claim": "Cited from outside the corpus.",
                },
            ],
        }
    )


def run_once(tmp_path: Path, reply: str) -> dict:
    return run(
        task_path=EXAMPLE_UNIT / "task.json",
        corpus_dir=EXAMPLE_UNIT / "corpus",
        out_path=tmp_path / "answer.json",
        client=MockModelClient(reply=reply),
        top_k=5,
    )


def test_pipeline_produces_grounded_schema_valid_answer(tmp_path):
    answer = run_once(tmp_path, make_mock_reply())

    assert answer["task_id"] == "t4-EXAMPLE-eps-beat"
    assert isinstance(answer["notes"], dict)
    [entity] = answer["entity_predictions"]
    assert entity["entity_id"] == "AAPL"
    assert entity["label"] == "beat"
    assert entity["interval"]["lo"] <= entity["interval"]["hi"]

    corpus = build_index(EXAMPLE_UNIT / "corpus")
    assert entity["claims"], "expected at least one grounded claim"
    for claim in entity["claims"]:
        doc_text = corpus.doc_texts[claim["doc_id"]]
        span_text = doc_text[claim["span_start"] : claim["span_end"]]
        assert span_text, "span must resolve to non-empty corpus text"
    # The unknown-doc evidence item can never survive.
    assert all(c["doc_id"] != "NOT_A_DOC" for c in entity["claims"])


def test_pipeline_is_deterministic(tmp_path):
    reply = make_mock_reply()
    first = run_once(tmp_path / "a", reply)
    second = run_once(tmp_path / "b", reply)
    assert first == second


def test_off_vocabulary_label_falls_back_deterministically(tmp_path):
    reply = json.loads(make_mock_reply())
    reply["label"] = "moon"
    answer = run_once(tmp_path, json.dumps(reply))
    [entity] = answer["entity_predictions"]
    assert entity["label"] == "beat"  # first allowed label


def test_missing_interval_gets_fallback_band(tmp_path):
    reply = json.loads(make_mock_reply())
    del reply["interval"]
    answer = run_once(tmp_path, json.dumps(reply))
    [entity] = answer["entity_predictions"]
    assert entity["interval"]["lo"] < entity["interval"]["hi"]


def test_mock_cli_flag_smoke(tmp_path):
    from baselines.strong_rag_baseline.cli import main

    exit_code = main(
        [
            "--task", str(EXAMPLE_UNIT / "task.json"),
            "--corpus", str(EXAMPLE_UNIT / "corpus"),
            "--out", str(tmp_path / "answer.json"),
            "--mock",
        ]
    )
    assert exit_code == 0
    answer = json.loads((tmp_path / "answer.json").read_text())
    assert answer["entity_predictions"]
