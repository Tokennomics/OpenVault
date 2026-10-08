"""answer.json marshalling for the minimal Track 4 baseline.

Assembles the ``entity_predictions`` array in the entity_predictions schema
(``qfbench2_common/schemas/analysis.schema.json``): each entity carries a label,
point_forecast, a 90% interval ({level, lo, hi}), and >=1 grounded claim. A final
embargo filter drops any citation whose document post-dates the cutoff.
"""
from __future__ import annotations

from typing import Any


def filter_embargoed(
    claims: list[dict[str, Any]],
    cutoff: str,
    doc_dates: dict[str, str | None],
) -> list[dict[str, Any]]:
    """Drop every claim whose cited document post-dates the embargo cutoff.

    A citation is dropped when ``doc_dates[doc_id] > cutoff`` on the ISO-8601 strings
    (lexicographic == chronological for that one spelling).

    An unknown or undated ``doc_id`` survives this filter, and that is NOT a safe outcome:
    since 2026-08-22 the scorer's embargo gate treats an unresolvable citation and a document
    with no usable ``doc_date`` as **violations**, not as "could not tell". Either one fails
    the unit outright — see ``qfbench2_track_analysis.corpus.CorpusIndex.embargo_report``,
    where `unresolved` and `undated` both count toward ``violation_count``, and
    ``qfbench2_common.scoring.faithfulness.EMBARGO_REASONS``, which has no "unknown" member.
    Passing this filter therefore means "not caught here", not "eligible". Cite only documents
    this agent resolved from the unit's own corpus, with a ``doc_date`` it actually read.
    """
    kept: list[dict[str, Any]] = []
    for claim in claims:
        doc_date = doc_dates.get(str(claim.get("doc_id")))
        if isinstance(doc_date, str) and doc_date > cutoff:
            continue
        kept.append(claim)
    return kept


def build_answer(
    task_id: str,
    entity_predictions: list[dict[str, Any]],
    trace: str,
    target_type: str | None = None,
) -> dict[str, Any]:
    """Assemble the answer envelope.

    ``target_type`` must match the unit's card (SUBMISSION_CLI.md invariant 7).
    Hardcoding ``"classification"`` fails every regression and ranking unit:
    ``align_predictions`` refuses a declared type that disagrees with the trusted
    task as ``t4.target_type_mismatch`` -> ``SCHEMA_INVALID_OUTPUT``, i.e. the whole
    submission scores ``W = -0.27``. ``None`` therefore omits the key rather than
    guessing: ``analysis.schema.json`` does not require ``target_type``, and an
    absent one is accepted and scored against whatever the card declares. Emitting
    nothing is safe; emitting the wrong thing is a DNF.
    """
    answer: dict[str, Any] = {
        "task_id": task_id,
        "schema_version": "3",
    }
    if target_type is not None:
        answer["target_type"] = target_type
    answer["entity_predictions"] = entity_predictions
    answer["evidence_trace"] = trace
    return answer


def build_entity_prediction(
    entity_id: str,
    label: str,
    point_forecast: float,
    lo: float,
    hi: float,
    claims: list[dict[str, Any]],
    *,
    cutoff: str | None = None,
    doc_dates: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    # Final embargo backstop: with a cutoff and the corpus dates in hand, silently drop
    # any citation whose doc_date > cutoff_date before writing output. Callers must
    # filter BEFORE their >=1-claim fallback (see cli.run) so an entity emptied here
    # was already re-filled with an eligible citation, not written schema-invalid.
    if cutoff is not None and doc_dates is not None:
        claims = filter_embargoed(claims, cutoff, doc_dates)
    return {
        "entity_id": entity_id,
        "label": label,
        "point_forecast": round(float(point_forecast), 4),
        "interval": {"level": 0.90, "lo": round(float(lo), 4), "hi": round(float(hi), 4)},
        "claims": claims,
    }
