"""Score an explicitly supplied candidate-2 forecast chain using the canonical Track 2 scorer.

## Executive summary (read this first)

This separate opt-in API verifies retained immutable evidence itself before running any gate.
Cards, complete input snapshots and the actual Common/Track 2 source trees must be committed
before outcomes. Exact retained C3 bytes become the forecast input; signed outcome and scale
bytes become organizer references. The ordinary verifier, C1/C2 binding and live factory do
not change. Results are private candidate diagnostics, always non-rankable: authenticated
claims do not establish live clock custody, mounted inputs, first-vintage attribution, full
worker/C7 evidence or reproduction of the scale-generation recipe.
"""
# Exact byte/boolean types prevent mutable look-alikes and implicit trust-mode coercion.
# ruff: noqa: E721

from __future__ import annotations

import json
import pathlib
import stat
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any

import jsonschema
import pyarrow as pa
import pyarrow.parquet as pq
import qfbench2_common
from qfbench2_common.contracts import EvaluationPlan, OrganizerFault, RunRecord, TrustStore
from qfbench2_common.contracts.errors import ContractError
from qfbench2_common.contracts.forecast_protocol import (
    INPUT_VERSION,
    ForecastChainVerification,
    verify_forecast_resolution,
)
from qfbench2_common.taskcard import load_card, load_schema

import qfbench2_track_forecasting

from .cutoff import (
    check_declared_cutoff,
    scan_panel_cutoff,
    scan_text_corpus_cutoff,
    trusted_asof,
)
from .failures import T2Refusal, organizer_fault
from .grid import grid_from_card, grid_from_plan_entry
from .limits import DEFAULT_LIMITS
from .official import OfficialResult, score_roster

_GAPS = (
    "live_receipt_clock_and_append_only_custody_unproved",
    "input_config_model_mounts_not_independently_observed",
    "full_worker_and_c7_evidence_not_validated",
    "source_first_vintage_attribution_not_independently_proved",
    "scale_recipe_generation_not_reproduced",
    "candidate_policy_not_adopted",
)


@dataclass(frozen=True, slots=True)
class CandidateResolutionScore:
    """Private diagnostic plus its verified chain, never an ordinary official-path receipt."""

    diagnostic: OfficialResult
    chain: ForecastChainVerification
    runtime_source_digests: Mapping[str, str]
    acceptance_blockers: tuple[str, ...] = field(default=_GAPS, init=False)
    rankable: bool = field(default=False, init=False)
    production_image_certified: bool = field(default=False, init=False)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise organizer_fault(reason)


def _snapshot(value: Mapping[str, bytes]) -> dict[str, bytes]:
    _require(isinstance(value, Mapping), "candidate evidence requires immutable byte mappings")
    copied = dict(value)
    _require(
        all(isinstance(k, str) and type(v) is bytes for k, v in copied.items()),
        "candidate evidence requires immutable byte members",
    )
    return copied


def _nested(value: Mapping[str, Mapping[str, bytes]]) -> dict[str, dict[str, bytes]]:
    _require(isinstance(value, Mapping), "candidate evidence requires nested byte mappings")
    return {key: _snapshot(members) for key, members in dict(value).items()}


def runtime_source_digests() -> dict[str, str]:
    """Hash complete actual imported package trees with the Hub's canonical build algorithm.

    This is a measured source identity, not a release stamp or wheel/production-image claim.
    No caller-supplied path, metadata identity or digest string can replace this measurement.
    """
    result = {}
    for prefix, package, key in (
        ("qfbench2_common", qfbench2_common, "common_source_tree_digest"),
        ("qfbench2_track_forecasting", qfbench2_track_forecasting, "track_source_tree_digest"),
    ):
        source_file = package.__file__
        if not isinstance(source_file, str):
            raise organizer_fault("candidate scorer package has no file origin")
        root = pathlib.Path(source_file).absolute().parent
        _require(
            not any(p.is_symlink() for p in (root, *root.parents))
            and tuple(pathlib.Path(p).absolute() for p in package.__path__) == (root,),
            "candidate scorer package has linked or ambiguous origins",
        )
        members = set()
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            if "__pycache__" in relative.parts or path.suffix in (".pyc", ".pyo"):
                continue
            _require(not path.is_symlink(), "candidate scorer package contains linked members")
            if path.is_file():
                _require(path.stat().st_nlink == 1, "candidate scorer source has multiple links")
                members.add(path.absolute())
        for name, module in tuple(sys.modules.items()):
            if name == prefix or name.startswith(prefix + "."):
                source = getattr(module, "__file__", None)
                spec = getattr(module, "__spec__", None)
                _require(
                    isinstance(source, str)
                    and pathlib.Path(source).absolute() in members
                    and spec is not None
                    and spec.origin == source,
                    "candidate scorer imported a module outside its complete source commitment",
                )
        result[key] = qfbench2_common.package_tree_digest(root)
    return result


def _stage(root: pathlib.Path, members: Mapping[str, bytes]) -> dict[pathlib.Path, bytes]:
    """Only already-verified canonical paths enter a fresh private directory."""
    expected = {}
    for name, raw in members.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with path.open("xb") as stream:
            stream.write(raw)
        path.chmod(0o400)
        expected[path] = raw
    return expected


def _scratch_snapshot(root: pathlib.Path) -> dict[str, bytes]:
    result = {}
    for path in root.rglob("*"):
        _require(not path.is_symlink(), "candidate staging acquired a link")
        if path.is_dir():
            continue
        info = path.stat()
        _require(
            stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
            "candidate staging acquired a nonplain file",
        )
        result[path.relative_to(root).as_posix()] = path.read_bytes()
    return result


def _card(root: pathlib.Path, protocol: dict[str, Any], entry: Any) -> dict[str, Any]:
    card = load_card(root)
    jsonschema.Draft202012Validator(load_schema("taskcard.schema.json")).validate(card)
    _require(card["task"]["track"] == "forecasting", "candidate card is not forecasting")
    _require(
        grid_from_card(card) == grid_from_plan_entry(entry),
        "candidate committed card and plan grid disagree",
    )
    params = card.get("scoring", {}).get("params", {})
    recipe = protocol["scale_recipe"]
    _require(
        params.get("weights") == recipe["weights"]
        and params.get("joint", "variogram") == recipe["joint_statistics"][entry.unit_handle],
        "candidate card and precommitted scale recipe disagree",
    )
    asof = trusted_asof(card)
    _require(
        asof <= protocol["schedule"]["information_cutoff"][:10],
        "candidate card cutoff exceeds the protocol information cutoff",
    )
    check_declared_cutoff(card, unit_handle=entry.unit_handle)
    # These are the canonical input scans, over exactly the trees committed before outcomes.
    information_cutoff = protocol["schedule"]["information_cutoff"]
    scan_panel_cutoff(root / "panels", asof, information_cutoff=information_cutoff)
    text_path = card.get("text", {}).get("path", "text/")
    _require(text_path == "text/", "candidate input profile requires the canonical text directory")
    scan_text_corpus_cutoff(
        root / "text", asof, information_cutoff=information_cutoff, strict_coverage=True
    )
    return card


def score_forecast_resolution(
    protocol: bytes,
    receipt: bytes,
    resolution: bytes,
    *,
    organizer_trust: TrustStore,
    runner_trust: TrustStore,
    receipt_trust: TrustStore,
    now: datetime,
    descriptor: bytes,
    config: bytes,
    model_dependencies: Mapping[str, bytes],
    records: Mapping[str, bytes],
    trees: Mapping[str, bytes],
    forecasts: Mapping[str, Mapping[str, bytes]],
    outcomes: Mapping[str, bytes],
    scales: Mapping[str, bytes],
    input_snapshots: Mapping[str, Mapping[str, bytes]],
    source_snapshots: Mapping[str, bytes],
    require_production_trust: bool = True,
) -> CandidateResolutionScore:
    """Verify candidate-2 bytes, then score them; evidence defects abort as organizer faults.

    ``require_production_trust=False`` is exclusively an explicit development-fixture rehearsal.
    No trust downgrade is inferred from documents, keys, timestamps or input directories.
    """
    try:
        _require(type(require_production_trust) is bool, "candidate trust mode must be boolean")
        # Snapshot once before verification: no later reopen of caller-owned maps or paths.
        models, records, trees, outcomes, scales, sources = (
            _snapshot(v)
            for v in (model_dependencies, records, trees, outcomes, scales, source_snapshots)
        )
        forecasts, inputs = _nested(forecasts), _nested(input_snapshots)
        chain = verify_forecast_resolution(
            protocol,
            receipt,
            resolution,
            organizer_trust=organizer_trust,
            runner_trust=runner_trust,
            receipt_trust=receipt_trust,
            now=now,
            descriptor=descriptor,
            config=config,
            model_dependencies=models,
            records=records,
            trees=trees,
            forecasts=forecasts,
            outcomes=outcomes,
            scales=scales,
            input_snapshots=inputs,
            source_snapshots=sources,
            require_production_trust=require_production_trust,
        )
        proto = json.loads(protocol)
        _require(proto["schema_version"] == INPUT_VERSION, "scoring requires candidate-2 evidence")
        actual_runtime = runtime_source_digests()
        _require(
            actual_runtime == proto["scoring_inputs"]["runtime"],
            "candidate scorer source identity differs from the precommitted runtime",
        )
        plan = EvaluationPlan.from_mapping(json.loads(resolution)["plan"])
        _require(
            not plan.required_evidence["judge"], "forecast candidate has unsupported judge evidence"
        )
        with tempfile.TemporaryDirectory(prefix="qfbench2-t2-candidate-") as temporary:
            scratch = pathlib.Path(temporary)
            input_root, ref_root, res_root = (scratch / name for name in ("inputs", "ref", "res"))
            digests = {}
            expected_files = {}
            for entry in plan.expected_units:
                handle = entry.unit_handle
                expected_files.update(_stage(input_root / handle, inputs[handle]))
                _card(input_root / handle, proto, entry)
                expected_files.update(
                    _stage(
                        ref_root / handle,
                        {
                            "card.toml": inputs[handle]["card.toml"],
                            "reference/ref_scale.json": scales[handle],
                        },
                    )
                )
                cells = json.loads(outcomes[handle])["cells"]
                table = pa.table(
                    {name: [cell[name] for cell in cells] for name in ("asset", "horizon", "value")}
                )
                buffer = pa.BufferOutputStream()
                pq.write_table(table, buffer)
                expected_files.update(
                    _stage(
                        ref_root / handle,
                        {
                            "reference/realized.parquet": buffer.getvalue().to_pybytes(),
                        },
                    )
                )
                expected_files.update(_stage(res_root / handle, forecasts[handle]))
                record = RunRecord.from_mapping(json.loads(records[handle]))
                digests[handle] = (
                    record.attestation_payload_digest(),
                    record.bindings["sanitized_tree_digest"],
                )
            before = {
                path.relative_to(scratch).as_posix(): raw for path, raw in expected_files.items()
            }
            _require(
                before == _scratch_snapshot(scratch),
                "candidate staging differs from verified bytes",
            )
            result = score_roster(
                plan, ref_root, res_root, limits=DEFAULT_LIMITS, evidence_digests=digests
            )
            _require(
                before == _scratch_snapshot(scratch),
                "candidate scoring inputs changed during evaluation",
            )
            _require(
                actual_runtime == runtime_source_digests(),
                "candidate runtime changed during evaluation",
            )
        provenance = dict(result.provenance)
        provenance.update(
            candidate_version=INPUT_VERSION,
            rankable=False,
            policy_adopted=False,
            protocol_digest=chain.protocol_digest,
            receipt_digest=chain.receipt_digest,
            resolution_digest=chain.resolution_digest,
            acceptance_blockers=list(_GAPS),
        )
        diagnostic = OfficialResult(
            result.aggregate, result.rows, provenance, result.operator_reasons
        )
        return CandidateResolutionScore(diagnostic, chain, MappingProxyType(actual_runtime))
    except OrganizerFault:
        raise
    except (
        ContractError,
        T2Refusal,
        ValueError,
        TypeError,
        KeyError,
        OSError,
        RecursionError,
        jsonschema.ValidationError,
    ) as exc:
        raise organizer_fault(
            "candidate forecast evidence is incomplete, inconsistent or unsupported; "
            "repair retained organizer evidence before scoring"
        ) from exc
