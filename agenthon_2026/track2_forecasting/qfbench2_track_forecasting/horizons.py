"""## Executive summary (read this first)
Resolve monthly sampling steps from explicit observation periods in supplied metadata.
Horizon integers remain the output grid keys. This helper never invents a calendar
conversion from those integers, reads answers, or changes the scoring contract.
The caller must use it only for target series whose observations are monthly.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, cast

import numpy as np


class HorizonMetadataError(ValueError):
    """The supplied inputs do not state one unambiguous monthly target grid."""


def _period(value: Any, *, date_label: bool = False) -> int:
    pattern = r"\d{4}-\d{2}-\d{2}" if date_label else r"\d{4}-\d{2}"
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        raise HorizonMetadataError("Use ISO observation periods or ISO date labels.")
    try:
        parsed = date.fromisoformat(value if date_label else value + "-01")
    except ValueError:
        raise HorizonMetadataError("The observation period or date label is invalid.") from None
    return parsed.year * 12 + parsed.month - 1


def monthly_horizon_steps(
    assets: Sequence[str],
    horizons: Sequence[int],
    last_observations: Mapping[str, str],
    *,
    asof: str,
    card: Mapping[str, Any] | None = None,
    forecast_spec: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Return an [asset, horizon] matrix of monthly steps from the panel anchor.

    Existing ``targets.target_dates`` or ``questions[].target_date`` are usable
    only when the task identifies them as monthly observation-period labels.
    The minimal date-free spelling is ``targets.observation_periods`` aligned
    with ``targets.horizons``, or ``observation_period`` on a question row.
    Every supplied spelling must agree. Absent mappings raise an actionable
    error rather than treating a horizon key as a number of monthly steps.

    ``last_observations`` must be computed from the selected monthly series after
    applying the task cutoff. The as-of date is a knowledge cutoff, not the walk
    anchor: publication lag may leave the final panel observation months earlier.
    No vintage choice or publication date is inferred from the observation month.
    """
    assets, horizons = list(assets), list(horizons)
    if (
        not assets
        or any(not isinstance(a, str) or not a for a in assets)
        or len(set(assets)) != len(assets)
        or not horizons
        or any(type(h) is not int or h <= 0 for h in horizons)  # noqa: E721 - require a built-in JSON integer, never bool
        or len(set(horizons)) != len(horizons)
    ):
        raise HorizonMetadataError("Provide unique assets and positive integer horizon keys.")
    _period(asof, date_label=True)
    grid = {(a, h) for a in assets for h in horizons}
    periods: dict[tuple[str, int], int] = {}

    def add(asset: Any, horizon: Any, value: Any, *, date_label: bool) -> None:
        if not isinstance(asset, str) or type(horizon) is not int or (asset, horizon) not in grid:  # noqa: E721 - require a built-in JSON integer, never bool
            raise HorizonMetadataError("Monthly metadata disagrees with the requested grid.")
        key, period = (asset, horizon), _period(value, date_label=date_label)
        if key in periods and periods[key] != period:
            raise HorizonMetadataError(
                "Conflicting monthly observation periods; correct the task inputs."
            )
        periods[key] = period

    for source in (card, forecast_spec):
        if source is None:
            continue
        targets = source.get("targets", {})
        if not isinstance(targets, Mapping):
            raise HorizonMetadataError("The targets metadata must be an object.")
        for field, is_date in (("observation_periods", False), ("target_dates", True)):
            if field not in targets:
                continue
            declared_h = targets.get("horizons", horizons)
            declared_a = targets.get("asset_ids", assets)
            values = targets[field]
            if (
                not isinstance(declared_h, list)
                or not isinstance(declared_a, list)
                or not isinstance(values, list)
                or len(values) != len(declared_h)
                or not declared_h
                or any(type(h) is not int for h in declared_h)  # noqa: E721 - require a built-in JSON integer, never bool
                or len(set(declared_h)) != len(declared_h)
                or not declared_a
                or any(not isinstance(a, str) for a in declared_a)
                or len(set(declared_a)) != len(declared_a)
            ):
                raise HorizonMetadataError(
                    "Align one monthly observation period with each declared horizon."
                )
            for h, value in zip(declared_h, values, strict=True):
                for asset in declared_a:
                    add(asset, h, value, date_label=is_date)
        questions = source.get("questions", [])
        if not isinstance(questions, list):
            raise HorizonMetadataError("The questions metadata must be an array.")
        seen = set()
        for row in questions:
            if not isinstance(row, Mapping):
                raise HorizonMetadataError("Each question must be an object.")
            if "observation_period" not in row and "target_date" not in row:
                continue
            asset, horizon = row.get("asset"), row.get("horizon")
            if not isinstance(asset, str) or type(horizon) is not int:  # noqa: E721 - require a built-in JSON integer, never bool
                raise HorizonMetadataError(
                    "Each monthly question needs an asset and integer horizon key."
                )
            key = (asset, horizon)
            if key in seen:
                raise HorizonMetadataError("Each monthly question must have a unique grid key.")
            seen.add(key)
            for field, is_date in (("observation_period", False), ("target_date", True)):
                if field in row:
                    add(asset, horizon, row[field], date_label=is_date)
    if set(periods) != grid:
        raise HorizonMetadataError(
            "Monthly target periods are missing; "
            "request the task's explicit observation-period mapping."
        )
    result = np.empty((len(assets), len(horizons)), dtype=float)
    for ai, asset in enumerate(assets):
        anchor = last_observations.get(asset)
        start = _period(anchor, date_label=True)
        if cast(str, anchor) > asof:
            raise HorizonMetadataError("Apply the task cutoff before selecting the panel anchor.")
        for hi, horizon in enumerate(horizons):
            steps = periods[(asset, horizon)] - start
            if steps <= 0:
                raise HorizonMetadataError(
                    "Monthly target periods must follow the panel anchor period."
                )
            result[ai, hi] = steps
    return result
