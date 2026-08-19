"""Quantile contract checker for TS-Arena model-service forecast responses.

This module is pure-Python (stdlib + optional numpy) and contains NO imports of any
model library. It enforces the *structural* probabilistic forecast contract that every
model service in this repo must satisfy, regardless of the underlying model.

Contract summary (see issue models-#13, Task D):
  - A forecast point ("ForecastItem") is a dict:
        {"ts": <str>, "value": <float>,
         "probabilistic_values": {"q_0.1": float, ..., "q_0.9": float}}
    where the quantile keys are exactly `q_<level>` for level in 0.1..0.9 (deciles).
  - `probabilistic_values` may be `{}` for point-only models.
  - Quantile values must be finite and monotone non-decreasing in level for each step.
  - When quantiles are present, `value` must equal `q_0.5` (median consistency), within tolerance.
  - A `/predict` response is `{"prediction": List[ForecastItem]}` (single series) or
    `{"prediction": List[List[ForecastItem]]}` (batch / multi-series).
  - Each quantile series' list length must equal the horizon.

Sample-based models are intentionally unseeded (issue models-#3): this checker
asserts STRUCTURAL properties only, never exact numeric values.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

EXPECTED_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
EXPECTED_KEYS = {f"q_{l}" for l in EXPECTED_LEVELS}


def _is_finite_number(x: Any) -> bool:
    """True if x is an int/float (not bool) and finite."""
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return True
    if isinstance(x, float):
        return not (math.isnan(x) or math.isinf(x))
    return False


def _level_from_key(key: str) -> float | None:
    """Parse a quantile key like 'q_0.5' -> 0.5; returns None if it does not parse."""
    if not isinstance(key, str) or not key.startswith("q_"):
        return None
    try:
        return float(key[2:])
    except ValueError:
        return None


def validate_point(
    point: dict,
    horizon: int,
    *,
    value_tol: float = 1e-6,
    mono_tol: float = 1e-9,
) -> list[str]:
    """Validate ONE forecast point dict; returns a list of human-readable errors
    (empty list == valid).

    Length-of-series is NOT checked here (it is a per-series property).
    """
    errors: list[str] = []
    if not isinstance(point, dict):
        return [f"point is not a dict (got {type(point).__name__})"]

    pv = point.get("probabilistic_values", {})

    if pv is None:
        errors.append("probabilistic_values is null (expected dict, possibly empty {})")
        return errors

    if not isinstance(pv, dict):
        errors.append(
            f"probabilistic_values must be a dict (got {type(pv).__name__})"
        )
        return errors

    keys = list(pv.keys())
    key_set = set(keys)

    # No duplicate keys (a dict can't actually hold duplicates, but set/list mismatch
    # catches anything pathological from JSON漂 / misuse). For a real dict this is a
    # no-op; we keep it for robustness against dict-like oddities.
    if len(key_set) != len(keys):
        errors.append("probabilistic_values has duplicate keys")

    # No unknown / extra keys.
    unknown = key_set - EXPECTED_KEYS
    if unknown:
        errors.append(f"unknown quantile keys (not in {sorted(EXPECTED_KEYS)}): {sorted(unknown)}")

    present = key_set & EXPECTED_KEYS

    # All quantile values must be finite numbers.
    for k in present:
        v = pv.get(k)
        if not _is_finite_number(v):
            errors.append(f"quantile {k} value is not a finite number: {v!r}")

    # Monotone non-decreasing in level for the levels that are present.
    present_levels = sorted(_level_from_key(k) for k in present)
    prev_level = None
    prev_val = None
    for lvl in present_levels:
        k = f"q_{lvl}"
        v = pv.get(k)
        if not _is_finite_number(v):
            continue  # already reported above
        if prev_level is not None:
            if v < prev_val - mono_tol:
                errors.append(
                    f"quantiles not monotone non-decreasing: q_{prev_level}={prev_val!r} "
                    f"> q_{lvl}={v!r}"
                )
        prev_level = lvl
        prev_val = v

    # Median consistency: if q_0.5 present and `value` present, |value - q_0.5| <= value_tol.
    median_key = "q_0.5"
    if median_key in present:
        value = point.get("value")
        q50 = pv.get(median_key)
        if value is not None and _is_finite_number(q50):
            if not _is_finite_number(value):
                errors.append(f"point value is not a finite number: {value!r}")
            else:
                if abs(float(value) - float(q50)) > value_tol:
                    errors.append(
                        f"value {float(value)!r} != q_0.5 {float(q50)!r} "
                        f"(diff {abs(float(value) - float(q50))!r} > tol {value_tol!r})"
                    )
    else:
        # q_0.5 absent but other quantiles present: unusual, warn only (not an error).
        if present and len(present) > 0:
            # Not an error per spec; nothing to append. (Caller may surface a warning.)
            pass

    return errors


def validate_series(
    series: list[dict],
    horizon: int,
    *,
    value_tol: float = 1e-6,
    mono_tol: float = 1e-9,
) -> list[str]:
    """Validate a single series (a list of ForecastItem dicts). Returns errors."""
    errors: list[str] = []
    if not isinstance(series, list):
        return [f"series is not a list (got {type(series).__name__})"]

    if len(series) != horizon:
        errors.append(f"series length {len(series)} != horizon {horizon}")

    # Validate each point.
    for i, point in enumerate(series):
        if not isinstance(point, dict):
            errors.append(f"series[{i}] is not a dict (got {type(point).__name__})")
            continue
        point_errors = validate_point(
            point, horizon, value_tol=value_tol, mono_tol=mono_tol
        )
        for e in point_errors:
            errors.append(f"series[{i}]: {e}")

    # Consistency: the set of quantile keys must be the SAME across all points
    # in the series (all-or-none per series is the convention).
    key_sets: list[set[str]] = []
    for point in series:
        if not isinstance(point, dict):
            key_sets.append(set())
            continue
        pv = point.get("probabilistic_values", {})
        if not isinstance(pv, dict):
            key_sets.append(set())
            continue
        key_sets.append(set(pv.keys()) & EXPECTED_KEYS)

    if key_sets:
        ref = key_sets[0]
        for i, ks in enumerate(key_sets[1:], start=1):
            if ks != ref:
                errors.append(
                    f"quantile-key set inconsistent within series: series[0] keys={sorted(ref)} "
                    f"vs series[{i}] keys={sorted(ks)}"
                )

    # For each present quantile key, its values list length must equal horizon.
    # (Already implied by len(series)==horizon + all-or-none, but be explicit.)
    if key_sets:
        present_keys = key_sets[0]
        for k in present_keys:
            vals = []
            for point in series:
                if not isinstance(point, dict):
                    vals.append(None)
                    continue
                pv = point.get("probabilistic_values", {})
                if not isinstance(pv, dict):
                    vals.append(None)
                    continue
                vals.append(pv.get(k))
            if len(vals) != horizon:
                errors.append(
                    f"quantile {k} values list length {len(vals)} != horizon {horizon}"
                )

    return errors


def _is_series_list(prediction_item: Any) -> bool:
    """True if a prediction-item looks like a single ForecastItem dict (not a list of them)."""
    return isinstance(prediction_item, dict)


def validate_response(
    prediction: Any,
    horizon: int,
    *,
    value_tol: float = 1e-6,
    mono_tol: float = 1e-9,
) -> dict:
    """Validate a raw `prediction` payload (single-series OR batch).

    Auto-detects batch vs single by inspecting `prediction[0]`:
      - if prediction[0] is a list -> batch: List[List[ForecastItem]]
      - if prediction[0] is a dict  -> single: List[ForecastItem]

    Returns:
      {
        "passed": bool,
        "n_series": int,
        "n_points": int,
        "n_with_quantiles": int,
        "errors": list[str],
      }
    """
    errors: list[str] = []
    n_series = 0
    n_points = 0
    n_with_quantiles = 0

    if not isinstance(prediction, list):
        return {
            "passed": False,
            "n_series": 0,
            "n_points": 0,
            "n_with_quantiles": 0,
            "errors": [f"prediction must be a list (got {type(prediction).__name__})"],
        }

    if len(prediction) == 0:
        return {
            "passed": False,
            "n_series": 0,
            "n_points": 0,
            "n_with_quantiles": 0,
            "errors": ["prediction list is empty"],
        }

    first = prediction[0]
    is_batch = isinstance(first, list)
    is_single = isinstance(first, dict)

    if not (is_batch or is_single):
        return {
            "passed": False,
            "n_series": 0,
            "n_points": 0,
            "n_with_quantiles": 0,
            "errors": [
                f"cannot detect single vs batch: prediction[0] is {type(first).__name__}"
            ],
        }

    if is_batch:
        # List[List[ForecastItem]]
        n_series = len(prediction)
        for i, series in enumerate(prediction):
            if not isinstance(series, list):
                errors.append(f"batch series[{i}] is not a list (got {type(series).__name__})")
                continue
            n_points += len(series)
            series_errors = validate_series(
                series, horizon, value_tol=value_tol, mono_tol=mono_tol
            )
            for e in series_errors:
                errors.append(f"series[{i}]: {e}")
            # Count points with quantiles for this series.
            for j, point in enumerate(series):
                if isinstance(point, dict):
                    pv = point.get("probabilistic_values", {})
                    if isinstance(pv, dict) and len(pv) > 0:
                        n_with_quantiles += 1
    else:
        # List[ForecastItem] (single series)
        n_series = 1
        n_points = len(prediction)
        series_errors = validate_series(
            prediction, horizon, value_tol=value_tol, mono_tol=mono_tol
        )
        errors.extend(series_errors)
        for j, point in enumerate(prediction):
            if isinstance(point, dict):
                pv = point.get("probabilistic_values", {})
                if isinstance(pv, dict) and len(pv) > 0:
                    n_with_quantiles += 1

    return {
        "passed": len(errors) == 0,
        "n_series": n_series,
        "n_points": n_points,
        "n_with_quantiles": n_with_quantiles,
        "errors": errors,
    }