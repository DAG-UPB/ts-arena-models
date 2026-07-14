"""Offline unit tests for the quantile contract checker.

Synthetic data only — no model libraries imported. Covers good and bad cases
for validate_point, validate_series and validate_response.
"""

import os
import sys

# Make the sibling quantile_contract.py importable regardless of how pytest is
# invoked (from repo root, from model-services/, or from inside tests/).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402

from quantile_contract import (  # noqa: E402
    EXPECTED_KEYS,
    EXPECTED_LEVELS,
    validate_point,
    validate_series,
    validate_response,
)


H = 5  # default horizon for tests


# ---------- helpers to build synthetic data ----------

def _full_quants(level_values: dict[float, float]) -> dict[str, float]:
    return {f"q_{l}": v for l, v in level_values.items()}


def _mono_quants(center: float, spread: float) -> dict[str, float]:
    """Monotone non-decreasing quantiles centered on `center` for q_0.5."""
    levels = EXPECTED_LEVELS
    spread_per_step = spread / 4.0  # so q_0.9 = center + spread, q_0.1 = center - spread
    out = {}
    for l in levels:
        out[f"q_{l}"] = center + (l - 0.5) * spread_per_step
    return out


def _make_series(horizon: int, *, with_quants: bool = True, base: float = 10.0) -> list[dict]:
    series = []
    for i in range(horizon):
        v = base + i
        pv = _mono_quants(v, spread=2.0) if with_quants else {}
        series.append({"ts": f"2026-01-01T{i:02d}:00:00.000Z", "value": float(v), "probabilistic_values": pv})
    return series


def _make_batch(n_series: int, horizon: int, **kw) -> list[list[dict]]:
    return [_make_series(horizon, **kw) for _ in range(n_series)]


# ---------- validate_point: good cases ----------

def test_point_good_full_quants():
    pv = _mono_quants(5.0, spread=1.0)
    pt = {"ts": "2026-01-01T00:00:00.000Z", "value": 5.0, "probabilistic_values": pv}
    assert validate_point(pt, horizon=H) == []


def test_point_good_empty_quants_point_only():
    pt = {"ts": "t", "value": 5.0, "probabilistic_values": {}}
    assert validate_point(pt, horizon=H) == []


def test_point_good_partial_quants_subset():
    # Only a subset of deciles, still monotone; q_0.5 present & consistent.
    pv = {"q_0.1": 3.0, "q_0.5": 5.0, "q_0.9": 7.0}
    pt = {"ts": "t", "value": 5.0, "probabilistic_values": pv}
    assert validate_point(pt, horizon=H) == []


def test_point_good_value_within_tol():
    pv = _mono_quants(5.0, spread=1.0)
    pt = {"ts": "t", "value": 5.0 + 5e-7, "probabilistic_values": pv}  # within default 1e-6 tol
    assert validate_point(pt, horizon=H) == []


def test_point_good_q50_absent_no_median_check():
    # q_0.5 absent but other quants present -> must NOT fail median consistency.
    pv = {"q_0.1": 3.0, "q_0.9": 7.0}
    pt = {"ts": "t", "value": 100.0, "probabilistic_values": pv}
    assert validate_point(pt, horizon=H) == []


def test_point_good_integer_values():
    pv = {"q_0.1": 3, "q_0.5": 5, "q_0.9": 7}  # ints, not floats
    pt = {"ts": "t", "value": 5, "probabilistic_values": pv}
    assert validate_point(pt, horizon=H) == []


# ---------- validate_point: bad cases ----------

def test_point_bad_non_monotone():
    pv = {"q_0.1": 9.0, "q_0.5": 5.0, "q_0.9": 1.0}
    pt = {"ts": "t", "value": 5.0, "probabilistic_values": pv}
    errs = validate_point(pt, horizon=H)
    assert any("not monotone" in e for e in errs)


def test_point_bad_unknown_key():
    pv = {**_mono_quants(5.0, spread=1.0), "q_0.05": 2.0}
    pt = {"ts": "t", "value": 5.0, "probabilistic_values": pv}
    errs = validate_point(pt, horizon=H)
    assert any("unknown" in e for e in errs)


def test_point_bad_extra_key_high():
    pv = {**_mono_quants(5.0, spread=1.0), "q_0.95": 8.0}
    pt = {"ts": "t", "value": 5.0, "probabilistic_values": pv}
    errs = validate_point(pt, horizon=H)
    assert any("unknown" in e for e in errs)


def test_point_bad_nan():
    pv = {**_mono_quants(5.0, spread=1.0), "q_0.1": float("nan")}
    pt = {"ts": "t", "value": 5.0, "probabilistic_values": pv}
    errs = validate_point(pt, horizon=H)
    assert any("not a finite number" in e for e in errs)


def test_point_bad_inf():
    pv = {**_mono_quants(5.0, spread=1.0), "q_0.9": float("inf")}
    pt = {"ts": "t", "value": 5.0, "probabilistic_values": pv}
    errs = validate_point(pt, horizon=H)
    assert any("not a finite number" in e for e in errs)


def test_point_bad_value_neq_q50():
    pv = _mono_quants(5.0, spread=1.0)
    pt = {"ts": "t", "value": 99.0, "probabilistic_values": pv}
    errs = validate_point(pt, horizon=H)
    assert any("!=" in e and "q_0.5" in e for e in errs)


def test_point_bad_pv_not_dict():
    pt = {"ts": "t", "value": 5.0, "probabilistic_values": [1, 2, 3]}
    errs = validate_point(pt, horizon=H)
    assert any("must be a dict" in e for e in errs)


def test_point_bad_pv_null():
    pt = {"ts": "t", "value": 5.0, "probabilistic_values": None}
    errs = validate_point(pt, horizon=H)
    assert any("null" in e for e in errs)


def test_point_bad_not_dict():
    errs = validate_point(["not", "a", "dict"], horizon=H)
    assert any("not a dict" in e for e in errs)


def test_point_bad_bool_value_rejected():
    # bool is a subclass of int -> must NOT be accepted as a finite number per our rule.
    pv = {**_mono_quants(5.0, spread=1.0), "q_0.5": True}
    pt = {"ts": "t", "value": 5.0, "probabilistic_values": pv}
    errs = validate_point(pt, horizon=H)
    assert any("not a finite number" in e for e in errs)


def test_point_bad_string_value_rejected():
    pv = {**_mono_quants(5.0, spread=1.0), "q_0.1": "5.0"}
    pt = {"ts": "t", "value": 5.0, "probabilistic_values": pv}
    errs = validate_point(pt, horizon=H)
    assert any("not a finite number" in e for e in errs)


def test_point_bad_value_nan_within_median_check():
    pv = _mono_quants(5.0, spread=1.0)
    pt = {"ts": "t", "value": float("nan"), "probabilistic_values": pv}
    errs = validate_point(pt, horizon=H)
    assert any("finite number" in e for e in errs)


# ---------- validate_series: good cases ----------

def test_series_good_full():
    s = _make_series(H, with_quants=True)
    assert validate_series(s, horizon=H) == []


def test_series_good_empty_quants():
    s = _make_series(H, with_quants=False)
    assert validate_series(s, horizon=H) == []


def test_series_good_partial_subset_consistent():
    s = []
    for i in range(H):
        pv = {"q_0.1": 1.0 + i, "q_0.5": 2.0 + i, "q_0.9": 3.0 + i}
        s.append({"ts": f"t{i}", "value": 2.0 + i, "probabilistic_values": pv})
    assert validate_series(s, horizon=H) == []


# ---------- validate_series: bad cases ----------

def test_series_bad_wrong_length():
    s = _make_series(H + 2, with_quants=True)
    errs = validate_series(s, horizon=H)
    assert any("length" in e and "horizon" in e for e in errs)


def test_series_bad_inconsistent_keys():
    s = _make_series(H, with_quants=True)
    s[2]["probabilistic_values"] = {}  # drop quants at one point
    errs = validate_series(s, horizon=H)
    assert any("inconsistent" in e for e in errs)


def test_series_bad_point_non_monotone():
    s = _make_series(H, with_quants=True)
    s[1]["probabilistic_values"]["q_0.5"] = 0.0   # smaller than q_0.1
    s[1]["probabilistic_values"]["q_0.1"] = 9.0
    errs = validate_series(s, horizon=H)
    assert any("not monotone" in e for e in errs)


def test_series_bad_not_list():
    errs = validate_series("not a list", horizon=H)
    assert any("not a list" in e for e in errs)


# ---------- validate_response: single series ----------

def test_response_good_single():
    pred = _make_series(H, with_quants=True)
    out = validate_response(pred, horizon=H)
    assert out["passed"] is True
    assert out["n_series"] == 1
    assert out["n_points"] == H
    assert out["n_with_quantiles"] == H
    assert out["errors"] == []


def test_response_good_single_point_only():
    pred = _make_series(H, with_quants=False)
    out = validate_response(pred, horizon=H)
    assert out["passed"] is True
    assert out["n_with_quantiles"] == 0


def test_response_bad_single_wrong_length():
    pred = _make_series(H - 1, with_quants=True)
    out = validate_response(pred, horizon=H)
    assert out["passed"] is False
    assert any("horizon" in e for e in out["errors"])


# ---------- validate_response: batch ----------

def test_response_good_batch():
    pred = _make_batch(3, H, with_quants=True)
    out = validate_response(pred, horizon=H)
    assert out["passed"] is True
    assert out["n_series"] == 3
    assert out["n_points"] == 3 * H
    assert out["n_with_quantiles"] == 3 * H
    assert out["errors"] == []


def test_response_good_batch_point_only():
    pred = _make_batch(2, H, with_quants=False)
    out = validate_response(pred, horizon=H)
    assert out["passed"] is True
    assert out["n_with_quantiles"] == 0
    assert out["n_series"] == 2


def test_response_bad_batch_one_series_bad():
    pred = _make_batch(3, H, with_quants=True)
    # break series[1]: wrong length
    pred[1] = _make_series(H - 1, with_quants=True)
    out = validate_response(pred, horizon=H)
    assert out["passed"] is False
    assert out["n_series"] == 3
    assert any("horizon" in e for e in out["errors"])


def test_response_bad_batch_one_series_non_monotone():
    pred = _make_batch(2, H, with_quants=True)
    pred[0][2]["probabilistic_values"]["q_0.5"] = 0.0
    pred[0][2]["probabilistic_values"]["q_0.1"] = 9.0
    out = validate_response(pred, horizon=H)
    assert out["passed"] is False
    assert any("not monotone" in e for e in out["errors"])


# ---------- validate_response: shape / detection errors ----------

def test_response_bad_not_list():
    out = validate_response({"prediction": []}, horizon=H)
    assert out["passed"] is False
    assert out["n_series"] == 0
    assert any("must be a list" in e for e in out["errors"])


def test_response_bad_empty_list():
    out = validate_response([], horizon=H)
    assert out["passed"] is False
    assert any("empty" in e for e in out["errors"])


def test_response_bad_unrecognized_shape():
    out = validate_response([123, 456], horizon=H)
    assert out["passed"] is False
    assert any("cannot detect" in e for e in out["errors"])


# ---------- constants sanity ----------

def test_expected_keys_are_deciles():
    assert EXPECTED_LEVELS == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert EXPECTED_KEYS == {
        "q_0.1", "q_0.2", "q_0.3", "q_0.4", "q_0.5",
        "q_0.6", "q_0.7", "q_0.8", "q_0.9",
    }