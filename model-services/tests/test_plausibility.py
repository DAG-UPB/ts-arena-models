"""Offline unit tests for the accuracy plausibility check. Synthetic data only."""

from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plausibility  # noqa: E402


def _items(values):
    return [{"ts": str(i), "value": v} for i, v in enumerate(values)]


def _seasonal_naive(case):
    return [h["value"] for h in case["history"]][-plausibility.PERIOD:]


def test_cases_are_deterministic_and_shaped():
    a, b = plausibility.build_cases(), plausibility.build_cases()
    assert a == b
    assert [len(c["history"]) for c in a] == list(plausibility.CONTEXT_LENGTHS)
    assert all(len(c["truth"]) == plausibility.HORIZON for c in a)


def test_horizon_on_weekdays_and_context_has_weekend():
    for case in plausibility.build_cases():
        stamps = [datetime.strptime(h["ts"], "%Y-%m-%dT%H:%M:%S.000Z") for h in case["history"]]
        assert any(t.weekday() >= 5 for t in stamps)
        assert all(t.weekday() < 5 for t in stamps[-plausibility.PERIOD:])


def test_truth_passes():
    cases = plausibility.build_cases()
    result = plausibility.evaluate([_items(c["truth"]) for c in cases], cases)
    assert result["passed"], result["errors"]
    assert all(s == 0.0 for s in result["scores"].values())


def test_seasonal_naive_passes():
    cases = plausibility.build_cases()
    result = plausibility.evaluate([_items(_seasonal_naive(c)) for c in cases], cases)
    assert result["passed"], result["errors"]


def test_flat_mean_fails():
    cases = plausibility.build_cases()
    flat = []
    for c in cases:
        values = [h["value"] for h in c["history"]]
        flat.append(_items([sum(values) / len(values)] * plausibility.HORIZON))
    result = plausibility.evaluate(flat, cases)
    assert not result["passed"]
    assert len(result["errors"]) == len(cases)


def test_phase_shifted_profile_fails():
    cases = plausibility.build_cases()
    half = plausibility.PERIOD // 2
    shifted = [_items(_seasonal_naive(c)[half:] + _seasonal_naive(c)[:half]) for c in cases]
    assert not plausibility.evaluate(shifted, cases)["passed"]


def test_wrong_series_count_or_length_fails():
    cases = plausibility.build_cases()
    good = [_items(c["truth"]) for c in cases]
    assert not plausibility.evaluate(good[:1], cases)["passed"]
    assert not plausibility.evaluate([s[:-1] for s in good], cases)["passed"]
    assert not plausibility.evaluate({"not": "a list"}, cases)["passed"]
