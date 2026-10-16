"""Accuracy plausibility check for TS-Arena model services.

The quantile contract checks structure only. This check posts strongly seasonal synthetic
series (15-min, daily period 96) of realistic context lengths and requires the returned
median to beat in-sample seasonal naive: MASE (scaled by the seasonal-naive MAE of the
context) below a threshold. A foundation model that misreads its input fails it clearly.

Pure stdlib, deterministic (seeded noise). NO model library is imported.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from typing import Any

PERIOD = 96
FREQ = "15min"
HORIZON = 96
CONTEXT_LENGTHS = (1000, 512)
MAX_MASE = 1.0


def _value(t: datetime, rng: random.Random, variant: int) -> float:
    day = 2 * math.pi * (t.hour * 4 + t.minute // 15) / PERIOD
    level = 100.0 + 10.0 * variant
    profile = 20.0 * math.sin(day - math.pi / 2) + 8.0 * math.sin(2 * day + variant)
    weekend = -12.0 if t.weekday() >= 5 else 0.0
    return level + profile + weekend + rng.gauss(0.0, 2.0)


def build_cases() -> list[dict[str, Any]]:
    """One case per (context length, variant): history items plus the held-out truth.

    The horizon and the day before it are weekdays; every context contains a weekend.
    """
    forecast_start = datetime(2026, 1, 14, 10, 0, 0)  # a Wednesday
    cases = []
    for variant, n in enumerate(CONTEXT_LENGTHS):
        rng = random.Random(1000 + variant)
        stamps = [forecast_start + timedelta(minutes=15 * (i - n)) for i in range(n + HORIZON)]
        values = [_value(t, rng, variant) for t in stamps]
        history = [
            {"ts": stamps[i].strftime("%Y-%m-%dT%H:%M:%S.000Z"), "value": values[i]}
            for i in range(n)
        ]
        cases.append({"name": f"seasonal-{n}", "history": history, "truth": values[n:]})
    return cases


def mase(context: list[float], truth: list[float], forecast: list[float], m: int = PERIOD) -> float:
    scale = sum(abs(context[i] - context[i - m]) for i in range(m, len(context))) / (len(context) - m)
    return sum(abs(f - t) for f, t in zip(forecast, truth)) / len(truth) / scale


def evaluate(prediction: Any, cases: list[dict[str, Any]], max_mase: float = MAX_MASE) -> dict:
    """Score a batch `prediction` (List[List[ForecastItem]]) against `cases`, in order."""
    errors: list[str] = []
    scores: dict[str, float] = {}
    if not isinstance(prediction, list) or len(prediction) != len(cases):
        n = len(prediction) if isinstance(prediction, list) else type(prediction).__name__
        return {"passed": False, "scores": scores, "errors": [f"expected {len(cases)} series, got {n}"]}
    for case, series in zip(cases, prediction):
        try:
            forecast = [float(p["value"]) for p in series]
        except (TypeError, KeyError, ValueError) as e:
            errors.append(f"{case['name']}: unreadable forecast ({e})")
            continue
        if len(forecast) != len(case["truth"]):
            errors.append(f"{case['name']}: {len(forecast)} points, expected {len(case['truth'])}")
            continue
        context = [h["value"] for h in case["history"]]
        score = mase(context, case["truth"], forecast)
        scores[case["name"]] = score
        if not math.isfinite(score) or score >= max_mase:
            errors.append(f"{case['name']}: MASE {score:.3f} >= {max_mase} (worse than seasonal naive)")
    return {"passed": not errors, "scores": scores, "errors": errors}
