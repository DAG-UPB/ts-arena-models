"""Service-level tests (run from the service directory:  pytest tests -q).  Needs weights/ and fastapi+httpx.
The generic structural contract is checked with the repository's own checker (model-services/tests/quantile_contract.py)."""
import math, os, sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SVC = os.path.dirname(HERE)
sys.path.insert(0, SVC)
sys.path.insert(0, os.path.join(SVC, "..", "tests"))   # model-services/tests/quantile_contract.py
from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402
import quantile_contract  # noqa: E402

client = TestClient(app)
STEP = {"1min": 60, "15min": 900, "30min": 1800, "h": 3600, "D": 86400, "W": 604800}


def _ts(t):
    return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")


WINTER_REF = np.load(os.path.join(SVC, "weights", "clock_ref.npz"))["winter"]


def load_like(n, step_s, end=datetime(2026, 3, 10, 7, 45, tzinfo=timezone.utc), seed=0, scale=12000.0, smard_clock=False):
    """Synthetic series.  smard_clock=True imitates a SMARD net-load series on the platform clock (daily profile of the
    reference winter shape by platform hour, weekend dip), which the specialist accepts; otherwise a generic sinusoid."""
    rng = np.random.default_rng(seed)
    per_day = max(1, 86400 // step_s)
    t = np.arange(n)
    ts = [end - timedelta(seconds=step_s * (n - 1 - i)) for i in range(n)]
    if smard_clock:
        hour = np.array([x.hour for x in ts]); dow = np.array([x.weekday() for x in ts])
        v = scale * WINTER_REF[hour] * np.where(dow >= 5, 0.88, 1.0) + rng.normal(0, scale * 0.004, n)
    else:
        v = scale * (1 + 0.25 * np.sin(2 * np.pi * (t % per_day) / per_day) + 0.05 * np.sin(2 * np.pi * t / (7 * per_day))) + rng.normal(0, scale * 0.005, n)
    return [{"ts": _ts(ts[i]), "value": float(v[i])} for i in range(n)]


def check(resp, horizon, n_series=None):
    assert resp.status_code == 200, resp.text
    pred = resp.json()["prediction"]
    r = quantile_contract.validate_response(pred, horizon=horizon)
    assert r["passed"], r["errors"][:5]
    assert r["n_with_quantiles"] == r["n_points"]
    if n_series is not None:
        assert r["n_series"] == n_series
    return pred


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "healthy"


@pytest.mark.parametrize("freq,horizon,n", [("15min", 96, 1000), ("h", 72, 1000), ("15min", 24, 1000), ("h", 24, 1000)])
def test_specialist_single(freq, horizon, n):
    hist = load_like(n, STEP[freq], smard_clock=True)
    pred = check(client.post("/predict", json={"history": hist, "horizon": horizon, "freq": freq}), horizon, 1)
    last = datetime.fromisoformat(hist[-1]["ts"].replace("Z", "+00:00"))
    assert pred[0]["ts"] == _ts(last + timedelta(seconds=STEP[freq]))
    assert pred[-1]["ts"] == _ts(last + timedelta(seconds=STEP[freq] * horizon))
    assert all(abs(p["value"] - p["probabilistic_values"]["q_0.5"]) < 1e-9 for p in pred)


def test_specialist_path_used_for_smard_like_clock_and_not_otherwise():
    from app.main import model
    end15 = datetime(2026, 3, 10, 7, 45, tzinfo=timezone.utc); end60 = datetime(2026, 3, 10, 8, 0, tzinfo=timezone.utc)
    model.predict(load_like(1000, 900, end=end15, smard_clock=True), 96, "15min")
    assert model.last_path == "specialist"
    model.predict(load_like(1000, 3600, end=end60, smard_clock=True), 72, "h")
    assert model.last_path == "specialist"
    # the same series 6 h later on the clock (e.g. a true-UTC US grid) is not what the specialist was built for
    shifted = load_like(1000, 900, end=end15 + timedelta(hours=6), smard_clock=True)
    for it, base in zip(shifted, load_like(1000, 900, end=end15, smard_clock=True)):
        it["value"] = base["value"]
    model.predict(shifted, 96, "15min")
    assert model.last_path == "fallback"
    model.predict(load_like(1000, 900, end=end15), 96, "15min")       # generic sinusoid: wrong profile
    assert model.last_path == "fallback"


def test_batch_ragged_with_none_and_gaps():
    series = []
    for k in range(10):
        s = load_like(1000, 900, end=datetime(2026, 3, 10, 7, 45, tzinfo=timezone.utc) - timedelta(minutes=15 * (k % 4)), seed=k, scale=2000 * (k + 1), smard_clock=True)
        if k == 3:                                   # missing points inside the context
            del s[400:410]
        if k == 5:                                   # None values
            for i in range(100, 130):
                s[i]["value"] = None
        series.append(s)
    pred = check(client.post("/predict", json={"history": series, "horizon": 96, "freq": "15min"}), 96, 10)
    for k, p in enumerate(pred):
        last = datetime.fromisoformat(series[k][-1]["ts"].replace("Z", "+00:00"))
        assert p[0]["ts"] == _ts(last + timedelta(minutes=15))


@pytest.mark.parametrize("freq", ["1min", "15min", "30min", "h", "D", "W", "M"])
@pytest.mark.parametrize("n", [1, 5, 48, 400])
def test_every_freq_and_short_histories(freq, n):
    step = STEP.get(freq, 86400 * 30)
    hist = load_like(n, step if freq != "M" else 86400 * 30, scale=50.0)
    check(client.post("/predict", json={"history": hist, "horizon": 12, "freq": freq}), 12, 1)


def test_default_freq_and_harness_style_history():
    base = datetime(2026, 1, 1)
    hist = [{"ts": _ts(base + timedelta(hours=i)), "value": 10.0 + 0.5 * i + 5.0 * math.sin(i / 4.0)} for i in range(48)]
    check(client.post("/predict", json={"history": hist, "horizon": 24, "freq": "h"}), 24, 1)
    check(client.post("/predict", json={"history": [hist, hist], "horizon": 24, "freq": "h"}), 24, 2)


def test_constant_negative_and_zero_series():
    n = 400
    base = datetime(2026, 3, 1, tzinfo=timezone.utc)
    const = [{"ts": _ts(base + timedelta(hours=i)), "value": 42.0} for i in range(n)]
    neg = [{"ts": _ts(base + timedelta(hours=i)), "value": -50.0 + 80.0 * math.sin(i / 3.7)} for i in range(n)]
    zero = [{"ts": _ts(base + timedelta(hours=i)), "value": 0.0} for i in range(n)]
    for h in (const, neg, zero):
        check(client.post("/predict", json={"history": h, "horizon": 72, "freq": "h"}), 72, 1)


def test_duplicate_and_unsorted_timestamps():
    hist = load_like(300, 3600)
    hist = hist[:100] + hist[90:]                    # duplicated block
    hist = hist[::-1][:150][::-1] + hist[:5]         # break order
    check(client.post("/predict", json={"history": hist, "horizon": 24, "freq": "h"}), 24, 1)


def test_horizon_longer_than_net_falls_back():
    check(client.post("/predict", json={"history": load_like(1000, 900), "horizon": 200, "freq": "15min"}), 200, 1)


def test_deterministic():
    hist = load_like(1000, 900)
    a = client.post("/predict", json={"history": hist, "horizon": 96, "freq": "15min"}).json()
    b = client.post("/predict", json={"history": hist, "horizon": 96, "freq": "15min"}).json()
    assert a == b


def test_empty_history_is_400():
    assert client.post("/predict", json={"history": [], "horizon": 4, "freq": "h"}).status_code == 400
