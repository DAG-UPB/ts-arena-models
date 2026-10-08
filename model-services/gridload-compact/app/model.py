"""Compact probabilistic grid-load forecaster (TS-Arena Reference-track service model).

Two tiny MLP quantile nets (15 min x 96 steps, 1 h x 72 steps), numpy feature pipeline, ONNX Runtime inference
(no torch in the image).  Anything the specialist was not built for goes to a generic seasonal fallback.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from . import cal_np, fallback, feats

logger = logging.getLogger(__name__)

LEVELS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
FREQ_MIN = {"15min": 15, "h": 60}
FREQ_SECONDS = {"1min": 60, "15min": 900, "30min": 1800, "h": 3600, "D": 86400, "W": 604800}
HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.environ.get("GRIDLOAD_WEIGHTS", os.path.join(HERE, "..", "weights"))


def parse_ts(s: str) -> Optional[int]:
    """ISO-8601 -> epoch seconds (UTC); None when unparsable."""
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return int(d.timestamp())
    except Exception:
        return None


def _add_months(ts: int, k: int) -> int:
    d = datetime.fromtimestamp(ts, tz=timezone.utc)
    m = d.month - 1 + k
    y, m = d.year + m // 12, m % 12 + 1
    day = min(d.day, 28)
    return int(d.replace(year=y, month=m, day=day).timestamp())


def future_timestamps(last_ts: int, horizon: int, freq: str) -> List[int]:
    if freq == "M":
        return [_add_months(last_ts, i) for i in range(1, horizon + 1)]
    step = FREQ_SECONDS.get(freq, 3600)
    return [last_ts + i * step for i in range(1, horizon + 1)]


class GridLoadCompact:
    def __init__(self, precision: Optional[str] = None):
        import onnxruntime as ort

        self.precision = (precision or os.getenv("GRIDLOAD_PRECISION", "int8")).lower()
        suffix = "_int8" if self.precision == "int8" else ""
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        z = np.load(os.path.join(WEIGHTS, "hol_table.npz"))
        self.cal = cal_np.Calendar(int(z["day0"]), z["table"])
        c = np.load(os.path.join(WEIGHTS, "clock_ref.npz"))
        self.clock_ref = {0: c["winter"], 1: c["summer"]}
        self.sessions: Dict[int, Any] = {}
        self.specs: Dict[int, feats.Spec] = {}
        self.info: Dict[int, Dict[str, Any]] = {}
        for fm in (15, 60):
            path = os.path.join(WEIGHTS, f"net{fm}{suffix}.onnx")
            self.sessions[fm] = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            info_path = os.path.join(WEIGHTS, f"net{fm}_info.json")      # written by tsb.export: the feature spec the net was trained with
            self.info[fm] = json.load(open(info_path)) if os.path.exists(info_path) else {}
            self.specs[fm] = feats.Spec(fm, **self.info[fm].get("spec_kwargs", {}))
            self._selfcheck(fm)
        logger.info("GridLoadCompact ready (precision=%s)", self.precision)

    def _selfcheck(self, fm: int) -> None:
        """Fail at start-up (not silently per request) when the ONNX net and the feature spec disagree."""
        sp = self.specs[fm]
        t = np.arange(sp.L)
        W = (10000.0 + 2000.0 * np.sin(2 * np.pi * t / sp.P)).astype(np.float32)
        last_ts = (1_772_000_000 // sp.step_s) * sp.step_s
        pos_ts = last_ts + (np.arange(sp.L + sp.H, dtype=np.int64) - (sp.L - 1)) * sp.step_s
        cal = {k: v[None] for k, v in self.cal.features(pos_ts).items()}
        f = feats.build(sp, np.concatenate([W, np.full(sp.H, np.nan, np.float32)])[None], cal, with_targets=False)
        q = self._run(fm, f)
        if tuple(q.shape) != (1, sp.H, 9) or not np.isfinite(q).all():
            raise RuntimeError(f"net{fm}: unexpected output {q.shape} (feature spec {self.info[fm].get('spec_kwargs')})")

    # ------------------------------------------------------------------ helpers
    def _run(self, fm: int, f: Dict[str, np.ndarray]) -> np.ndarray:
        return self.sessions[fm].run(
            ["q"], {"ctx": f["ctx"], "lag": f["lag"], "step": f["step"], "base": f["base"]}
        )[0]

    def _is_load_like(self, W: np.ndarray, P: int) -> bool:
        """Positive, densely observed, moderately dispersed series.  (Periodicity is checked by the clock-profile match.)"""
        with np.errstate(all="ignore"):
            tail = W[-4 * P:]
            fin = np.isfinite(tail)
            if fin.mean() < 0.9 or not fin.any():
                return False
            x = tail[fin]
            return bool(x.min() > 0 and x.std() / x.mean() <= 0.6)

    def _clock_ok(self, W: np.ndarray, last_ts: int, sp: "feats.Spec") -> bool:
        """The specialist was trained on SMARD series in the platform's timestamp convention (UTC stamps that run 1-2 h early).
        Accept only contexts whose mean daily profile, by platform hour-of-day, matches the reference profile of the
        season within +-2 h; contexts on another clock or another kind of grid go to the fallback."""
        with np.errstate(all="ignore"):
            n = min(14, sp.L // sp.P) * sp.P            # 10 days at 15 min, 14 days at 1 h
            seg = W[-n:]
            if seg.size < n or np.isfinite(seg).mean() < 0.9 or np.nanmin(seg) <= 0:
                return False
            ts = last_ts + (np.arange(-n + 1, 1, dtype=np.int64)) * sp.step_s
            hour = (ts % 86400) // 3600
            prof = np.array([np.nanmean(seg[hour == h]) for h in range(24)])
            if not np.isfinite(prof).all():
                return False
            prof = prof / prof.mean()
            summer = int(bool(cal_np._is_summer(np.array([last_ts + 7200]))[0]))
            ref = self.clock_ref[summer]
            d = {k: float(np.mean((np.roll(prof, k) - ref) ** 2)) for k in range(-6, 7)}
            best = min(d, key=d.get)
            return abs(best) <= 2 and d[best] < 0.0085        # SMARD contexts: max 0.0070 over 1500 draws; a flat series scores ~0.0125

    def _specialist(self, ts: np.ndarray, vals: np.ndarray, horizon: int, fm: int):
        sp = self.specs[fm]
        W, last_ts = _regularize(ts, vals, sp.step_s, sp.L)
        if W is None or not self._is_load_like(W, sp.P) or not self._clock_ok(W, last_ts, sp):
            return None
        if np.isfinite(W[sp.L - sp.W7:]).sum() < 3 * sp.P:      # need at least 3 days inside the weekly window
            return None
        H = sp.H
        pos_ts = last_ts + (np.arange(sp.L + H, dtype=np.int64) - (sp.L - 1)) * sp.step_s
        cal = {k: v[None] for k, v in self.cal.features(pos_ts).items()}
        Wfull = np.concatenate([W, np.full(H, np.nan, np.float32)])[None]
        f = feats.build(sp, Wfull, cal, with_targets=False)
        if not f["ok"][0]:
            return None
        q = (self._run(fm, f)[0].astype(np.float64) + 1.0) * float(f["mu"][0])
        return last_ts, q

    # ------------------------------------------------------------------ public API
    def predict_series(self, items: List[Dict[str, Any]], horizon: int, freq: str):
        """items: [{'ts': str, 'value': float|None}, ...] -> (last_ts_provided, q[horizon, 9])."""
        pairs = []
        for it in items:
            t = parse_ts(it.get("ts"))
            if t is None:
                continue
            v = it.get("value")
            pairs.append((t, float(v) if v is not None and np.isfinite(float(v)) else np.nan))
        if not pairs:
            raise ValueError("History has no parsable timestamps")
        pairs.sort(key=lambda p: p[0])
        ts = np.array([p[0] for p in pairs], dtype=np.int64)
        vals = np.array([p[1] for p in pairs], dtype=np.float64)
        last_provided = int(ts[-1])
        fm = FREQ_MIN.get(freq)
        q = None
        self.last_path = "fallback"
        if fm is not None and 1 <= horizon <= self.specs[fm].H and np.isfinite(vals).any():
            sp = self.specs[fm]
            last_fin = int(ts[np.flatnonzero(np.isfinite(vals))[-1]])
            gap = int(round((last_provided - last_fin) / sp.step_s))
            if gap + horizon <= sp.H:
                try:
                    res = self._specialist(ts, vals, horizon + gap, fm)
                except Exception as e:  # never fail a round because of the specialist
                    logger.warning("specialist failed (%s); using fallback", e)
                    res = None
                if res is not None:
                    q = res[1][gap:gap + horizon]
                    self.last_path = "specialist"
        if q is None:
            grid = _grid_values(ts, vals, freq)
            _, q = fallback.forecast(grid, horizon, freq)
        q = np.nan_to_num(np.sort(np.asarray(q, dtype=np.float64), axis=1),
                          nan=float(vals[np.isfinite(vals)][-1]) if np.isfinite(vals).any() else 0.0)
        return last_provided, q

    def predict(self, history, horizon: int, freq: str) -> Dict[str, Any]:
        is_batch = isinstance(history[0], list)
        series = history if is_batch else [history]
        out = []
        for s in series:
            last_ts, q = self.predict_series(s, horizon, freq)
            q = np.sort(q, axis=1)
            out.append({"ts": future_timestamps(last_ts, horizon, freq), "forecasts": q[:, 4].tolist(), "path": self.last_path,
                        "quantiles": {lv: q[:, i].tolist() for i, lv in enumerate(LEVELS)}})
        return {"series": out, "batch": is_batch}


def _regularize(ts, vals, step_s, L):
    ts = np.asarray(ts, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float64)
    fin = np.isfinite(vals)
    if not fin.any():
        return None, None
    last_idx = np.flatnonzero(fin)[-1]
    ts, vals = ts[: last_idx + 1], vals[: last_idx + 1]
    last_ts = int(ts[-1])
    k = np.rint((ts - last_ts) / step_s).astype(np.int64)
    W = np.full(L, np.nan, dtype=np.float32)
    keep = k >= -(L - 1)
    W[(L - 1) + k[keep]] = vals[keep]
    return W, last_ts


def _grid_values(ts, vals, freq):
    """Regular grid by timestamp when the frequency is a fixed step; raw order otherwise."""
    step = FREQ_SECONDS.get(freq)
    if step is None or len(ts) < 2:
        return vals
    k = np.rint((ts - ts[-1]) / step).astype(np.int64)
    n = int(min(-k.min() + 1, 5000))
    W = np.full(n, np.nan)
    keep = k >= -(n - 1)
    W[(n - 1) + k[keep]] = vals[keep]
    return W
