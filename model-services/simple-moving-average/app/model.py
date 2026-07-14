from __future__ import annotations

import math
from typing import Any, Dict, List, Union

# Standard 9 decile levels used across TS-Arena model services.
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _percentile(data: List[float], q: float) -> float:
    """Linear-interpolation percentile, matching numpy's default method.

    ``q`` is a probability in [0, 1]. An empty ``data`` collapses to 0.0
    (degenerate: all bands equal the point forecast).
    """
    if not data:
        return 0.0
    s = sorted(float(v) for v in data)
    n = len(s)
    if n == 1:
        return s[0]
    rank = q * (n - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return s[lo]
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _quantile_series(
    point_series: List[float], residuals: List[float]
) -> Dict[str, Any]:
    """Build monotone quantile bands around a point forecast.

    ``point_series`` is the per-step point forecast (length ``horizon``);
    ``residuals`` are the in-sample residuals of the baseline. For each level
    ``l`` in ``QUANTILE_LEVELS`` the quantile band is
    ``point_series + quantile(residuals, l)``. The 9 bands are then sorted
    ascending per step and re-assigned to levels 0.1..0.9, which guarantees
    ``q_0.1 <= … <= q_0.9`` by construction. The point forecast returned is
    the ``q_0.5`` band (median consistency).

    Returns ``(forecasts, quantiles)`` where ``forecasts`` is the ``q_0.5``
    series and ``quantiles`` maps level strings ("0.1".."0.9") to a per-step
    list of length ``horizon``.
    """
    levels = [str(l) for l in QUANTILE_LEVELS]
    if not residuals:
        # Degenerate case (e.g. single-point context): no empirical residual
        # information, so all quantiles collapse onto the point forecast.
        residuals = [0.0]
    pt = [float(v) for v in point_series]
    q_vals = {l: [p + _percentile(residuals, float(l)) for p in pt] for l in levels}
    # Enforce monotone non-decreasing quantiles per step: for each horizon
    # step sort the 9 band values ascending and re-assign to levels 0.1..0.9.
    horizon = len(pt)
    quantiles: Dict[str, List[float]] = {l: [0.0] * horizon for l in levels}
    for h in range(horizon):
        step_vals = sorted(q_vals[l][h] for l in levels)
        for i, l in enumerate(levels):
            quantiles[l][h] = step_vals[i]
    forecasts = quantiles["0.5"]
    return forecasts, quantiles


class SimpleMovingAverageModel:
    """Simple moving-average baseline with empirical-residual quantile bands.

    The point forecast is the sample mean of the context window, used for every
    horizon step. Residuals are computed in-sample as
    ``series[t] - mean(series[t-w:t])`` over the context window using a rolling
    window ``w`` (default 3); the same pooled residual distribution is used for
    every horizon step. If the context is shorter than the window, all
    available prior observations are used.
    """

    DEFAULT_WINDOW = 3

    def predict(
        self, history: Union[List[float], List[List[float]]], horizon: int = 1
    ) -> Dict[str, Any]:
        if not history:
            raise ValueError("History must not be empty.")

        is_batch = isinstance(history[0], list)

        if is_batch:
            all_forecasts: List[List[float]] = []
            all_quantiles: List[Dict[str, List[float]]] = []
            for series in history:
                if not series:
                    raise ValueError("History series must not be empty.")
                f, q = self._predict_single(series, horizon)
                all_forecasts.append(f)
                all_quantiles.append(q)
            return {"forecasts": all_forecasts, "quantiles": all_quantiles}

        forecasts, quantiles = self._predict_single(history, horizon)
        return {"forecasts": forecasts, "quantiles": quantiles}

    def _predict_single(
        self, series: List[float], horizon: int
    ) -> Dict[str, Any]:
        mean_value = float(sum(series) / len(series))
        point_series = [mean_value] * horizon
        window = self.DEFAULT_WINDOW
        residuals: List[float] = []
        for t in range(1, len(series)):
            start = max(0, t - window)
            window_vals = series[start:t]
            if not window_vals:
                continue
            residuals.append(float(series[t]) - float(sum(window_vals) / len(window_vals)))
        return _quantile_series(point_series, residuals)