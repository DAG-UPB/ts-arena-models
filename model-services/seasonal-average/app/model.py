from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union

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


class SeasonalAverageModel:
    """Seasonal-average baseline with empirical-residual quantile bands.

    The point forecast for a future step is the seasonal average of the phase
    that step falls into. Residuals are computed in-sample as
    ``series[t] - seasonal_average(phase_of_t)`` over the context window, pooled
    across phases; the same pooled residual distribution is used for every
    horizon step.
    """

    def __init__(self, num_seasons: Optional[int] = None):
        self.num_seasons = num_seasons

    def predict(
        self,
        history: Union[List[float], List[List[float]]],
        horizon: int = 1,
        seasonality: int = 24,
        offset: Union[int, List[int]] = 0,
    ) -> Dict[str, Any]:
        if not history:
            raise ValueError("History must not be empty.")
        if seasonality < 1:
            raise ValueError("Seasonality must be at least 1.")

        is_batch = isinstance(history[0], list)

        if is_batch:
            offsets = offset if isinstance(offset, list) else [offset] * len(history)
            all_forecasts: List[List[float]] = []
            all_quantiles: List[Dict[str, List[float]]] = []
            for idx, series in enumerate(history):
                if not series:
                    raise ValueError("History series must not be empty.")
                current_offset = offsets[idx] if idx < len(offsets) else 0
                f, q = self._predict_single(series, horizon, seasonality, current_offset)
                all_forecasts.append(f)
                all_quantiles.append(q)
            return {"forecasts": all_forecasts, "quantiles": all_quantiles}

        current_offset = offset if isinstance(offset, int) else 0
        forecasts, quantiles = self._predict_single(history, horizon, seasonality, current_offset)
        return {"forecasts": forecasts, "quantiles": quantiles}

    def _predict_single(
        self, series: List[float], horizon: int, seasonality: int, current_offset: int
    ) -> Dict[str, Any]:
        seasonal_averages: List[float] = []
        for p in range(seasonality):
            target_rem = (p - current_offset) % seasonality
            values = series[target_rem::seasonality]
            if self.num_seasons is not None and self.num_seasons > 0:
                values = values[-self.num_seasons:]
            if values:
                avg = sum(values) / len(values)
            else:
                avg = 0.0
            seasonal_averages.append(avg)

        # In-sample residuals: series[t] - seasonal_average(phase_of_t).
        residuals: List[float] = []
        for t in range(len(series)):
            phase = (t + current_offset) % seasonality
            residuals.append(float(series[t]) - float(seasonal_averages[phase]))

        start_pred_phase = (current_offset + len(series)) % seasonality
        point_series = [
            seasonal_averages[(start_pred_phase + h) % seasonality] for h in range(horizon)
        ]
        return _quantile_series(point_series, residuals)