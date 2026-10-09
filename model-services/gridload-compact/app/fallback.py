"""Generic seasonal fallback with residual-based deciles.  Used whenever the grid-load specialist does not apply
(other frequencies, other horizons, non load-like series, short or degenerate contexts).  NaN-safe, never raises on content."""
import numpy as np

LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
SEASON = {"1min": 60, "15min": 96, "30min": 48, "h": 24, "D": 7, "W": 52, "M": 12}

def forecast(values, horizon, freq, num_seasons=1):
    """values: 1-D float array, NaN allowed.  Returns (median [H], q [H,9])."""
    v = np.asarray(values, dtype=np.float64)
    fin = np.isfinite(v)
    if not fin.any():
        z = np.zeros(horizon)
        return z, np.tile(z[:, None], (1, 9))
    # carry the last finite value as the series end (platform anchors on the last timestamp)
    last = v[np.flatnonzero(fin)[-1]]
    P = SEASON.get(freq, 24)
    n = len(v)
    vv = np.where(fin, v, np.nan)
    if n < 2 * P:
        # too short for seasonal structure: persistence with residuals of first differences
        d = np.diff(vv[fin])
        spread = np.nanstd(d) if len(d) > 1 else 0.0
        sc = spread * np.sqrt(np.arange(1, horizon + 1))[:, None]
        q = last + sc * np.array([-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816])[None, :]
        q = np.sort(q, axis=1)
        return q[:, 4].copy(), q
    # seasonal profile = median of the last num_seasons values at the same phase (1 season = the platform's seasonal-naive rule)
    phase_idx = (np.arange(horizon) + 1) % P
    prof = np.full(horizon, last)
    S = []
    for h in range(horizon):
        # step h+1 after the end has index n+h; the same phase inside the history is n+h - P*k with k >= h//P + 1
        k0 = h // P + 1
        idx = n + h - P * (k0 + np.arange(num_seasons))
        idx = idx[(idx >= 0) & (idx < n)]
        vals = vv[idx]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            prof[h] = np.median(vals)
    # in-sample residuals of the same rule over the last 3 seasons
    res = []
    for t in range(max(P * 2, n - 3 * P), n):
        idx = t - P * np.arange(1, num_seasons + 1)
        idx = idx[idx >= 0]
        vals = vv[idx]; vals = vals[np.isfinite(vals)]
        if len(vals) and np.isfinite(vv[t]):
            res.append(vv[t] - np.median(vals))
    res = np.asarray(res) if len(res) else np.zeros(1)
    rq = np.quantile(res, LEVELS)
    rq = rq - rq[4]                      # centre so that the median equals the seasonal median
    q = prof[:, None] + rq[None, :]
    q = np.sort(q, axis=1)
    return q[:, 4].copy(), q
