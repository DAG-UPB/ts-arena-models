"""Feature construction shared by training and the service (numpy in, numpy out).

A sample = one series at one origin.  Window array Wfull[B, L+H]: columns 0..L-1 are the served context
(oldest first, last column = origin value, NaN = missing), columns L..L+H-1 are NaN at inference (targets in training).
cal[...] arrays of shape [B, L+H] give calendar features at every column (computed from timestamps).
"""
import numpy as np

class Spec:
    def __init__(self, freq_minutes, L=1000, J=48, dmeans=0):
        assert freq_minutes in (15, 60)
        self.freq = freq_minutes
        self.step_s = freq_minutes * 60
        self.P = 96 if freq_minutes == 15 else 24      # steps per day
        self.H = 96 if freq_minutes == 15 else 72
        self.L = L
        self.J = J                                      # recent window length (steps)
        self.dmeans = dmeans                            # number of per-day mean levels added to the context vector
        self.nd = 10                                    # consecutive daily lags
        self.extra = [] if freq_minutes == 15 else [14, 21, 28]
        s = np.arange(1, self.H + 1)
        dmin = np.ceil(s / self.P).astype(int)
        lag_days = np.stack([dmin + k for k in range(self.nd)], axis=1)
        if self.extra:
            lag_days = np.concatenate([lag_days, np.tile(np.array(self.extra), (self.H, 1))], axis=1)
        self.lag_days = lag_days                         # [H,K]
        self.K = lag_days.shape[1]
        self.lag_off = (s[:, None] - self.P * lag_days)  # position = (L-1) + lag_off
        self.lag_pos = (self.L - 1) + self.lag_off
        self.lag_valid = self.lag_pos >= 0
        self.nblocks = int(np.ceil(self.H / self.P))
        self.W7 = 7 * self.P
        assert self.L - self.J - self.W7 >= 0
        self.n_ctx = 2 * self.J + 4 + self.dmeans
        self.n_lag_f = 7
        self.n_step_f = 4 + 7 + 5 + 2 + 1 + (self.nblocks if self.nblocks > 1 else 0)

def fill_gaps(W, max_gap=96):
    """Linear interpolation of NaN holes (<= max_gap) inside each row; leading/trailing NaN: nearest valid."""
    W = W.copy()
    B, T = W.shape
    nanmask = np.isnan(W)
    rows = np.flatnonzero(nanmask.any(1))
    x = np.arange(T)
    for b in rows:
        v = ~nanmask[b]
        if v.sum() == 0:
            continue
        W[b] = np.interp(x, x[v], W[b, v])
        # do not bridge very long gaps: restore NaN where the nearest valid neighbours are > max_gap apart
        idx = np.flatnonzero(v)
        gaps = np.flatnonzero(np.diff(idx) > max_gap)
        for g in gaps:
            W[b, idx[g] + 1: idx[g + 1]] = np.nan
    return W

def build(spec, Wfull, cal, with_targets=False):
    """-> dict of float32 arrays (+ 'mu', 'base', optional 'y', 'naive', 'w')."""
    L, H, J, K, P = spec.L, spec.H, spec.J, spec.K, spec.P
    B = Wfull.shape[0]
    Wc = Wfull[:, :L].astype(np.float32)
    Wf = fill_gaps(Wc)
    ok_origin = np.isfinite(Wf[:, L - 1])
    tail = Wf[:, L - spec.W7:]
    mu = np.nanmean(np.where(np.isfinite(tail), tail, np.nan), axis=1)
    mu = np.where(np.isfinite(mu) & (mu > 0), mu, np.nan).astype(np.float32)
    muc = np.where(np.isnan(mu), 1.0, mu)[:, None]
    xr = Wf[:, L - J:L] / muc
    x7 = Wf[:, L - J - spec.W7: L - spec.W7] / muc
    last_day = np.nanmean(Wf[:, L - P:], axis=1, keepdims=True) / muc
    prev_day = np.nanmean(Wf[:, L - 2 * P: L - P], axis=1, keepdims=True) / muc
    hr_ok = np.isfinite(xr).mean(1, keepdims=True)
    parts = [np.nan_to_num(xr - 1.0), np.nan_to_num(x7 - 1.0), np.nan_to_num(last_day - 1.0), np.nan_to_num(prev_day - 1.0),
             np.nan_to_num(xr[:, -1:] - x7[:, -1:]), hr_ok]
    if spec.dmeans:
        dm = np.stack([np.nanmean(Wf[:, L - (k + 1) * P: L - k * P], axis=1) for k in range(spec.dmeans)], axis=1) / muc
        parts.append(np.nan_to_num(dm - 1.0))
    ctx = np.concatenate(parts, axis=1).astype(np.float32)
    assert ctx.shape[1] == spec.n_ctx, (ctx.shape, spec.n_ctx)
    # lags
    lag_pos = spec.lag_pos.reshape(-1)
    lv = np.where(spec.lag_valid.reshape(-1)[None, :], Wf[:, np.clip(lag_pos, 0, L - 1)], np.nan) / muc   # [B, H*K]
    lmask = np.isfinite(lv)
    lv = np.nan_to_num(lv - 1.0).reshape(B, H, K)
    lmask_f = lmask.reshape(B, H, K).astype(np.float32)
    lh = cal["hol"][:, np.clip(lag_pos, 0, L + H - 1)]             # [B, H*K, 6]
    lh = lh[..., [0, 1, 3, 4, 5]].reshape(B, H, K, 5)
    lag_f = np.concatenate([lv[..., None], lmask_f[..., None], lh], axis=-1).astype(np.float32)   # [B,H,K,7]
    # base: masked mean of lag values (in normalized-1 units)
    cnt = lmask_f.sum(-1)
    base = np.where(cnt > 0, (lv * lmask_f).sum(-1) / np.maximum(cnt, 1), 0.0).astype(np.float32)   # [B,H] (value-1)
    # step features at target columns L..L+H-1
    sl = slice(L, L + H)
    hour = cal["hour"][:, sl]; dow = cal["dow"][:, sl]; holt = cal["hol"][:, sl][..., [0, 1, 3, 4, 5]]; doy = cal["doy"][:, sl]
    ang = 2 * np.pi * hour / 24.0
    f_hour = np.stack([np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang)], -1)
    f_dow = np.eye(7, dtype=np.float32)[dow.astype(np.int64)]
    angy = 2 * np.pi * doy / 366.0
    f_doy = np.stack([np.sin(angy), np.cos(angy)], -1)
    s = (np.arange(1, H + 1, dtype=np.float32) / H)[None, :, None].repeat(B, 0)
    parts = [f_hour, f_dow, holt, f_doy, s]
    if spec.nblocks > 1:
        blk = ((np.arange(H) // P))
        parts.append(np.eye(spec.nblocks, dtype=np.float32)[blk][None].repeat(B, 0))
    step_f = np.concatenate(parts, -1).astype(np.float32)
    out = dict(ctx=ctx, lag=lag_f, step=step_f, base=base, mu=mu, ok=ok_origin & np.isfinite(mu))
    if with_targets:
        y = Wfull[:, L:L + H].astype(np.float32) / muc
        naive = (Wf[:, L - 1:L] / muc)
        mae_naive = np.nanmean(np.abs(y - naive), axis=1)
        out.update(y=np.nan_to_num(y - 1.0), ymask=np.isfinite(y).astype(np.float32), naive=(naive - 1.0).astype(np.float32), mae_naive=mae_naive.astype(np.float32))
    return out
