import os
from collections import defaultdict
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel

HF_REPO = os.environ.get("MODEL_ID", "paris-noah/Tabby")
MAX_CONTEXT = int(os.environ.get("TABBY_MAX_CONTEXT", "8192"))
BATCH_SIZE = int(os.environ.get("TABBY_BATCH_SIZE", "32"))
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _clean(series):
    s = pd.Series([x.get("value") for x in series], dtype="float64")
    s = s.ffill().bfill().fillna(0.0)
    return s.to_numpy(dtype=np.float32)[-MAX_CONTEXT:]


class TabbyModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(HF_REPO, trust_remote_code=True)
        self.model = self.model.to(self.device).eval()
        print(f"Tabby loaded on {self.device}", flush=True)

    @torch.no_grad()
    def _run(self, arrays, horizon):
        x = torch.from_numpy(np.stack(arrays)).to(self.device)
        q = self.model.predict(x, prediction_length=horizon)
        return q.float().cpu().numpy()[:, :, :horizon]

    def predict(self, history, horizon, freq="h", quantile_levels=LEVELS):
        is_batch = isinstance(history[0], list)
        batch = history if is_batch else [history]
        contexts = [_clean(s) for s in batch]
        groups = defaultdict(list)
        for i, c in enumerate(contexts):
            groups[len(c)].append(i)
        outputs = [None] * len(contexts)
        for idxs in groups.values():
            for k in range(0, len(idxs), BATCH_SIZE):
                chunk = idxs[k:k + BATCH_SIZE]
                out = self._run([contexts[i] for i in chunk], horizon)
                for j, i in enumerate(chunk):
                    outputs[i] = out[j]
        forecasts, quantiles = [], {}
        for i, q in enumerate(outputs):
            grid = np.arange(1, q.shape[0] + 1) / (q.shape[0] + 1)
            pick = lambda lv: int(np.argmin(np.abs(grid - lv)))
            forecasts.append(q[pick(0.5)].tolist())
            quantiles[i] = {f"q_{lv}": q[pick(lv)].tolist() for lv in LEVELS}
        if is_batch:
            return {"forecasts": forecasts, "quantiles": quantiles}
        return {"forecasts": forecasts[0], "quantiles": quantiles[0]}
