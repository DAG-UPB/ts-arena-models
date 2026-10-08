import logging

logger = logging.getLogger(__name__)

import os
import warnings
from typing import Any, Dict, List, Union

import numpy as np
import torch

# Vendored from github.com/foundation-model-research/Kairos@df2618f (Apache-2.0), unmodified.
from .kairos import KairosModel as KairosNetwork

# Horizons beyond 128 steps roll out on the median by design; upstream warns on every call.
warnings.filterwarnings("ignore", message="Prediction length .* is greater than")


class KairosModel:
    """
    Kairos forecasting model wrapper (Kairos_10m, Kairos_23m, Kairos_50m).

    The native output grid is the nine deciles, so nothing is interpolated; the point
    forecast is the native median.
    """

    QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(self) -> None:
        logger.info("Initializing Kairos model...")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model_id = os.getenv("MODEL_ID", "mldi-lab/Kairos_50m")
        # fp32: proteus' Turing GPUs have no native bf16.
        self.model = KairosNetwork.from_pretrained(model_id).to(self.device).eval()

        native_levels = [round(float(q), 2) for q in self.model.config.quantiles]
        if native_levels != self.QUANTILE_LEVELS:
            raise RuntimeError(f"Unexpected Kairos quantile levels: {native_levels}")

        # Series per forward pass; bounds GPU memory on large rounds.
        self.batch_size = int(os.getenv("KAIROS_BATCH_SIZE", "64"))
        # Longest context fed to the model; older points are dropped. Defaults to the
        # training window (2048).
        self.max_context = int(os.getenv("KAIROS_MAX_CONTEXT", str(self.model.config.context_length)))

        logger.info(
            f"Kairos initialized (model={model_id}, device={self.device}, "
            f"max_context={self.max_context})"
        )

    def predict(
        self,
        data: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        horizon: int,
        freq: str = "h",
    ) -> Dict[str, Any]:
        """
        Forecast one series ([{"ts", "value"}, ...]) or a batch (list of such lists).

        Returns {'forecasts': ..., 'quantiles': ...}, per series in the batch case.
        """
        if not data:
            return {'forecasts': [], 'quantiles': {}}

        is_batch = isinstance(data[0], list)
        series_list = data if is_batch else [data]

        all_forecasts: List[List[float]] = []
        all_quantiles: List[Dict[str, List[float]]] = []
        for start in range(0, len(series_list), self.batch_size):
            chunk = series_list[start:start + self.batch_size]
            forecasts, quantiles = self._predict_chunk(chunk, horizon)
            all_forecasts.extend(forecasts)
            all_quantiles.extend(quantiles)

        if not is_batch:
            return {'forecasts': all_forecasts[0], 'quantiles': all_quantiles[0]}
        return {'forecasts': all_forecasts, 'quantiles': all_quantiles}

    def _predict_chunk(self, chunk: List[List[Dict[str, Any]]], horizon: int):
        values = [
            np.array(
                [np.nan if item["value"] is None else float(item["value"]) for item in series],
                dtype=np.float32,
            )[-self.max_context:]
            for series in chunk
        ]

        # Left-pad with NaN; the model masks NaN as unobserved.
        width = max(len(v) for v in values)
        context = np.full((len(values), width), np.nan, dtype=np.float32)
        for i, v in enumerate(values):
            context[i, width - len(v):] = v

        with torch.no_grad():
            out = self.model(
                past_target=torch.from_numpy(context).to(self.device),
                prediction_length=horizon,
                generation=True,
                infer_is_positive=True,
                force_flip_invariance=True,
            )
        # [batch, 9, horizon] -> [batch, horizon, 9]; sorted per point so the deciles
        # stay monotone after flip averaging.
        q = np.sort(out["prediction_outputs"].float().cpu().numpy().transpose(0, 2, 1), axis=-1)

        forecasts, quantiles = [], []
        for deciles in q:
            quantiles.append({
                str(level): deciles[:, j].tolist() for j, level in enumerate(self.QUANTILE_LEVELS)
            })
            forecasts.append(deciles[:, self.QUANTILE_LEVELS.index(0.5)].tolist())
        return forecasts, quantiles
