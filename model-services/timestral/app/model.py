import logging

logger = logging.getLogger(__name__)

import math
import os
from typing import Any, Dict, List, Union

import numpy as np
import torch
from timestral import TimestralForecaster


class TimestralModel:
    """
    Timestral forecasting model wrapper (Mistral AI; 5M, 28M, 123M, 209M).

    The native grid (0.01, 0.05, 0.1, 0.15, ..., 0.95, 0.99) holds every decile, and
    the point forecast is the native median. The forecaster's test-time defaults
    (symmetric averaging, non-negative clipping) stay on, as in the published results.
    """

    QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(self) -> None:
        logger.info("Initializing Timestral model...")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model_id = os.getenv("MODEL_ID", "mistralai/Timestral-209M")
        revision = os.getenv("MODEL_REVISION") or None
        self.forecaster = TimestralForecaster.from_pretrained(model_id, device=self.device, revision=revision)

        # Series per forward pass; bounds GPU memory on large rounds.
        self.batch_size = int(os.getenv("TIMESTRAL_BATCH_SIZE", "64"))

        native = list(self.forecaster.quantile_levels)
        self.level_index = [
            next(i for i, n in enumerate(native) if math.isclose(n, level)) for level in self.QUANTILE_LEVELS
        ]

        logger.info(
            f"Timestral initialized (model={model_id}, revision={revision or 'main'}, device={self.device}, "
            f"native_levels={native})"
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
            np.array([np.nan if item["value"] is None else float(item["value"]) for item in series], dtype=np.float32)
            for series in chunk
        ]

        # Right-aligned; the NaN left padding counts as missing.
        context = np.full((len(values), max(len(v) for v in values)), np.nan, dtype=np.float32)
        for i, v in enumerate(values):
            context[i, context.shape[1] - len(v):] = v

        out = self.forecaster.forecast(context, prediction_length=horizon)
        # [batch, horizon, 9]
        q = out.quantiles[:, 0][..., self.level_index].float().cpu().numpy()

        forecasts, quantiles = [], []
        for deciles in q:
            quantiles.append({
                str(level): deciles[:, j].tolist() for j, level in enumerate(self.QUANTILE_LEVELS)
            })
            forecasts.append(deciles[:, self.QUANTILE_LEVELS.index(0.5)].tolist())
        return forecasts, quantiles
