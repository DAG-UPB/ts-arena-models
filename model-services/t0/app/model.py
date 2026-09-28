import logging

logger = logging.getLogger(__name__)

import os
from typing import Any, Dict, List, Union

import numpy as np
import torch
from t0 import T0Forecaster, batch_series


class T0Model:
    """
    t0 forecasting model wrapper (The Forecasting Company; t0-alpha, t0-beta).

    t0-alpha's native quantile levels are 0.1, 0.25, 0.5, 0.75 and 0.9, and the
    package interpolates the other deciles between them. t0-beta's native grid
    (0.01..0.99 in steps of 0.05) holds every decile. Either way the point forecast
    is the native median.
    """

    QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(self) -> None:
        logger.info("Initializing t0 model...")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model_id = os.getenv("MODEL_ID", "theforecastingcompany/t0-alpha")
        # fp32: the package only autocasts for bf16/fp16, and proteus' Turing GPUs
        # have no native bf16.
        self.model = T0Forecaster.from_pretrained(model_id).to(self.device).eval()

        # Series per forward pass; bounds GPU memory on large rounds.
        self.batch_size = int(os.getenv("T0_BATCH_SIZE", "64"))
        # Longest context fed to the model; older points are dropped. 8192 is the
        # training window and what the paper's benchmarks use.
        self.max_context = int(os.getenv("T0_MAX_CONTEXT", "8192"))

        logger.info(
            f"t0 initialized (model={model_id}, device={self.device}, "
            f"native_levels={list(self.model.config.quantile_levels)})"
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
        values = []
        for series in chunk:
            v = np.array(
                [np.nan if item["value"] is None else float(item["value"]) for item in series],
                dtype=np.float32,
            )
            values.append(v[-self.max_context:])

        # Right-aligns the series: the left padding is marked PAD, NaN as MISSING.
        context, mask, group_ids = batch_series(values)
        out = self.model.predict(
            context,
            horizon=horizon,
            quantile_levels=self.QUANTILE_LEVELS,
            mask=mask,
            group_ids=group_ids,
        )
        # [batch, horizon, 9]; sorted per point so the deciles are monotone even if
        # the interpolated levels cross.
        q = np.sort(out.quantiles.float().cpu().numpy(), axis=-1)

        forecasts, quantiles = [], []
        for deciles in q:
            quantiles.append({
                str(level): deciles[:, j].tolist() for j, level in enumerate(self.QUANTILE_LEVELS)
            })
            forecasts.append(deciles[:, self.QUANTILE_LEVELS.index(0.5)].tolist())
        return forecasts, quantiles
