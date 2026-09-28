import logging

logger = logging.getLogger(__name__)

import math
import os
from typing import Any, Dict, List, Union

import numpy as np
import torch
from toto2 import Toto2Model as _Toto2


class Toto2Model:
    """
    Toto 2.0 forecasting model wrapper.

    Toto 2.0 predicts quantiles at the fixed knot positions of its output head. The
    arena contract wants deciles, so the knots are interpolated onto 0.1..0.9 and the
    point forecast is the interpolated median.
    """

    QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(self) -> None:
        logger.info("Initializing Toto 2.0 model...")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model_id = os.getenv("MODEL_ID", "Datadog/Toto-2.0-22m")
        self.model = _Toto2.from_pretrained(model_id).to(self.device).eval()
        self.patch_size = self.model.config.patch_size
        self.knots = np.asarray(self.model.output_head.knots, dtype=np.float64)

        # Series per forward pass; bounds GPU memory on large rounds.
        self.batch_size = int(os.getenv("TOTO2_BATCH_SIZE", "64"))
        # Longest context fed to the model; older points are dropped.
        self.max_context = int(os.getenv("TOTO2_MAX_CONTEXT", "4096"))
        # 0 = single forward pass, the upstream recommendation for short horizons.
        self.decode_block_size = int(os.getenv("TOTO2_DECODE_BLOCK_SIZE", "0"))

        logger.info(
            f"Toto 2.0 initialized (model={model_id}, device={self.device}, "
            f"patch_size={self.patch_size}, knots={self.knots.tolist()})"
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

        # Left-pad every series to one length that is a whole number of patches;
        # padding and missing values are masked out.
        length = max(len(v) for v in values)
        length = math.ceil(length / self.patch_size) * self.patch_size
        target = np.zeros((len(values), 1, length), dtype=np.float32)
        mask = np.zeros((len(values), 1, length), dtype=bool)
        for i, v in enumerate(values):
            observed = np.isfinite(v)
            target[i, 0, length - len(v):] = np.where(observed, v, 0.0)
            mask[i, 0, length - len(v):] = observed

        inputs = {
            "target": torch.from_numpy(target).to(self.device),
            "target_mask": torch.from_numpy(mask).to(self.device),
            "series_ids": torch.zeros(len(values), 1, dtype=torch.long, device=self.device),
        }
        with torch.inference_mode():
            q = self.model.forecast(
                inputs,
                horizon=horizon,
                decode_block_size=self.decode_block_size,
                # Flash attention is only valid without gaps or padding.
                has_missing_values=not bool(mask.all()),
            )
        # [n_knots, batch, 1, horizon] -> [batch, horizon, n_knots], sorted per point
        # so the interpolated deciles are monotone even if the knots cross.
        q = np.sort(q[:, :, 0, :].float().cpu().numpy().transpose(1, 2, 0), axis=-1)

        forecasts, quantiles = [], []
        for per_series in q:
            deciles = np.array([
                np.interp(self.QUANTILE_LEVELS, self.knots, point) for point in per_series
            ])  # [horizon, 9]
            quantiles.append({
                str(level): deciles[:, j].tolist() for j, level in enumerate(self.QUANTILE_LEVELS)
            })
            forecasts.append(deciles[:, self.QUANTILE_LEVELS.index(0.5)].tolist())
        return forecasts, quantiles
