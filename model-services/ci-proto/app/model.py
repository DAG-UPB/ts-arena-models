import logging

logger = logging.getLogger(__name__)

import os
from typing import Any, Dict, List, Union

import continuousintelligence as ci
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download


class CIProtoModel:
    """
    ci-proto forecasting model wrapper (Continuous Intelligence).

    All series of one request are forecast in one call, as one cohort. Series whose
    histories end at the same time are forecast together, and the controller's batch
    holds a whole round. Each series' timestamps are passed with its values, for
    ci-proto's calendar features.
    """

    QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Longest history fed to the model; older steps are dropped. The model card gives
    # 16,384 steps as the longest history for model.ckpt.
    MAX_CONTEXT = 16384

    def __init__(self) -> None:
        logger.info("Initializing ci-proto model...")

        model_id = os.getenv("MODEL_ID", "ContinuousIntelligence/ci-proto")
        revision = os.getenv("MODEL_REVISION") or None
        # The checkpoint and its tta.json settings, the only files this service uses.
        paths = {
            name: hf_hub_download(model_id, name, revision=revision)
            for name in ("model.ckpt", "tta.json")
        }
        # "auto" runs on the GPU when PyTorch can use CUDA, otherwise on the CPU. The
        # package picks its fused Mamba-3 kernels only on Ampere or newer GPUs with a
        # 580-series driver, and its portable PyTorch backend everywhere else.
        self.model = ci.load_model(paths["model.ckpt"], paths["tta.json"], device="auto")

        levels = self.model.quantiles.tolist()
        if len(levels) != len(self.QUANTILE_LEVELS) or any(
            abs(level - wanted) > 1e-6 for level, wanted in zip(levels, self.QUANTILE_LEVELS)
        ):
            raise ValueError(f"ci-proto returns the quantile levels {levels}, not the deciles")

        logger.info(
            f"ci-proto initialized (model={model_id}, revision={revision or 'main'}, "
            f"device={self.model.device}, backend={self.model.backend})"
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
        step = pd.tseries.frequencies.to_offset(freq).nanos

        # Each series goes on a regular grid at freq that ends at its last timestamp, with
        # a NaN step for every absent timestamp. The package lines up the series it
        # forecasts together by position, so the grid keeps every value at its time. The
        # grid keeps the last MAX_CONTEXT steps.
        grids = []
        for series in (data if is_batch else [data]):
            index = pd.to_datetime(
                [item["ts"] for item in series], utc=True, format="ISO8601"
            ).as_unit("ns")
            observed = pd.Series(
                [np.nan if item["value"] is None else float(item["value"]) for item in series],
                index=index,
                dtype=np.float32,
            )
            steps = int((index.asi8.max() - index.asi8.min()) // step) + 1
            grid = pd.date_range(
                end=index.max(), periods=min(steps, self.MAX_CONTEXT), freq=freq, unit="ns"
            )
            grids.append((observed.reindex(grid).to_numpy(), grid.asi8))

        # Shorter series are padded at the front with NaN values and int64's minimum
        # timestamp, the package's marker for padding. Every series' future steps
        # continue from its own last timestamp, as the platform scores them.
        length = max(len(values) for values, _ in grids)
        history = np.full((len(grids), length), np.nan, dtype=np.float32)
        timestamps = np.full(
            (len(grids), length + horizon), np.iinfo(np.int64).min, dtype=np.int64
        )
        for row, (values, stamps) in enumerate(grids):
            start = length - len(values)
            history[row, start:] = values
            timestamps[row, start:length] = stamps
            timestamps[row, length:] = stamps[-1] + step * np.arange(1, horizon + 1)

        out = self.model.forecast(
            torch.from_numpy(history),
            horizon,
            timestamps=torch.from_numpy(timestamps),
            freq=freq,
            series_ids=[f"{row:05d}" for row in range(len(grids))],
        )
        # [series, 9, horizon], on the CPU like the input. The deciles come in order,
        # and the package's calibration sorts them at every step.
        q = out.numpy()

        forecasts, quantiles = [], []
        for deciles in q:
            quantiles.append({
                str(level): deciles[j].tolist() for j, level in enumerate(self.QUANTILE_LEVELS)
            })
            forecasts.append(deciles[self.QUANTILE_LEVELS.index(0.5)].tolist())

        if not is_batch:
            return {'forecasts': forecasts[0], 'quantiles': quantiles[0]}
        return {'forecasts': forecasts, 'quantiles': quantiles}
