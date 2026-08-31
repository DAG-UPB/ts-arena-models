import os
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

# Root logging is configured once, in app/main.py (ts-arena #15).
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Set float32 matmul precision as recommended for performance
torch.set_float32_matmul_precision("high")

# TimesFM 3.0's own hard cap (timesfm3.timesfm3_forecaster._MAX_CONTEXT_LENGTH).
MAX_CONTEXT = 15360


class TimesFMModel:
    """TimesFM 3.0 wrapper.

    Note the API break versus the 2.5 service: 3.0 lives in the `timesfm3` package
    and is driven through `TimesFM3Evaluator.predict_batch(...)`, not
    `TimesFM_2p5_200M_torch.from_pretrained(...)` + `ForecastConfig` + `forecast(...)`.
    There is also no compile step and no max-horizon to recompile against — the
    forecaster rounds the horizon up to an output-patch boundary internally.

    The quantile axis differs too: 3.0 returns exactly the nine deciles
    `[0.1 ... 0.9]` at indices 0..8, whereas the 2.5 head returns ten columns with
    the mean at index 0. `self.quantile_levels` carries the levels read off the
    loaded checkpoint so `main.py` never has to hard-code that offset.
    """

    def __init__(self) -> None:
        # Imported here rather than at module import time so the logging setup in
        # main.py is already in place when timesfm3 emits its own load messages.
        from timesfm3 import ModelConfig, TimesFM3Evaluator

        self.model_id = os.getenv("MODEL_ID", "google/timesfm-3.0-pytorch")
        logger.info(f"Loading model from {self.model_id}...")

        # Check if we have a local model path mapped
        # Assuming standard mapping /models/<repo_name>
        local_dir = Path(f"/models/{self.model_id.split('/')[-1]}")

        load_path = self.model_id
        if local_dir.exists() and any(local_dir.iterdir()):
            logger.info(f"Found local model at {local_dir}")
            load_path = str(local_dir)
        else:
            logger.info(
                f"Model not found locally at {local_dir}, using {self.model_id} "
                "(will download if not cached)..."
            )

        self.max_context = MAX_CONTEXT
        self.batch_size = int(os.getenv("PER_CORE_BATCH_SIZE", "32"))

        try:
            config = ModelConfig(
                checkpoint_path=load_path,
                per_core_batch_size=self.batch_size,
                device=device,
            )
            self.model = TimesFM3Evaluator(config)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # After loading, the forecaster syncs its config with the checkpoint, so this
        # is the checkpoint's own quantile list, not our guess at it.
        self.quantile_levels: List[float] = [
            float(q) for q in self.model.config.quantiles
        ]
        logger.info(
            f"Model loaded. quantile levels={self.quantile_levels}, "
            f"max_context={self.max_context}, batch_size={self.batch_size}"
        )

    @staticmethod
    def _to_array(series: Sequence[Optional[float]]) -> np.ndarray:
        """Coerce one history to float32, turning missing values into NaN.

        TimesFM 3.0 linearly interpolates NaNs itself and trims a leading run of
        them, so gaps are better handed over as NaN than dropped or zero-filled.
        """
        return np.array(
            [np.nan if v is None else float(v) for v in series], dtype=np.float32
        )

    def predict(
        self,
        history: Union[List[float], List[List[float]]],
        horizon: int,
    ) -> Tuple[list, Optional[list]]:
        """Forecast one series or a batch.

        Returns `(point_forecast, quantile_forecast)`. For a single series that is
        `(horizon,)` and `(horizon, 9)`; for a batch, both gain a leading batch axis.
        """
        if not history:
            raise ValueError("History cannot be empty.")

        is_single_series = not isinstance(history[0], list)
        if is_single_series:
            history = [history]

        contexts = []
        for series in history:
            arr = self._to_array(series)
            # Truncate if too long
            if arr.shape[0] > self.max_context:
                arr = arr[-self.max_context :]
            contexts.append(arr)

        outputs = list(
            self.model.predict_batch(
                contexts=contexts,
                horizon=horizon,
                return_quantiles=True,
            )
        )

        point_forecast = [np.asarray(o.forecast).tolist() for o in outputs]
        if all(o.quantiles is not None for o in outputs):
            quantile_forecast = [np.asarray(o.quantiles).tolist() for o in outputs]
        else:
            # Never fabricate quantiles (models #13) — drop them for the whole batch.
            logger.warning("Model returned no quantiles; falling back to point-only.")
            quantile_forecast = None

        if is_single_series:
            return point_forecast[0], quantile_forecast[0] if quantile_forecast else None
        return point_forecast, quantile_forecast
