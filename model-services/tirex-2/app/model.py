import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tirex2 import ForecastModel, TimeseriesType, load_model

def _to_float_list(values: List[Optional[float]]) -> List[float]:
    """Convert a list with possible None entries to floats, None -> NaN (masked by TiRex-2)."""
    return [float("nan") if v is None else float(v) for v in values]


class TiRex2Model:
    def __init__(self) -> None:
        """
        Initializes the TiRex-2 model from HuggingFace.
        """
        model_id = os.getenv("MODEL_ID", "NX-AI/TiRex-2")
        device = os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading TiRex-2 model from {model_id} (device: {device})...")
        self.model: ForecastModel = load_model(model_id, device=device)
        if device.startswith("cuda"):
            # The fused flashrnn sLSTM CUDA kernel requires compute capability >= 8.0
            # (Ampere). Verify it actually runs; otherwise fall back to CPU.
            try:
                smoke = TimeseriesType(
                    target=torch.linspace(0.0, 6.0, 128).sin().unsqueeze(0),
                    past_covariates=None,
                    future_covariates=None,
                )
                with torch.no_grad():
                    self.model.forecast([smoke], prediction_length=1, output_type="numpy")
            except Exception as e:
                print(f"CUDA inference not usable on this GPU ({e}); falling back to CPU")
                self.model = load_model(model_id, device="cpu")
        # Quantile levels natively forecast by the checkpoint (e.g. 0.1 ... 0.9)
        self.quantile_levels: List[float] = [round(float(q), 6) for q in self.model.quantiles]
        self.median_idx = min(
            range(len(self.quantile_levels)),
            key=lambda i: abs(self.quantile_levels[i] - 0.5),
        )
        print(f"TiRex-2 model loaded successfully (quantiles: {self.quantile_levels})")

    def predict(
        self,
        data: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        horizon: int,
        freq: str = "h",
        covariates: Union[None, Dict[str, Any], List[Optional[Dict[str, Any]]]] = None,
        multivariate: bool = False,
    ) -> Dict[str, Any]:
        """
        Makes a forecast using TiRex-2.

        Args:
            data: Either a single time series as a list of dicts [{"ts": <timestamp>, "value": <float>}, ...]
                  or a batch of time series as a list of such lists.
            horizon: Number of steps to forecast
            freq: Frequency string (not used by TiRex-2 but kept for API compatibility)
            covariates: Optional covariates, mirroring the shape of `data`:
                        a single dict for a single series, or a list (entries may be None)
                        parallel to the batch. Each dict has the form
                        {"past": {name: [T floats]}, "future": {name: [T+H or H floats]}}.
                        With multivariate=True a single dict applies to the joint series.
            multivariate: If True and `data` is a batch, all series are treated as channels
                          of one multivariate series and forecast jointly.

        Returns:
            Dictionary with 'forecasts' and 'quantiles'
        """
        if not data:
            return {'forecasts': [], 'quantiles': {}}

        # Detect single-series vs. batch input
        is_batch = isinstance(data[0], list)
        data_as_batch = data if is_batch else [data]

        multivariate = bool(multivariate) and is_batch and len(data_as_batch) > 1

        if multivariate:
            timeseries = [self._build_multivariate_timeseries(data_as_batch, covariates, horizon)]
        else:
            cov_list = self._normalize_covariates(covariates, len(data_as_batch), is_batch)
            timeseries = [
                self._build_univariate_timeseries(series, cov, horizon, idx)
                for idx, (series, cov) in enumerate(zip(data_as_batch, cov_list))
            ]

        with torch.no_grad():
            forecasts = self.model.forecast(
                timeseries,
                prediction_length=horizon,
                output_type="numpy",
            )

        # Each forecast has shape [n_targets, n_quantiles, forecast_len]
        if multivariate:
            channel_arrays = [forecasts[0][v] for v in range(forecasts[0].shape[0])]
        else:
            channel_arrays = [f[0] for f in forecasts]

        results = []
        quantiles_results = []
        for arr in channel_arrays:
            arr = self._extend_to_horizon(arr, horizon)
            quantile_dict = {
                str(level): arr[qi].tolist() for qi, level in enumerate(self.quantile_levels)
            }
            quantiles_results.append(quantile_dict)
            # Use the median as point forecast for consistency
            results.append(arr[self.median_idx].tolist())

        if not is_batch:
            return {
                "forecasts": results[0],
                "quantiles": quantiles_results[0]
            }

        return {
            "forecasts": results,
            "quantiles": quantiles_results
        }

    @staticmethod
    def _normalize_covariates(
        covariates: Union[None, Dict[str, Any], List[Optional[Dict[str, Any]]]],
        num_series: int,
        is_batch: bool,
    ) -> List[Optional[Dict[str, Any]]]:
        """Align covariates to one (optionally None) dict per series."""
        if covariates is None:
            return [None] * num_series
        if isinstance(covariates, dict):
            if is_batch and num_series > 1:
                raise ValueError("For batch input pass covariates as a list parallel to the series")
            return [covariates] * num_series
        if len(covariates) != num_series:
            raise ValueError(
                f"Got covariates for {len(covariates)} series but {num_series} series in history"
            )
        return list(covariates)

    def _build_univariate_timeseries(
        self,
        series: List[Dict[str, Any]],
        covariates: Optional[Dict[str, Any]],
        horizon: int,
        series_idx: int,
    ) -> TimeseriesType:
        target = torch.tensor(
            _to_float_list([item["value"] for item in series]), dtype=torch.float32
        ).unsqueeze(0)  # [1, T]
        past_cov, future_cov = self._build_covariate_tensors(
            covariates, target.shape[-1], horizon, series_idx
        )
        return TimeseriesType(target=target, past_covariates=past_cov, future_covariates=future_cov)

    def _build_multivariate_timeseries(
        self,
        data_as_batch: List[List[Dict[str, Any]]],
        covariates: Union[None, Dict[str, Any], List[Optional[Dict[str, Any]]]],
        horizon: int,
    ) -> TimeseriesType:
        """Stack all series of the batch as channels of one joint multivariate series."""
        channels = [_to_float_list([item["value"] for item in series]) for series in data_as_batch]
        max_len = max(len(c) for c in channels)
        # Left-pad shorter channels with NaN (masked by the model) so all channels align
        padded = [[float("nan")] * (max_len - len(c)) + c for c in channels]
        target = torch.tensor(padded, dtype=torch.float32)  # [V, T]

        if isinstance(covariates, list):
            raise ValueError(
                "With multivariate=true pass a single covariates object for the joint series"
            )
        past_cov, future_cov = self._build_covariate_tensors(covariates, max_len, horizon, None)
        return TimeseriesType(target=target, past_covariates=past_cov, future_covariates=future_cov)

    @staticmethod
    def _build_covariate_tensors(
        covariates: Optional[Dict[str, Any]],
        context_len: int,
        horizon: int,
        series_idx: Optional[int],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Build past [V_p, T] and future [V_f, T+H] covariate tensors from a covariate dict."""
        if not covariates:
            return None, None
        label = f"series {series_idx}" if series_idx is not None else "the series"

        past_rows = []
        for name, values in (covariates.get("past") or {}).items():
            if len(values) != context_len:
                raise ValueError(
                    f"Past covariate '{name}' of {label} has length {len(values)}, "
                    f"expected the history length ({context_len})"
                )
            past_rows.append(_to_float_list(values))

        future_rows = []
        for name, values in (covariates.get("future") or {}).items():
            vals = _to_float_list(values)
            if len(vals) == horizon:
                # Only the forecast window given: historical part unknown -> NaN (masked)
                vals = [float("nan")] * context_len + vals
            elif len(vals) != context_len + horizon:
                raise ValueError(
                    f"Future covariate '{name}' of {label} has length {len(values)}, expected "
                    f"history+horizon ({context_len + horizon}) or horizon ({horizon})"
                )
            future_rows.append(vals)

        past = torch.tensor(past_rows, dtype=torch.float32) if past_rows else None
        future = torch.tensor(future_rows, dtype=torch.float32) if future_rows else None
        return past, future

    @staticmethod
    def _extend_to_horizon(arr: np.ndarray, horizon: int) -> np.ndarray:
        """TiRex-2 caps the prediction length at its supported maximum; extend a truncated
        [Q, H'] forecast by repeating the last step so the full horizon is covered."""
        if arr.shape[-1] < horizon:
            pad = np.repeat(arr[:, -1:], horizon - arr.shape[-1], axis=-1)
            arr = np.concatenate([arr, pad], axis=-1)
        return arr
