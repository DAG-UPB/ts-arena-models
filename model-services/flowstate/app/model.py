import torch
import numpy as np
from typing import List, Union, Dict, Any
import os

from tsfm_public import FlowStateForPrediction

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Scale factor mapping for common frequencies
SCALE_FACTORS = {
    # Minute intervals (assuming daily cycle)
    "1min": 0.0167,   # 24 / 1440 (minutes in a day)
    "5min": 0.0833,   # 24 / 288 (5-min intervals in a day)
    "10min": 0.1667,  # 24 / 144 (10-min intervals in a day)
    "15min": 0.25,    # 24 / 96 (quarter-hourly with daily cycle)
    "30min": 0.5,     # 24 / 48 (half-hourly with daily cycle)
    # Hourly (base scale)
    "h": 1.0,         # 24 / 24 (hourly with daily cycle)
    # Daily (assuming weekly cycle)
    "D": 3.43,        # 24 / 7 (daily with weekly cycle)
    # Weekly (assuming yearly cycle)
    "W": 0.46,        # 24 / 52 (weekly with yearly cycle)
    # Monthly (assuming yearly cycle)
    "M": 2.0,         # 24 / 12 (monthly with yearly cycle)
}


class FlowstateModel:
    def __init__(self) -> None:
        """
        Initializes the FlowState model from HuggingFace.
        """
        model_id = os.getenv("MODEL_ID", "ibm-research/flowstate")
        
        print(f"Loading FlowState model: {model_id}")
        
        self.model = FlowStateForPrediction.from_pretrained(model_id)
        self.model = self.model.to(device)
        self.model.eval()
        
        print("FlowState model loaded successfully")

    def _get_scale_factor(self, freq: str) -> float:
        """Get scale factor for given frequency."""
        if freq in SCALE_FACTORS:
            return SCALE_FACTORS[freq]
        # Default to hourly
        return 1.0

    def predict(
        self,
        data: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        horizon: int,
        freq: str = "h",
        quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ) -> Dict[str, Any]:
        """
        Makes a forecast using FlowState.

        Args:
            data: Either a single time series as a list of dicts [{"ts": <timestamp>, "value": <float>}, ...]
                  or a batch of time series as a list of such lists.
            horizon: Number of steps to forecast
            freq: Frequency string (used to determine scale factor)
            quantile_levels: Quantile levels for probabilistic forecasting

        Returns:
            Dictionary with 'forecasts' and 'quantiles'
        """
        if not data:
            return {'forecasts': [], 'quantiles': {}}
        
        # Detect single-series vs. batch input
        is_batch = isinstance(data[0], list)
        
        if is_batch:
            data_as_batch = data
        else:
            data_as_batch = [data]

        scale_factor = self._get_scale_factor(freq)
        
        results = []
        quantiles_results = []
        
        for series_data in data_as_batch:
            # Extract values from the data format
            values = [item["value"] for item in series_data]
            
            # Convert to numpy and normalize
            context_np = np.array(values, dtype=np.float32)
            mean = context_np.mean()
            std = context_np.std() + 1e-6
            context_norm = (context_np - mean) / std
            
            # Convert to tensor (context, batch=1, channels=1)
            x = torch.tensor(context_norm, dtype=torch.float32).unsqueeze(1).unsqueeze(-1).to(device)
            
            with torch.no_grad():
                # FlowState forward pass
                outputs = self.model(
                    x, 
                    scale_factor=scale_factor, 
                    prediction_length=horizon,
                    batch_first=False
                )
                
                # Get prediction outputs, shape: (batch, forecast_length, n_ch)
                if hasattr(outputs, 'prediction_outputs'):
                    forecast = outputs.prediction_outputs
                else:
                    forecast = outputs
                
                # forecast shape: (batch, forecast_length, n_ch) -> squeeze to (forecast_length,)
                point_forecast_norm = forecast.squeeze(0).squeeze(-1).cpu().numpy()
                
                # Get quantile outputs if available - shape: (batch, quantiles, forecast_length, n_ch)
                quantile_forecast = None
                if hasattr(outputs, 'quantile_outputs') and outputs.quantile_outputs is not None:
                    quantile_forecast = outputs.quantile_outputs.squeeze(0).squeeze(-1).cpu().numpy()
                    # Now shape is (quantiles, forecast_length)
            
            # Denormalize point forecast
            point_forecast = point_forecast_norm * std + mean
            
            # Extract quantiles from model output.
            # FlowState outputs shape: (batch, n_quantiles, forecast_length, n_ch).
            EXPECTED_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

            quantiles_dict = {}
            if quantile_forecast is not None and len(quantile_forecast.shape) == 2:
                n_quantiles = quantile_forecast.shape[0]
                # Defensive guard: `tsfm_public` is NOT importable offline, so the
                # declared quantile grid of FlowStateForPrediction cannot be
                # introspected here. The model is expected to emit exactly 9
                # deciles ordered 0.1..0.9 along axis 0. The exact level ordering
                # MUST be confirmed live on the GPU host by inspecting
                # self.model's config / output metadata. If the grid differs,
                # the assertion below raises loudly rather than silently
                # mislabelling heads.
                if n_quantiles != len(EXPECTED_QUANTILE_LEVELS):
                    raise RuntimeError(
                        f"FlowState returned {n_quantiles} quantile heads, "
                        f"expected {len(EXPECTED_QUANTILE_LEVELS)} "
                        f"(levels {EXPECTED_QUANTILE_LEVELS}). Confirm the "
                        f"model's quantile grid live on the GPU host."
                    )

                # Denormalize all quantile heads at once: (9, forecast_length)
                q_denorm = quantile_forecast * std + mean

                # Enforce monotonicity per forecast step: sort the 9 denormalized
                # quantile values ascending and re-assign to levels 0.1..0.9 so
                # q_0.1 <= ... <= q_0.9. This mirrors timesfm2_5's
                # `fix_quantile_crossing` intent and guarantees a valid
                # (non-crossing) quantile distribution after denormalization.
                q_sorted = np.sort(q_denorm, axis=0)

                # Emit a q_* entry only for levels the model actually produced.
                # Do NOT fabricate entries for missing/unavailable levels (the
                # backend handles degenerate substitution centrally per
                # backend #13).
                for level, row in zip(EXPECTED_QUANTILE_LEVELS, q_sorted):
                    quantiles_dict[level] = row.tolist()

                # Median consistency: q_0.5 is the middle (index 4) of the
                # sorted 9-decile grid and is used as the point `value`.
                results.append(q_sorted[len(EXPECTED_QUANTILE_LEVELS) // 2].tolist())
            else:
                # No quantile heads emitted by the model: emit NO quantiles
                # (empty dict) rather than fabricating a degenerate distribution
                # of point-forecast copies. The backend handles degenerate
                # substitution centrally per backend #13.
                results.append(point_forecast.tolist())

            quantiles_results.append(quantiles_dict)
        
        if not is_batch:
            return {
                "forecasts": results[0],
                "quantiles": quantiles_results[0]
            }
        
        return {
            "forecasts": results,
            "quantiles": quantiles_results
        }
