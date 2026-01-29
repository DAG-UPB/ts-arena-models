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
            
            # Extract quantiles from model output
            # FlowState outputs shape: (batch, 9, forecast_length, n_ch)
            quantiles_dict = {}
            if quantile_forecast is not None and len(quantile_forecast.shape) == 2:
                n_quantiles = quantile_forecast.shape[0]
                quantile_mapping = {
                    0.1: 0,
                    0.2: 1,
                    0.3: 2,
                    0.4: 3,
                    0.5: 4,
                    0.6: 5,
                    0.7: 6,
                    0.8: 7,
                    0.9: 8
                }
                for q in quantile_levels:
                    if q in quantile_mapping and quantile_mapping[q] < n_quantiles:
                        q_values = quantile_forecast[quantile_mapping[q]] * std + mean
                        quantiles_dict[q] = q_values.tolist()
                    else:
                        quantiles_dict[q] = point_forecast.tolist()
                
                # Use q_0.5 (median) as point forecast for consistency
                if 0.5 in quantiles_dict:
                    results.append(quantiles_dict[0.5])
                else:
                    results.append(point_forecast.tolist())
            else:
                # No quantiles available, use point forecast for all
                for q in quantile_levels:
                    quantiles_dict[q] = point_forecast.tolist()
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
