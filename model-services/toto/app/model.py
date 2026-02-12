import numpy as np
import os
from typing import List, Union, Dict, Any

import torch
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto


class TotoModel:
    """
    Toto forecasting model wrapper.
    Supports probabilistic forecasting with quantiles via Student-T mixture model.
    """
    
    # Standard quantile levels
    QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Number of samples for probabilistic forecasting
    NUM_SAMPLES = 256
    
    def __init__(self) -> None:
        """
        Initializes the Toto model from HuggingFace.
        Uses the Toto-Open-Base-1.0 checkpoint.
        """
        print("Initializing Toto model...")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model from HuggingFace
        model_name = os.getenv("MODEL_NAME", "Datadog/Toto-Open-Base-1.0")
        self.toto = Toto.from_pretrained(model_name)
        self.toto = self.toto.to(self.device)
        
        # Create forecaster
        self.forecaster = TotoForecaster(self.toto.model)
        
        print(f"Toto initialized (device={self.device})")

    def predict(
        self,
        data: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        horizon: int,
        freq: str = "h",
    ) -> Dict[str, Any]:
        """
        Makes a forecast using Toto.

        Args:
            data: Either a single time series as a list of dicts [{"ts": <timestamp>, "value": <float>}, ...]
                  or a batch of time series as a list of such lists.
            horizon: Number of steps to forecast
            freq: Frequency string (e.g., "h", "D", "W", "M", "15min")

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
        
        # Process each series
        all_forecasts = []
        all_quantiles = []
        
        for idx, series in enumerate(data_as_batch):
            try:
                # Extract values
                values = [item["value"] for item in series]
                
                # Convert to tensor: (1, seq_len) for univariate
                input_series = torch.tensor(values, dtype=torch.float32).unsqueeze(0).to(self.device)
                n_vars, seq_len = input_series.shape
                
                # Prepare timestamp information (not used by current model but required by API)
                timestamp_seconds = torch.zeros(n_vars, seq_len, device=self.device)
                
                # Determine time interval from frequency
                time_interval = self._freq_to_seconds(freq)
                time_interval_seconds = torch.full((n_vars,), time_interval, device=self.device)
                
                # Create MaskedTimeseries object
                inputs = MaskedTimeseries(
                    series=input_series,
                    padding_mask=torch.full_like(input_series, True, dtype=torch.bool),
                    id_mask=torch.zeros_like(input_series),
                    timestamp_seconds=timestamp_seconds,
                    time_interval_seconds=time_interval_seconds,
                )
                
                # Generate forecast
                forecast = self.forecaster.forecast(
                    inputs,
                    prediction_length=horizon,
                    num_samples=self.NUM_SAMPLES,
                    samples_per_batch=min(self.NUM_SAMPLES, 64),  # Control memory usage
                )
                
                # Extract median as point forecast
                # Shape is [batch, vars, horizon] = [1, 1, horizon]
                median_prediction = forecast.median.cpu().numpy()
                forecast_values = median_prediction[0, 0, :].tolist()  # First batch, first variable
                
                # Extract quantiles
                quantiles = {}
                for q in self.QUANTILE_LEVELS:
                    q_values = forecast.quantile(q).cpu().numpy()
                    quantiles[str(q)] = q_values[0, 0, :].tolist()  # First batch, first variable
                
                all_forecasts.append(forecast_values)
                all_quantiles.append(quantiles)
                
            except Exception as e:
                print(f"Error predicting series {idx}: {e}")
                import traceback
                traceback.print_exc()
                all_forecasts.append([0.0] * horizon)
                all_quantiles.append({str(q): [0.0] * horizon for q in self.QUANTILE_LEVELS})
        
        # Return results
        if not is_batch:
            return {
                'forecasts': all_forecasts[0],
                'quantiles': all_quantiles[0]
            }
        
        return {
            'forecasts': all_forecasts,
            'quantiles': all_quantiles
        }
    
    def _freq_to_seconds(self, freq: str) -> int:
        """Convert frequency string to seconds."""
        freq_map = {
            '15min': 15 * 60,
            '30min': 30 * 60,
            'h': 60 * 60,
            'H': 60 * 60,
            '1h': 60 * 60,
            'D': 24 * 60 * 60,
            'd': 24 * 60 * 60,
            'W': 7 * 24 * 60 * 60,
            'M': 30 * 24 * 60 * 60,
        }
        return freq_map.get(freq, 60 * 60)  # Default to 1 hour
