import torch
from tirex import load_model, ForecastModel
import numpy as np
import os
from typing import List, Union, Dict, Any

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


class TiRexModel:
    def __init__(self) -> None:
        """
        Initializes the TiRex model from HuggingFace.
        """
        model_id = os.getenv("MODEL_ID", "NX-AI/TiRex")
        print(f"Loading TiRex model from {model_id}...")
        self.model: ForecastModel = load_model(model_id)
        self.model = self.model.to(device)
        print("TiRex model loaded successfully")

    def predict(
        self,
        data: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        horizon: int,
        freq: str = "h",
        quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ) -> Dict[str, Any]:
        """
        Makes a forecast using TiRex.

        Args:
            data: Either a single time series as a list of dicts [{"ts": <timestamp>, "value": <float>}, ...]
                  or a batch of time series as a list of such lists.
            horizon: Number of steps to forecast
            freq: Frequency string (not used by TiRex but kept for API compatibility)
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
        
        # Extract values from dict format
        history = [[item["value"] for item in series] for series in data_as_batch]

        # Convert to tensor, shape (batch, time_steps)
        # Pad shorter series to max length
        max_len = max(len(h) for h in history)
        padded_history = []
        for h in history:
            if len(h) < max_len:
                # Pad with the first value
                padding = [h[0]] * (max_len - len(h))
                padded_history.append(padding + h)
            else:
                padded_history.append(h)
        
        context = torch.tensor(padded_history, dtype=torch.float32).to(device)
        
        # TiRex returns quantiles and mean
        with torch.no_grad():
            quantiles, mean = self.model.forecast(
                context=context,
                prediction_length=horizon
            )
        
        # Convert to numpy
        mean_np = mean.cpu().numpy()
        quantiles_np = quantiles.cpu().numpy()
        
        results = []
        quantiles_results = []
        
        for i in range(len(history)):
            # Quantile forecasts
            quantile_dict = {}
            for level in quantile_levels:
                if level in quantile_levels:
                    idx = quantile_levels.index(level)
                    quantile_dict[str(level)] = quantiles_np[i, :, idx].tolist()
            
            quantiles_results.append(quantile_dict)
            
            # Use q_0.5 (median) as point forecast for consistency
            median_idx = quantile_levels.index(0.5)
            results.append(quantiles_np[i, :, median_idx].tolist())

        if not is_batch:
            return {
                "forecasts": results[0],
                "quantiles": quantiles_results[0]
            }
        
        return {
            "forecasts": results,
            "quantiles": quantiles_results
        }
