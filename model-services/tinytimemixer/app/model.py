import torch
import numpy as np
from typing import List, Union, Dict, Any
import os

from tsfm_public.toolkit.get_model import get_model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


class TinyTimeMixerModel:
    def __init__(self) -> None:
        """
        Initializes the TinyTimeMixer model from HuggingFace.
        Uses get_model() for automatic model selection based on context/prediction length.
        """
        # Model ID
        model_id = os.getenv("MODEL_ID", "ibm-granite/granite-timeseries-ttm-r2")
        
        # Default context and prediction lengths
        self.context_length = int(os.getenv("TTM_CONTEXT_LENGTH", "512"))
        self.prediction_length = int(os.getenv("TTM_PREDICTION_LENGTH", "96"))
        
        print(f"Loading TinyTimeMixer model: {model_id}")
        print(f"Context length: {self.context_length}, Prediction length: {self.prediction_length}")
        
        # Use get_model for automatic model selection
        self.model = get_model(
            model_path=model_id,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
        )
        self.model = self.model.to(device)
        self.model.eval()
        
        print("TinyTimeMixer model loaded successfully")

    def predict(
        self,
        data: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        horizon: int,
        freq: str = "h",
        quantile_levels: List[float] = [0.1, 0.5, 0.9]
    ) -> Dict[str, Any]:
        """
        Makes a forecast using TinyTimeMixer.

        Args:
            data: Either a single time series as a list of dicts [{"ts": <timestamp>, "value": <float>}, ...]
                  or a batch of time series as a list of such lists.
            horizon: Number of steps to forecast
            freq: Frequency string
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

        results = []
        quantiles_results = []
        
        for series in history:
            # Prepare context
            if len(series) > self.context_length:
                context = series[-self.context_length:]
            else:
                # Pad from the start if shorter
                padding = [series[0]] * (self.context_length - len(series))
                context = padding + series
            
            # Convert to numpy and normalize
            context_np = np.array(context, dtype=np.float32)
            mean = context_np.mean()
            std = context_np.std() + 1e-6
            context_norm = (context_np - mean) / std
            
            # Convert to tensor (batch=1, seq_len, channels=1)
            x = torch.tensor(context_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            
            with torch.no_grad():
                # TTM forward pass
                outputs = self.model(x)
                
                # Get prediction outputs
                if hasattr(outputs, 'prediction_outputs'):
                    forecast = outputs.prediction_outputs
                elif isinstance(outputs, dict) and 'prediction_outputs' in outputs:
                    forecast = outputs['prediction_outputs']
                elif isinstance(outputs, tuple):
                    forecast = outputs[0]
                else:
                    forecast = outputs
                
                # Shape: (batch, pred_len, channels)
                forecast = forecast.squeeze().cpu().numpy()
                
                # Handle different output shapes
                if len(forecast.shape) > 1:
                    forecast = forecast[:, 0]  # Take first channel
                
                # Take required horizon
                forecast = forecast[:horizon]
                
                # Denormalize
                forecast = forecast * std + mean
            
            results.append(forecast.tolist())
            quantiles_results.append({})

        if not is_batch:
            return {
                "forecasts": results[0],
                "quantiles": quantiles_results[0]
            }
        
        return {
            "forecasts": results,
            "quantiles": quantiles_results
        }
