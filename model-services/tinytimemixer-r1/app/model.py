import torch
import numpy as np
from typing import List, Union, Dict, Any
import os
import logging

from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


class TinyTimeMixerR1Model:
    def __init__(self) -> None:
        """
        Initializes the TinyTimeMixer R1 model from HuggingFace.
        """
        model_id = os.getenv("MODEL_ID", "ibm-granite/granite-timeseries-ttm-r1")
        revision = os.getenv("TTM_R1_REVISION", "main")  # main = 512-96 variant
        
        self.context_length = int(os.getenv("TTM_R1_CONTEXT_LENGTH", "512"))
        self.prediction_length = int(os.getenv("TTM_R1_PREDICTION_LENGTH", "96"))
        
        print(f"Loading TinyTimeMixer R1 model: {model_id} (revision: {revision})")
        print(f"Context length: {self.context_length}, Prediction length: {self.prediction_length}")
        
        self.model = TinyTimeMixerForPrediction.from_pretrained(
            model_id,
            revision=revision
        )
        self.model = self.model.to(device)
        self.model.eval()
        
        print("TinyTimeMixer R1 model loaded successfully")

    def predict(
        self,
        data: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        horizon: int,
        freq: str = "h",
        quantile_levels: List[float] = [0.1, 0.5, 0.9]
    ) -> Dict[str, Any]:
        """
        Makes a forecast using TinyTimeMixer R1.

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
            # Take last context_length values
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
                
                # forecast shape: (batch, prediction_length, channels)
                forecast = forecast.squeeze(0).squeeze(-1).cpu().numpy()
            
            # Denormalize
            forecast_denorm = forecast * std + mean
            
            # Truncate to requested horizon (warn if exceeds model capacity)
            max_pred_len = len(forecast_denorm)
            if horizon > max_pred_len:
                logger.warning(
                    f"Requested horizon {horizon} exceeds model max prediction length {max_pred_len}. "
                    f"Truncating to {max_pred_len} steps."
                )
                forecast_denorm = forecast_denorm[:max_pred_len]
            else:
                forecast_denorm = forecast_denorm[:horizon]
            
            results.append(forecast_denorm.tolist())
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
