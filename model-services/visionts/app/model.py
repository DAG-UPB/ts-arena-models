import torch
from visionts import VisionTSpp, freq_to_seasonality_list
import numpy as np
import os
from typing import List, Union, Dict, Any

device = "cuda" if torch.cuda.is_available() else "cpu"

class VisionTSModel:
    """
    VisionTS++ forecasting model.
    Uses visual masked autoencoders for zero-shot time series forecasting.
    """
    
    def __init__(self) -> None:
        """
        Initializes the VisionTS++ model.
        MODEL_ID specifies the checkpoint file name (visiontspp_base.ckpt, visiontspp_large.ckpt)
        VM_ARCH specifies the architecture (mae_base or mae_large)
        """
        checkpoint_name = os.getenv("MODEL_ID", "visiontspp_base.ckpt")
        self.arch = os.getenv("VM_ARCH", "mae_base")  # mae_base or mae_large
        ckpt_dir = "/models/visionts"
        
        # Ensure checkpoint directory exists
        os.makedirs(ckpt_dir, exist_ok=True)
        
        ckpt_path = os.path.join(ckpt_dir, checkpoint_name)
        
        print(f"Loading VisionTS++ model: checkpoint={checkpoint_name}, arch={self.arch}")
        print(f"Checkpoint path: {ckpt_path}")
        
        # VisionTS++ with probabilistic forecasting support
        self.model = VisionTSpp(
            arch=self.arch,
            finetune_type='ln',
            ckpt_dir=ckpt_dir,
            ckpt_path=ckpt_path,
            load_ckpt=True,  # Let VisionTSpp handle downloading
            quantile=True,  # Enable probabilistic forecasting
        )
        
        self.model = self.model.to(device)
        self.model.eval()
        
        print("VisionTS++ model loaded successfully")

    def predict(
        self,
        data: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        horizon: int,
        freq: str = "h",
        quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ) -> Dict[str, Any]:
        """
        Makes a forecast using VisionTS.

        Args:
            data: Either a single time series as a list of dicts [{"ts": <timestamp>, "value": <float>}, ...]
                  or a batch of time series as a list of such lists.
            horizon: Number of steps to forecast
            freq: Frequency string (e.g., "h", "D", "W", "M")
            quantile_levels: Quantile levels for probabilistic forecasting

        Returns:
            Dictionary with 'forecasts' and optionally 'quantiles'
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

        # Determine periodicity from frequency
        periodicity = self._get_periodicity(freq)
        
        # Find max length for padding
        max_len = max(len(h) for h in history)
        context_len = max_len
        
        # Update model config
        self.model.update_config(
            context_len=context_len,
            pred_len=horizon,
            periodicity=periodicity,
            norm_const=0.4,
            align_const=0.4,
        )

        results = []
        quantiles_results = []
        
        # Order in list: index 0=10%, 1=20%, 2=30%, 3=40%, 4=60%, 5=70%, 6=80%, 7=90%
        visiontspp_quantile_order = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
        
        with torch.no_grad():
            for series in history:
                # Pad shorter series if needed
                if len(series) < max_len:
                    padding = [series[0]] * (max_len - len(series))
                    series = padding + series
                
                # Convert to tensor: [batch, context_len, nvars]
                x = torch.tensor(series, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                x = x.to(device)
                
                # Forward pass - VisionTS++ returns [y, y_quantile_list]
                output = self.model(x)
                if isinstance(output, list) and len(output) == 2:
                    # VisionTS++ with quantiles returns [y, y_quantile_list]
                    y = output[0]  # Main prediction is median
                    y_quantile_list = output[1]  # List of 8 quantile tensors
                else:
                    y = output
                    y_quantile_list = None
                
                # Extract forecast: [batch, pred_len, nvars] -> [pred_len]
                # Use median as point forecast
                forecast = y[0, :, 0].cpu().numpy().tolist()
                results.append(forecast)
                
                # Extract quantiles if available (VisionTS++ only)
                quantile_dict = {}
                if y_quantile_list is not None and len(y_quantile_list) == 8:
                    # Map the 8 quantile outputs to their levels
                    for idx, q_level in enumerate(visiontspp_quantile_order):
                        if q_level in quantile_levels:
                            q_values = y_quantile_list[idx][0, :, 0].cpu().numpy().tolist()
                            quantile_dict[str(q_level)] = q_values
                    
                    # Add median (0.5) from main output
                    if 0.5 in quantile_levels:
                        quantile_dict[str(0.5)] = forecast
                    
                    quantiles_results.append(quantile_dict)

        if not is_batch:
            output = {'forecasts': results[0]}
            if quantiles_results:
                output['quantiles'] = quantiles_results[0]
            return output
        
        output = {'forecasts': results}
        if quantiles_results:
            output['quantiles'] = quantiles_results
        return output

    def _get_periodicity(self, freq: str) -> int:
        """
        Get periodicity based on frequency string.
        """
        # Use VisionTS's built-in utility if available
        try:
            periods = freq_to_seasonality_list(freq)
            if periods:
                return periods[0]
        except Exception:
            pass
        
        # Fallback mapping using frequency strings
        freq_to_period = {
            # Minute intervals: periodicity = intervals per day
            "1min": 1440,   # 24*60 = 1440 minutes per day
            "15min": 96,    # 24*60/15 = 96 intervals per day
            "30min": 48,    # 24*60/30 = 48 intervals per day
            # Hourly: 24 hours per day
            "h": 24,
            # Daily: 7 days per week
            "D": 7,
            # Weekly: 52 weeks per year
            "W": 52,
            # Monthly: 12 months per year
            "M": 12,
        }
        return freq_to_period.get(freq, 24)
