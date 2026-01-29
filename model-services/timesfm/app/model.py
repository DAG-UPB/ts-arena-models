import torch
import timesfm
import numpy as np
import os
from typing import List, Union, Dict, Any
from huggingface_hub import hf_hub_download
from pathlib import Path

device = "gpu" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class TimesFMModel:
    def __init__(self) -> None:
        self.model_id = os.getenv("MODEL_ID", "google/timesfm-2.0-500m-pytorch")
        print(f"Loading model from {self.model_id}...")
        local_dir = Path(f"/models/{self.model_id.split('/')[-1]}")
        print(f"Ensuring model is present in {local_dir}...")
        filename = "torch_model.ckpt"
        hf_hub_download(repo_id=self.model_id, local_dir=local_dir, filename=filename)

        checkpoint_path = local_dir / filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        # For Torch
        if "2.0-500m" in self.model_id:
            num_layers = 50
            context_len = 2048
            use_pos_emb = False
        elif "1.0-200m" in self.model_id:
            num_layers = 20
            context_len = 512
            use_pos_emb = Trues
        else:
            print(f"Warning: Unknown model ID {self.model_id}, using default parameters for 2.0-500m")
            num_layers = 50
            context_len = 2048
            use_pos_emb = False

        self.tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=device,
                per_core_batch_size=32,
                horizon_len=168,
                num_layers=num_layers,
                use_positional_embedding=use_pos_emb,
                context_len=context_len,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self.model_id,
                local_dir=local_dir,
            ),
        )

    def predict(
        self,
        history: Union[List[float], List[List[float]]],
        horizon: int,
        quantile_levels: List[float] = None
    ) -> Dict[str, Any]:
        """
        Generate forecasts with TimesFM model.
        
        TimesFM 1.0 and 2.0 both output native quantiles via experimental_quantiles.
        Output shape is (batch, horizon, 10):
        - Column 0: mean/point forecast  
        - Columns 1-9: quantiles 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
        
        Args:
            history: Time series data (single or batch)
            horizon: Forecast horizon
            quantile_levels: Quantile levels to compute (default: 0.1 to 0.9)
        
        Returns:
            Dict with 'forecasts' (point forecasts = median q_0.5) and 'quantiles'
        """
        if quantile_levels is None:
            quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            
        if not history:
            raise ValueError("History cannot be empty.")

        is_single_series = not history or not isinstance(history[0], list)
        if is_single_series:
            history = [history]

        processed_history = []
        for series in history:
            if len(series) > self.tfm.hparams.context_len:
                series = series[-self.tfm.hparams.context_len:]
            processed_history.append(np.array(series))

        frequency_input = [0] * len(processed_history)

        # TimesFM forecast returns (point_forecast, quantile_forecast)
        # quantile_forecast shape: (batch, horizon, 10)
        # Col 0 = mean, Cols 1-9 = quantiles 0.1-0.9
        point_forecast, quantile_forecast = self.tfm.forecast(
            processed_history,
            freq=frequency_input,
        )
        
        all_forecasts = []
        all_quantiles = []
        
        for i in range(len(processed_history)):
            # Get quantile forecast for this series
            # Shape: (horizon, 10) where col 0 = mean, cols 1-9 = q_0.1 to q_0.9
            q_forecast = quantile_forecast[i]
            
            quantile_dict = {}
            for q in quantile_levels:
                # Map quantile level to column index
                # 0.1 -> col 1, 0.2 -> col 2, ..., 0.5 -> col 5, ..., 0.9 -> col 9
                col_idx = int(round(q * 10))
                q_values = q_forecast[:horizon, col_idx]
                quantile_dict[str(q)] = q_values.tolist()
            
            all_quantiles.append(quantile_dict)
            
            # Use q_0.5 (median) as point forecast for consistency
            all_forecasts.append(quantile_dict['0.5'])

        # Return structured output
        if is_single_series:
            return {
                'forecasts': all_forecasts[0],
                'quantiles': all_quantiles[0]
            }
        
        # For batch, format quantiles as dict with series index
        quantiles_by_series = {i: all_quantiles[i] for i in range(len(all_quantiles))}
        return {
            'forecasts': all_forecasts,
            'quantiles': quantiles_by_series
        }