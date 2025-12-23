import torch
import timesfm
import numpy as np
import os
from typing import List, Union
from huggingface_hub import snapshot_download
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
        elif "1.0-200m" in self.model_id:
            num_layers = 20
            context_len = 2048
        else:
            print(f"Warning: Unknown model ID {self.model_id}, using default parameters for 2.0-500m")
            num_layers = 50
            context_len = 2048

        self.tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=device,
                per_core_batch_size=32,
                horizon_len=168,
                num_layers=num_layers,
                use_positional_embedding=False,
                context_len=context_len,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self.model_id,
                local_dir=local_dir,
            ),
        )

    def predict(self, history: Union[List[float], List[List[float]]], horizon: int) -> Union[List[float], List[List[float]]]:
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

        # Scaling for TimesFM 1.0
        if "1.0-200m" in self.model_id:
            scaled_history = []
            scales = []
            means = []
            for series in processed_history:
                mean = np.mean(series)
                std = np.std(series)
                if std == 0:
                    std = 1.0
                scaled_series = (series - mean) / std
                scaled_history.append(scaled_series)
                scales.append(std)
                means.append(mean)
            
            input_history = scaled_history
        else:
            input_history = processed_history

        point_forecast, _ = self.tfm.forecast(
            input_history,
            freq=frequency_input,
        )
        
        result = []
        if "1.0-200m" in self.model_id:
             for i, pf in enumerate(point_forecast):
                rescaled_pf = (pf * scales[i]) + means[i]
                result.append(rescaled_pf[:horizon].tolist())
        else:
             result = [pf[:horizon].tolist() for pf in point_forecast]

        if is_single_series:
            return result[0]
        return result