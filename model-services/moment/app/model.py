import torch
import os
from typing import List, Union, Dict
from momentfm import MOMENTPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class MomentModel:
    def __init__(self) -> None:
        self.pipeline_cache: Dict[int, MOMENTPipeline] = {}
        self.model_name = os.getenv("MODEL_ID", "AutonLab/MOMENT-1-large")
        self.cache_dir = f"/models/{self.model_name.split('/')[-1]}"

    def _get_pipeline(self, forecast_horizon: int) -> MOMENTPipeline:
        if forecast_horizon not in self.pipeline_cache:
            print(f"Loading MOMENT pipeline for horizon {forecast_horizon}")
            self.pipeline_cache[forecast_horizon] = MOMENTPipeline.from_pretrained(
                self.model_name,
                model_kwargs={
                    'task_name': 'forecasting',
                    'forecast_horizon': forecast_horizon,
                    'head_dropout': 0.1,
                    'weight_decay': 0,
                    'freeze_encoder': True,
                    'freeze_embedder': True,
                    'freeze_head': False,
                },
                cache_dir=self.cache_dir,
            )
        return self.pipeline_cache[forecast_horizon]

    def predict(self, data: List[List[float]], horizon: int) -> List[List[float]]:
        # Ensure all elements are lists (of floats)
        assert isinstance(data, list) and all(isinstance(seq, list) for seq in data), \
            "Input data must be a list of lists (univariate time series)."

        # Pad sequences to the same length with NaN (left-padding)
        max_len = max(len(seq) for seq in data)
        padded = []
        for seq in data:
            pad_len = max_len - len(seq)
            padded_seq = [float('nan')] * pad_len + seq
            padded.append(padded_seq)

        # Shape: [batch_size, 1, seq_len]
        context = torch.tensor(padded, dtype=torch.float32).unsqueeze(1).to(device)

        # Mask: 1 for valid, 0 for NaN/padded (same shape as [batch_size, seq_len])
        input_mask = (~torch.isnan(context.squeeze(1))).float()

        pipeline = self._get_pipeline(horizon)
        pipeline.to(device)
        print("Horizon loaded in pipeline:", pipeline.head.linear.out_features)
        # Forecast
        outputs = pipeline.forecast(x_enc=context, input_mask=input_mask)
        forecast = outputs.forecast  # [batch_size, n_channels, forecast_horizon]
        print("Forecast shape:", forecast.shape)
        # Convert back to [batch_size, forecast_horizon]
        result = [
            forecast[i, 0, :horizon].detach().cpu().tolist()
            for i in range(forecast.shape[0])
        ]
        return result