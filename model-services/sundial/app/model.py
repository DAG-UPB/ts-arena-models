import torch
import os
import numpy as np
from transformers import AutoModelForCausalLM
from typing import List, Union, Optional, cast, Dict, Any

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class SundialModel:
    def __init__(self) -> None:
        self.device = torch.device(device)
        model_id = os.getenv("MODEL_ID", "thuml/sundial-base-128m")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=f"/models/{model_id.split('/')[-1]}/",
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        history: Union[List[float], List[List[float]]],
        horizon: int,
        num_samples: int = 100,
        quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ) -> Dict[str, Any]:
        """
        Generate forecasts with Sundial model.
        
        Args:
            history: Time series data (single or batch)
            horizon: Forecast horizon
            num_samples: Number of samples for probabilistic forecasting
            quantile_levels: Quantile levels to compute (default: 0.1 to 0.9)
        
        Returns:
            Dict with 'forecasts' (point forecasts) and 'quantiles' (quantile predictions)
        """
    
        if not history:
            raise ValueError("history cannot be empty")
        if horizon <= 0:
            raise ValueError("horizon must be > 0")

        # Normalize to batch form [batch, length]. History itself is the lookback.
        is_single_series = not isinstance(history[0], list)
        if is_single_series:
            series_list: List[List[float]] = [cast(List[float], history)]
        else:
            series_list = cast(List[List[float]], history)

        # Check if sequences have different lengths
        lengths = [len(seq) for seq in series_list]
        has_different_lengths = len(set(lengths)) != 1

        all_forecasts = []
        all_quantiles = []

        # Process sequences individually if they have different lengths
        if has_different_lengths:
            for seq in series_list:
                seqs = torch.tensor([seq], dtype=torch.float32, device=self.device)
                
                # Generate multiple samples
                out = self.model.generate(
                    seqs,
                    max_new_tokens=horizon,
                    num_samples=num_samples,
                )
                
                # Expect [1, num_samples, horizon]
                if out.ndim != 3 or out.shape[0] != 1 or out.shape[2] != horizon:
                    raise RuntimeError(
                        f"expected output shape (batch=1, num_samples={num_samples}, horizon={horizon}), got {tuple(out.shape)}"
                    )
                
                samples = out.squeeze(0).detach().cpu().numpy()  # [num_samples, horizon]
                
                # Point forecast: median of samples
                point_forecast = np.median(samples, axis=0).tolist()
                all_forecasts.append(point_forecast)
                
                # Compute quantiles
                quantile_dict = {}
                for q in quantile_levels:
                    q_values = np.quantile(samples, q, axis=0).tolist()
                    quantile_dict[str(q)] = q_values
                all_quantiles.append(quantile_dict)
        else:
            # Batch processing for sequences with same length
            seqs = torch.tensor(series_list, dtype=torch.float32, device=self.device)

            # Generate multiple samples
            out = self.model.generate(
                seqs,
                max_new_tokens=horizon,
                num_samples=num_samples,
            )

            B = seqs.shape[0]
            # Expect [batch, samples, horizon]
            if out.ndim != 3 or out.shape[0] != B or out.shape[2] != horizon:
                raise RuntimeError(
                    f"expected output shape (batch={B}, num_samples={num_samples}, horizon={horizon}), got {tuple(out.shape)}"
                )
            
            samples = out.detach().cpu().numpy()  # [batch, num_samples, horizon]
            
            for i in range(B):
                series_samples = samples[i]  # [num_samples, horizon]
                
                # Point forecast: median
                point_forecast = np.median(series_samples, axis=0).tolist()
                all_forecasts.append(point_forecast)
                
                # Compute quantiles
                quantile_dict = {}
                for q in quantile_levels:
                    q_values = np.quantile(series_samples, q, axis=0).tolist()
                    quantile_dict[str(q)] = q_values
                all_quantiles.append(quantile_dict)

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