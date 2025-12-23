import torch
import os
from transformers import AutoModelForCausalLM
from typing import List, Union, Optional, cast

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
        num_samples: int = 20,
    ) -> Union[List[float], List[List[float]]]:
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

        # Process sequences individually if they have different lengths
        if has_different_lengths:
            all_predictions = []
            for seq in series_list:
                seqs = torch.tensor([seq], dtype=torch.float32, device=self.device)
                
                # Generate multiple samples; return median as point forecast
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
                
                if num_samples == 1:
                    pred = out.squeeze(1).squeeze(0)  # [horizon]
                else:
                    pred = torch.median(out.squeeze(0), dim=0).values  # [horizon]
                
                all_predictions.append(pred.detach().cpu().tolist())
            
            return all_predictions[0] if is_single_series else all_predictions
        
        # Batch processing for sequences with same length
        seqs = torch.tensor(series_list, dtype=torch.float32, device=self.device)

        # Generate multiple samples; return median as point forecast
        out = self.model.generate(
            seqs,
            max_new_tokens=horizon,
            num_samples=num_samples,
        )

        B = seqs.shape[0]
        S = num_samples
        # Strictly expect [batch, samples, horizon]
        if out.ndim != 3 or out.shape[0] != B or out.shape[2] != horizon:
            raise RuntimeError(
                f"expected output shape (batch={B}, num_samples={S}, horizon={horizon}), got {tuple(out.shape)}"
            )
        if S == 1:
            if out.shape[1] != 1:
                raise RuntimeError(
                    f"expected output shape (batch={B}, num_samples=1, horizon={horizon}), got {tuple(out.shape)}"
                )
            preds = out.squeeze(1)
        else:
            if out.shape[1] != S:
                raise RuntimeError(
                    f"expected output shape (batch={B}, num_samples={S}, horizon={horizon}), got {tuple(out.shape)}"
                )
            preds = torch.median(out, dim=1).values  # [B, horizon]

        result = preds.detach().cpu().tolist()
        return result[0] if is_single_series else result