import torch
from transformers import AutoModelForCausalLM
import numpy as np
import os
from typing import List, Union
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
class TimeMoEModel:
    def __init__(self) -> None:
        """
        Initializes the TimeMoE model
        """
        model_id = os.getenv("MODEL_ID", "Maple728/TimeMoE-50M")
        # Prepare pre-trained model by downloading model weights from huggingface hub
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,  # use "cpu" for CPU inference, and "cuda" for GPU inference.
            trust_remote_code=True,
            cache_dir=f"/models/{model_id.split('/')[-1]}/",  # Directory to cache the model
        )

    def predict(self, history: Union[List[float], List[List[float]]], horizon: int) -> Union[List[float], List[List[float]]]:
        """
        Makes a forecast using the Moirai-MoE model.

        Args:
            history: A list of historical values.
            horizon: The number of steps to forecast into the future.

        Returns:
            A list of forecasted values.
        """
        is_single_series = not history or not isinstance(history[0], list)
        if is_single_series:
            history = [history]

        # convert history to float and add batch dimension
        history_np = np.array(history, dtype=np.float32)
        # Convert history to tensor
        seqs = torch.from_numpy(history_np).to(device)

        # normalize seqs
        mean, std = seqs.mean(dim=-1, keepdim=True), seqs.std(dim=-1, keepdim=True)
        std = std + 1e-8
        normed_seqs = (seqs - mean) / std

        # forecast
        output = self.model.generate(normed_seqs, max_new_tokens=horizon, min_new_tokens=horizon, eos_token_id=None)
        normed_predictions = output[:, -horizon:]

        # inverse normalize
        predictions = normed_predictions * std + mean

        result = predictions.tolist()
        if is_single_series:
            return result[0]
        return result