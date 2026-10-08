import logging

logger = logging.getLogger(__name__)

import os
from typing import Any, Dict, List, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel

# The modelling code ships only with the merged repo; both variants run it, pinned.
TABBY_REPO = "paris-noah/Tabby"
TABBY_REVISION = "1b2381d7f1b0982aa1d5f09ce01b99eb4ee57e01"
PRETRAIN_REPO = "paris-noah/Tabby-Pretrain"
PRETRAIN_REVISION = "b26c277bf96087712ea3ae65740947ffe276ae61"


class TabbyModel:
    """
    Tabby (Huawei Noah's Ark Paris) in two variants, picked by MODEL_ID:

    - paris-noah/Tabby: the backbone with its tuned prompt, as published.
    - paris-noah/Tabby-Pretrain: the zero-shot backbone. Same code with the prompt
      disabled (prompt_len=0), backbone weights from the pretrain repo.

    One forward pass returns 99 monotone quantiles (0.01 ... 0.99); the deciles are
    picked from them by level and the point forecast is the median.
    """

    QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(self) -> None:
        logger.info("Initializing Tabby model...")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_id = os.getenv("MODEL_ID", TABBY_REPO)

        if model_id == TABBY_REPO:
            model = AutoModel.from_pretrained(
                TABBY_REPO, revision=TABBY_REVISION, trust_remote_code=True
            )
        elif model_id == PRETRAIN_REPO:
            config = AutoConfig.from_pretrained(
                TABBY_REPO, revision=TABBY_REVISION, trust_remote_code=True,
                prompt_len=0, context_aware=False,
            )
            model = AutoModel.from_config(
                config, trust_remote_code=True, code_revision=TABBY_REVISION
            )
            weights = hf_hub_download(PRETRAIN_REPO, "model.safetensors", revision=PRETRAIN_REVISION)
            model.backbone.load_state_dict(load_file(weights), strict=True)
        else:
            raise ValueError(f"Unsupported MODEL_ID: {model_id}")
        # fp32: proteus' Turing GPUs have no native bf16.
        self.model = model.float().to(self.device).eval()

        levels = [round(level, 4) for level in self.model.quantile_levels]
        missing = [q for q in self.QUANTILE_LEVELS if q not in levels]
        if missing:
            raise RuntimeError(f"Tabby quantile grid lacks levels {missing}")
        self.level_idx = [levels.index(q) for q in self.QUANTILE_LEVELS]

        # Series per forward pass; bounds GPU memory on large rounds.
        self.batch_size = int(os.getenv("TABBY_BATCH_SIZE", "32"))
        # History fed to the model; 8096 is the published evaluation protocol. The
        # model further trims it to fit the 8192 window next to the forecast span.
        self.max_context = int(os.getenv("TABBY_MAX_CONTEXT", "8096"))

        logger.info(
            f"Tabby initialized (model={model_id}, device={self.device}, "
            f"prompt_len={self.model.config.prompt_len}, "
            f"min_forecast_span={self.model.config.min_forecast_span})"
        )

    def predict(
        self,
        data: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        horizon: int,
        freq: str = "h",
    ) -> Dict[str, Any]:
        """
        Forecast one series ([{"ts", "value"}, ...]) or a batch (list of such lists).

        Returns {'forecasts': ..., 'quantiles': ...}, per series in the batch case.
        """
        if not data:
            return {'forecasts': [], 'quantiles': {}}

        is_batch = isinstance(data[0], list)
        series_list = data if is_batch else [data]

        all_forecasts: List[List[float]] = []
        all_quantiles: List[Dict[str, List[float]]] = []
        for start in range(0, len(series_list), self.batch_size):
            chunk = series_list[start:start + self.batch_size]
            forecasts, quantiles = self._predict_chunk(chunk, horizon)
            all_forecasts.extend(forecasts)
            all_quantiles.extend(quantiles)

        if not is_batch:
            return {'forecasts': all_forecasts[0], 'quantiles': all_quantiles[0]}
        return {'forecasts': all_forecasts, 'quantiles': all_quantiles}

    @torch.no_grad()
    def _predict_chunk(self, chunk: List[List[Dict[str, Any]]], horizon: int):
        values = [
            np.array(
                [np.nan if item["value"] is None else float(item["value"]) for item in series],
                dtype=np.float32,
            )[-self.max_context:]
            for series in chunk
        ]

        # Right-align the batch: left padding is flagged as padding, NaN inside a
        # series stays NaN and is treated as unobserved.
        width = max(len(v) for v in values)
        context = np.full((len(values), width), np.nan, dtype=np.float32)
        past_is_pad = np.ones((len(values), width), dtype=bool)
        for i, v in enumerate(values):
            context[i, width - len(v):] = v
            past_is_pad[i, width - len(v):] = False

        out = self.model(
            context=torch.from_numpy(context).to(self.device),
            past_is_pad=torch.from_numpy(past_is_pad).to(self.device),
            prediction_length=horizon,
        )["quantile_preds"]
        # [batch, 99, horizon] -> [batch, horizon, 9]; the head is monotone, the sort
        # only guards against float ties.
        q = out[:, self.level_idx, :].float().cpu().numpy().transpose(0, 2, 1)
        q = np.sort(q, axis=-1)

        forecasts, quantiles = [], []
        for deciles in q:
            quantiles.append({
                str(level): deciles[:, j].tolist() for j, level in enumerate(self.QUANTILE_LEVELS)
            })
            forecasts.append(deciles[:, self.QUANTILE_LEVELS.index(0.5)].tolist())
        return forecasts, quantiles
