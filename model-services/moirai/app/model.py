import os
from typing import List, Union, cast, Dict, Any

import torch
import pandas as pd
from gluonts.dataset.pandas import PandasDataset

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module  # Moirai 2.0[web:23]


def _infer_family_from_model_id(model_id: str) -> str:
    """
    Infer the Moirai model family from the full Hugging Face model ID.

    Expected patterns (last component after the slash):
      - 'moirai-1.x-...'      -> 'moirai'
      - 'moirai-moe-1.x-...'  -> 'moirai-moe'
      - 'moirai-2.0-...'      -> 'moirai2'

    Raises:
        ValueError: If the model_id does not match any known Moirai family.
    """
    # Only look at the last segment after the namespace (e.g. 'Salesforce/')
    name = model_id.split("/")[-1]

    if name.startswith("moirai-moe"):
        return "moirai-moe"
    if name.startswith("moirai-2."):
        return "moirai2"
    if name.startswith("moirai-1.") or name.startswith("moirai-"):
        # Fallback for classic Moirai variants
        return "moirai"

    raise ValueError(f"Could not infer Moirai family from model_id='{model_id}'")


class MoiraiModel:
    def __init__(self) -> None:
        """
        Initialize a unified Moirai wrapper using a single MODEL_ID environment variable.

        The full Hugging Face model ID is read from the environment and used
        both to:
          1. Download the correct checkpoint, and
          2. Infer which Moirai family to use (Moirai 1.x, Moirai-MoE, or Moirai 2.0).

        Environment:
            MODEL_ID (str): Full HF model ID. Examples:
                - 'Salesforce/moirai-1.1-R-large'
                - 'Salesforce/moirai-moe-1.0-R-base'
                - 'Salesforce/moirai-2.0-R-small'

        Attributes:
            model_id (str): The resolved HF model ID.
            family (str): One of {'moirai', 'moirai-moe', 'moirai2'}.
            module: The underlying uni2ts module instance corresponding
                to the chosen family.
        """
        model_id = os.getenv("MODEL_ID", "Salesforce/moirai-1.1-R-large")
        self.model_id = model_id
        self.family = _infer_family_from_model_id(model_id)

        if self.family == "moirai":
            self.module = MoiraiModule.from_pretrained(
                model_id,
                cache_dir=f"/models/{model_id.split('/')[-1]}",
            )
        elif self.family == "moirai-moe":
            self.module = MoiraiMoEModule.from_pretrained(
                model_id,
                cache_dir=f"/models/{model_id.split('/')[-1]}",
            )
        elif self.family == "moirai2":
            self.module = Moirai2Module.from_pretrained(
                model_id,
                cache_dir=f"/models/{model_id.split('/')[-1]}",
            )
        else:
            raise ValueError(f"Unsupported Moirai family: {self.family}")

    def _build_forecast_model(
        self,
        prediction_length: int,
        context_length: int,
    ):
        """
        Build the appropriate Forecast object for the currently selected family.

        Args:
            prediction_length: Number of steps to forecast into the future.
            context_length: Number of past time steps used as context.

        Returns:
            An instance of a uni2ts Forecast class compatible with GluonTS.

        Raises:
            ValueError: If the inferred family is not supported.
        """
        if self.family == "moirai":
            return MoiraiForecast(
                module=self.module,
                prediction_length=prediction_length,
                context_length=context_length,
                patch_size="auto",
                num_samples=100,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        elif self.family == "moirai-moe":
            # Moirai-MoE typically uses a fixed patch size (e.g., 16) in examples.[web:23]
            return MoiraiMoEForecast(
                module=self.module,
                prediction_length=prediction_length,
                context_length=context_length,
                patch_size=16,
                num_samples=100,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        elif self.family == "moirai2":
            # Moirai 2.0 keeps a compatible Forecast API in uni2ts.[web:23]
            return Moirai2Forecast(
                module=self.module,
                prediction_length=prediction_length,
                context_length=context_length,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        else:
            raise ValueError(f"Unsupported Moirai family: {self.family}")

    def predict(
        self,
        data: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        horizon: int,
        frequency: str = "h",
    ) -> Union[List[float], List[List[float]]]:
        """
        Run forecasting with the chosen Moirai model.

        Args:
            data:
                Either:
                  - A single time series as a list of dicts:
                        [{'ts': <timestamp>, 'value': <float>}, ...]
                  - A batch of time series as a list of such lists.
            horizon:
                Number of forecast steps (prediction_length).
            frequency:
                Pandas/GluonTS frequency string, e.g. "h", "D", "1min".

        Returns:
            If a single time series is passed:
                List[float] of length `horizon`.
            If a batch is passed:
                List[List[float]] with one forecast per input series.
        """
        if not data:
            return []

        # Detect single-series vs. batch input
        is_batch = isinstance(data[0], list)

        if is_batch:
            data_as_batch = cast(List[List[Dict[str, Any]]], data)
        else:
            data_as_batch = [cast(List[Dict[str, Any]], data)]

        all_forecasts = []

        # Process each series individually to handle varying lengths and avoid padding artifacts
        for series_data in data_as_batch:
            ts_values = [item["value"] for item in series_data]
            ts_timestamps = [item["ts"] for item in series_data]

            index = pd.to_datetime(ts_timestamps)
            series = pd.Series(ts_values, index=index, name="target")

            # Ensure index is sorted and uniformly spaced
            series = series.sort_index()
            series = series.resample(frequency).mean()

            df = series.to_frame()
            df["item_id"] = "0"
            
            # Use the actual length of this series as context length
            context_length = len(df)

            # Build the family-specific Forecast object for this specific series
            model = self._build_forecast_model(
                prediction_length=horizon,
                context_length=context_length,
            )

            ds = PandasDataset.from_long_dataframe(
                df,
                target="target",
                item_id="item_id",
                freq=frequency,
            )

            # Create predictor and run inference
            predictor = model.create_predictor(batch_size=1)
            forecasts = list(predictor.predict(ds))
            
            if forecasts:
                all_forecasts.append(forecasts[0].mean.tolist())
            else:
                all_forecasts.append([])

        # Return a flat list for single-series input
        if not is_batch:
            return all_forecasts[0]

        return all_forecasts
