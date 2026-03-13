import numpy as np
import pandas as pd
from typing import List, Dict, Union, Any
import logging
import os

from tabpfn_time_series import (
    TabPFNTSPipeline,
    TabPFNMode,
    DEFAULT_QUANTILE_CONFIG,
)

logger = logging.getLogger(__name__)

QUANTILE_LEVELS = DEFAULT_QUANTILE_CONFIG  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class TabPFNTSModel:
    def __init__(self) -> None:
        """
        Initializes the TabPFN-TS pipeline in local mode.
        """
        max_context_length = int(os.getenv("MAX_CONTEXT_LENGTH", "4096"))

        logger.info("Loading TabPFN-TS pipeline in local mode...")
        self.pipeline = TabPFNTSPipeline(
            max_context_length=max_context_length,
            tabpfn_mode=TabPFNMode.LOCAL,
        )
        logger.info("TabPFN-TS pipeline loaded successfully")

    def predict(
        self,
        data: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        horizon: int,
        freq: str = "h",
    ) -> Dict[str, Any]:
        """
        Run forecasting with TabPFN-TS pipeline.

        Args:
            data: Either a single time series as a list of dicts [{"ts": <timestamp>, "value": <float>}, ...]
                  or a batch of time series as a list of such lists.
            horizon: Number of forecast steps (prediction_length).
            freq: Pandas frequency string, e.g. "h", "D", "1min".

        Returns:
            Dictionary with:
                - 'forecasts': List[float] for single series or List[List[float]] for batch
                - 'quantiles': Optional dict with quantile predictions
        """
        if not data:
            return {"forecasts": [], "quantiles": {}}

        # Detect single-series vs. batch input
        is_batch = isinstance(data[0], list)

        if is_batch:
            data_as_batch = data
        else:
            data_as_batch = [data]

        # Build a combined context DataFrame with item_id, timestamp, target
        all_rows = []
        for idx, series_data in enumerate(data_as_batch):
            item_id = f"series_{idx}"
            for item in series_data:
                all_rows.append(
                    {
                        "item_id": item_id,
                        "timestamp": pd.Timestamp(item["ts"]),
                        "target": float(item["value"]),
                    }
                )

        context_df = pd.DataFrame(all_rows)
        # Ensure timestamp column is timezone-naive datetime64 (required by TabPFN-TS TimeSeriesDataFrame)
        context_df["timestamp"] = pd.to_datetime(context_df["timestamp"], utc=True).dt.tz_localize(None)

        # Run prediction via predict_df
        try:
            pred_df = self.pipeline.predict_df(
            context_df,
            prediction_length=horizon,
            quantiles=QUANTILE_LEVELS,
        )
        except Exception as e:
            logger.error(f"TabPFN-TS prediction failed: {e}", exc_info=True)
            raise

        # Parse results back into the expected format
        if is_batch:
            all_forecasts = []
            all_quantiles = {}
            for idx in range(len(data_as_batch)):
                item_id = f"series_{idx}"
                item_pred = pred_df.loc[item_id] if item_id in pred_df.index.get_level_values(0) else pred_df
                all_forecasts.append(item_pred["target"].tolist())
                quantile_dict = {}
                for q in QUANTILE_LEVELS:
                    col = str(q)
                    if col in item_pred.columns:
                        quantile_dict[col] = item_pred[col].tolist()
                all_quantiles[idx] = quantile_dict
            return {"forecasts": all_forecasts, "quantiles": all_quantiles}
        else:
            if isinstance(pred_df.index, pd.MultiIndex):
                pred_df = pred_df.droplevel(0)
            forecasts = pred_df["target"].tolist()
            quantile_dict = {}
            for q in QUANTILE_LEVELS:
                col = str(q)
                if col in pred_df.columns:
                    quantile_dict[col] = pred_df[col].tolist()
            return {"forecasts": forecasts, "quantiles": quantile_dict}
