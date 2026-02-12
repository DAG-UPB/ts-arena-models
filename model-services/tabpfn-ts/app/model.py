import numpy as np
import pandas as pd
from typing import List, Dict, Union, Any
import logging
import os

from tabpfn_time_series import TabPFNTSPipeline, TabPFNMode

logger = logging.getLogger(__name__)


class TabPFNTSModel:
    def __init__(self) -> None:
        """
        Initializes the TabPFN-TS pipeline in local mode.
        """
        max_context_length = int(os.getenv("MAX_CONTEXT_LENGTH", "4096"))

        logger.info("Loading TabPFN-TS pipeline in local mode...")
        self.pipeline = TabPFNTSPipeline(
            tabpfn_mode=TabPFNMode.LOCAL,
            max_context_length=max_context_length,
        )
        logger.info("TabPFN-TS pipeline loaded successfully")

    def predict(
        self,
        data: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        horizon: int,
        freq: str = "h",
        past_covariates: Union[None, pd.DataFrame] = None,
        future_covariates: Union[None, pd.DataFrame] = None,
        quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ) -> Dict[str, Any]:
        """
        Run forecasting with TabPFN-TS pipeline.

        Args:
            data: Either a single time series as a list of dicts [{"ts": <timestamp>, "value": <float>}, ...]
                  or a batch of time series as a list of such lists.
            horizon: Number of forecast steps (prediction_length).
            freq: Pandas frequency string, e.g. "h", "D", "1min".
            quantile_levels: List of quantile levels for probabilistic forecasting.

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
                context_df=context_df,
                prediction_length=horizon,
                quantiles=quantile_levels,
            )
        except Exception as e:
            logger.error(f"TabPFN-TS prediction failed: {e}", exc_info=True)
            raise

        # Parse results back into the expected format
        if is_batch:
            all_forecasts = []
            all_quantiles = {}

            for idx, series_data in enumerate(data_as_batch):
                item_id = f"series_{idx}"
                # pred_df is indexed by (item_id, timestamp)
                item_pred = pred_df.loc[item_id]

                # Point forecasts from 'target' column
                forecasts = item_pred["target"].tolist()
                all_forecasts.append(forecasts)

                # Quantiles: 0.1, 0.2, ...
                quantile_dict = {}
                for q in quantile_levels:
                    if q in item_pred.columns:
                        quantile_dict[str(q)] = item_pred[q].tolist()
                all_quantiles[idx] = quantile_dict

            return {"forecasts": all_forecasts, "quantiles": all_quantiles}
        else:
            # Single series
            # pred_df may have a dummy item_id level
            if isinstance(pred_df.index, pd.MultiIndex):
                pred_df = pred_df.droplevel(0)

            forecasts = pred_df["target"].tolist()
            quantile_dict = {}
            for q in quantile_levels:
                if q in pred_df.columns:
                    quantile_dict[str(q)] = pred_df[q].tolist()

            return {"forecasts": forecasts, "quantiles": quantile_dict}
