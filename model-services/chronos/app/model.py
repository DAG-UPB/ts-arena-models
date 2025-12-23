import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Union, Any
from datetime import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"

class ChronosModel:
    def __init__(self) -> None:
        model_id = os.getenv("MODEL_ID", "amazon/chronos-bolt-base")
        
        if "chronos-2" in model_id:
            self.pipeline = Chronos2Pipeline.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype=torch.float32,
            )
        else:
            self.pipeline = BaseChronosPipeline.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype=torch.float32,
            )

    def predict(
            self,
            data: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
            horizon: int,
            freq: str = "h",
            past_covariates: Union[None, pd.DataFrame] = None,
            future_covariates: Union[None, pd.DataFrame] = None,
            quantile_levels: List[float] = [0.1, 0.5, 0.9]
        ) -> Dict[str, Any]:
        """
        Run forecasting with Chronos model.
        
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
            return {'forecasts': [], 'quantiles': {}}
        
        # Detect single-series vs. batch input
        is_batch = isinstance(data[0], list)
        
        if is_batch:
            data_as_batch = data
        else:
            data_as_batch = [data]
        
        # Build DataFrame from the data with timestamps
        dfs = []
        for idx, series_data in enumerate(data_as_batch):
            ts_values = [item["value"] for item in series_data]
            ts_timestamps = [item["ts"] for item in series_data]
            
            # Parse timestamps
            index = pd.to_datetime(ts_timestamps)
            
            df_seq = pd.DataFrame({
                "id": f"series_{idx}",
                "timestamp": index,
                "target": ts_values
            })
            dfs.append(df_seq)
        
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        
        
        # Für Chronos-2 mit DataFrames
        if isinstance(self.pipeline, Chronos2Pipeline):
            # Annahme: data, past_covariates, future_covariates sind DataFrames mit passenden Spalten
            pred_df = self.pipeline.predict_df(
                df,
                future_df=future_covariates,
                prediction_length=horizon,
                quantile_levels=quantile_levels,
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )
            
            if df['id'].nunique() > 1:
                # Multiple series
                forecasts_list = []
                quantiles_dict = {}
                for i, id_val in enumerate(df['id'].unique()):
                    series_pred = pred_df[pred_df['id'] == id_val]['0.5'].tolist()
                    forecasts_list.append(series_pred)
                    
                    # Extract quantiles for this series
                    series_quantiles = {}
                    for level in quantile_levels:
                        level_str = str(level)
                        if level_str in pred_df.columns:
                            series_quantiles[level_str] = pred_df[pred_df['id'] == id_val][level_str].tolist()
                    quantiles_dict[i] = series_quantiles
                
                return {'forecasts': forecasts_list, 'quantiles': quantiles_dict}
            else:
                # Single series
                series_pred = pred_df["0.5"].tolist() if "0.5" in pred_df else pred_df.iloc[:, -1].tolist()
                
                # Extract quantiles
                quantiles_dict = {}
                for level in quantile_levels:
                    level_str = str(level)
                    if level_str in pred_df.columns:
                        quantiles_dict[level_str] = pred_df[level_str].tolist()
                
                return {'forecasts': series_pred, 'quantiles': quantiles_dict}

        else:
            # Chronos-Bolt: Input als Tensor
            # Extract just the values for the tensor
            values_list = []
            for series_data in data_as_batch:
                values = [item["value"] for item in series_data]
                values_list.append(values)
            
            # Convert to tensor
            if len(values_list) > 1:
                # Batch case: pad sequences to same length
                max_len = max(len(seq) for seq in values_list)
                padded = []
                for seq in values_list:
                    pad_len = max_len - len(seq)
                    padded_seq = [float('nan')] * pad_len + seq
                    padded.append(padded_seq)
                arr = np.array(padded, dtype=np.float32)
                context = torch.tensor(arr, dtype=torch.float32).to(device)
            else:
                # Single series
                arr = np.array(values_list[0], dtype=np.float32)
                context = torch.tensor(arr.reshape(1, -1), dtype=torch.float32).to(device)
            
            # Get predictions
            quantiles, mean = self.pipeline.predict_quantiles(
                inputs=context,
                prediction_length=horizon,
                quantile_levels=quantile_levels
            )
            
            # mean.shape is (batch, horizon)
            result = mean.tolist()
            
            if is_batch:
                # Multiple series
                forecasts_list = result  # Already a list of lists
                
                # Extract quantiles for each series
                quantiles_dict = {}
                for i in range(len(result)):
                    series_quantiles = {}
                    for j, level in enumerate(quantile_levels):
                        # quantiles shape is (batch, horizon, num_quantiles)
                        series_quantiles[str(level)] = quantiles[i, :, j].tolist()
                    quantiles_dict[i] = series_quantiles
                
                return {'forecasts': forecasts_list, 'quantiles': quantiles_dict}
            else:
                # Single series
                forecasts = result[0]
                
                # Extract quantiles
                quantiles_dict = {}
                for j, level in enumerate(quantile_levels):
                    # quantiles shape is (batch, horizon, num_quantiles)
                    quantiles_dict[str(level)] = quantiles[0, :, j].tolist()
                
                return {'forecasts': forecasts, 'quantiles': quantiles_dict}

