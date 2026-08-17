from __future__ import annotations

# --- logging (ts-arena #15) ---------------------------------------------
# Configure the ROOT logger, not just this module's. Left unconfigured, root keeps
# its default level WARNING with no handler at all: every logger.info() below is
# dropped before the record is even built, and WARNING+ escapes through
# logging.lastResort as bare text with no timestamp, no level and no logger name.
# That is the defect that hid the ELO job for ten days in backend #75. Format and
# LOG_LEVEL semantics match ts-arena-backend's logging_setup.py so the whole fleet
# reads alike. Must stay above the `.model` import, which logs at import time.
import logging
import os
import sys
import time

_LOG_LEVELS = {"CRITICAL": logging.CRITICAL, "FATAL": logging.CRITICAL,
               "ERROR": logging.ERROR, "WARNING": logging.WARNING,
               "WARN": logging.WARNING, "INFO": logging.INFO,
               "DEBUG": logging.DEBUG}
logging.Formatter.converter = time.gmtime  # asctime in UTC, hence the trailing Z
logging.basicConfig(
    level=_LOG_LEVELS.get(os.getenv("LOG_LEVEL", "").strip().upper(), logging.INFO),
    format="%(asctime)sZ | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------

from typing import List, Union, Dict, Optional
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .model import NeuralForecastManager

# Pydantic models
class HistoryItem(BaseModel):
    ts: str
    value: Optional[float] = None

class PredictionRequest(BaseModel):
    history: Union[List[List[HistoryItem]], List[HistoryItem]]
    horizon: int
    freq: Optional[str] = "h"

class ForecastItem(BaseModel):
    ts: str
    value: float
    probabilistic_values: Dict[str, float] = {}

class PredictionResponse(BaseModel):
    prediction: Union[List[ForecastItem], List[List[ForecastItem]]]

# Helper functions
def generate_future_timestamps(last_timestamp: datetime, horizon: int, freq: str) -> List[str]:
    """Generate future timestamps based on the last known timestamp"""
    timestamps = []
    for i in range(1, horizon + 1):
        if freq == "1min":
            next_time = last_timestamp + timedelta(minutes=1*i)
        elif freq == "15min":
            next_time = last_timestamp + timedelta(minutes=15*i)
        elif freq == "30min":
            next_time = last_timestamp + timedelta(minutes=30*i)
        elif freq == "h":
            next_time = last_timestamp + timedelta(hours=i)
        elif freq == "D":
            next_time = last_timestamp + timedelta(days=i)
        elif freq == "W":
            next_time = last_timestamp + timedelta(weeks=i)
        elif freq == "M":
            next_time = last_timestamp + relativedelta(months=i)
        else:
            # Default to hourly if unknown or complex
            next_time = last_timestamp + timedelta(hours=i)
        
        if freq in ["D", "W", "M"]:
            next_time = next_time.replace(hour=0, minute=0, second=0, microsecond=0)
        
        timestamps.append(next_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"))
    return timestamps

def create_forecast_items(timestamps: List[str], values: List[float]) -> List[ForecastItem]:
    """Create ForecastItem list from timestamps and values"""
    return [
        ForecastItem(ts=ts, value=float(val), probabilistic_values={})
        for ts, val in zip(timestamps, values)
    ]

# App setup
MODELS_DIR = os.getenv("MODELS_DIR", "/models")
os.makedirs(MODELS_DIR, exist_ok=True)

app = FastAPI(title="NeuralForecast Service")
manager = NeuralForecastManager(models_dir=MODELS_DIR)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    logger.info(f"Received prediction request. Horizon: {request.horizon}, Freq: {request.freq}, Batch size: {len(request.history) if isinstance(request.history, list) and isinstance(request.history[0], list) else 1}")
    if not request.history:
        logger.warning("Received empty history in request")
        raise HTTPException(status_code=400, detail="History cannot be empty")
    
    freq = request.freq or "h"
    is_batch = isinstance(request.history[0], list)
    
    # Prepare data for NeuralForecast
    data_list = []
    
    if is_batch:
        series_list = request.history
    else:
        series_list = [request.history]
        
    for idx, series in enumerate(series_list):
        if not series:
            continue
        
        for item in series:
            data_list.append({
                "unique_id": str(idx),
                "ds": item.ts, 
                "y": item.value
            })

    if not data_list:
        logger.error("No valid history data provided after processing")
        raise HTTPException(status_code=400, detail="No valid history data provided")

    # Convert to DataFrame
    df = pd.DataFrame(data_list)
    # Ensure ds is datetime
    df['ds'] = pd.to_datetime(df['ds'])
    
    logger.info(f"Prepared DataFrame with {len(df)} rows. Starting training/prediction...")

    # Fit and Predict
    try:
        fcst_df = manager.fit_and_predict(
            df=df,
            freq=freq,
            h=request.horizon
        )
        logger.info("Prediction completed successfully")
    except Exception as e:
        logger.error(f"Training/Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training/Prediction failed: {str(e)}")
    
    # Identify prediction column (it's the one that is not unique_id or ds)
    pred_cols = [c for c in fcst_df.columns if c not in ['unique_id', 'ds']]
    if not pred_cols:
        raise HTTPException(status_code=500, detail="No prediction column found in result")
    pred_col = pred_cols[0]
    
    all_forecasts = []
    
    for idx in range(len(series_list)):
        uid = str(idx)
        series_pred = fcst_df[fcst_df['unique_id'] == uid]
        
        if series_pred.empty:
            # Should not happen if training worked
            all_forecasts.append([])
            continue

        # Sort by ds
        series_pred = series_pred.sort_values('ds')
        
        pred_values = series_pred[pred_col].values.tolist()
        
        # Use timestamps from the model output
        timestamps = [pd.to_datetime(ts).strftime("%Y-%m-%dT%H:%M:%S.000Z") for ts in series_pred['ds']]
        
        forecasts = create_forecast_items(timestamps, pred_values)
        all_forecasts.append(forecasts)

    if is_batch:
        return {"prediction": all_forecasts}
    else:
        return {"prediction": all_forecasts[0]}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "on-demand"}
