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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Dict, Optional
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from .model import SimpleMovingAverageModel


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
            next_time = last_timestamp + timedelta(hours=i)
        
        if freq in ["D", "W", "M"]:
            next_time = next_time.replace(hour=0, minute=0, second=0, microsecond=0)
        
        timestamps.append(next_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"))
    return timestamps


def create_forecast_items(
    timestamps: List[str],
    values: List[float],
    quantiles_dict: Optional[Dict[str, List[float]]] = None,
) -> List[ForecastItem]:
    """Create ForecastItem list from timestamps, values, and optional quantiles"""
    forecasts = []
    for i, (ts, val) in enumerate(zip(timestamps, values)):
        probabilistic_values = {}
        if quantiles_dict:
            for level, quantile_values in quantiles_dict.items():
                if i < len(quantile_values):
                    probabilistic_values[f"q_{level}"] = float(quantile_values[i])
        forecasts.append(
            ForecastItem(
                ts=ts,
                value=float(val),
                probabilistic_values=probabilistic_values,
            )
        )
    return forecasts


app = FastAPI()
model = SimpleMovingAverageModel()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not request.history:
        raise HTTPException(status_code=400, detail="History cannot be empty")
    
    freq = request.freq or "h"
    is_batch = isinstance(request.history[0], list)
    
    if is_batch:
        history_values = []
        last_timestamps = []
        for series in request.history:
            values = [item.value for item in series]
            history_values.append(values)
            # Parse last timestamp
            last_ts = datetime.fromisoformat(series[-1].ts.replace('Z', '+00:00').replace('+00:00', ''))
            last_timestamps.append(last_ts)
        
        # Get predictions from model
        result = model.predict(history_values, request.horizon)
        predictions = result["forecasts"]
        quantiles_list = result.get("quantiles", [])
        
        # Create ForecastItems for each series
        all_forecasts = []
        for i, last_ts in enumerate(last_timestamps):
            future_ts = generate_future_timestamps(last_ts, request.horizon, freq)
            pred_values = predictions[i]
            quantiles_dict = quantiles_list[i] if i < len(quantiles_list) else None
            forecasts = create_forecast_items(future_ts, pred_values, quantiles_dict)
            all_forecasts.append(forecasts)
        
        return {"prediction": all_forecasts}
    else:
        # Single series
        history_values = [item.value for item in request.history]
        last_ts = datetime.fromisoformat(request.history[-1].ts.replace('Z', '+00:00').replace('+00:00', ''))
        
        # Get prediction from model
        result = model.predict(history_values, request.horizon)
        prediction = result["forecasts"]
        quantiles_dict = result.get("quantiles")
        
        # Create ForecastItems
        future_ts = generate_future_timestamps(last_ts, request.horizon, freq)
        forecasts = create_forecast_items(future_ts, prediction, quantiles_dict)
        
        return {"prediction": forecasts}


@app.get("/health")
async def health_check():
    try:
        result = model.predict([1, 2, 3, 4, 5], horizon=1)
        if result is not None and "forecasts" in result:
            return {"status": "healthy", "model": "ready"}
        else:
            raise HTTPException(status_code=503, detail="Model not ready")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")