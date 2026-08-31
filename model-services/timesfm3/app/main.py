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
from .model import TimesFMModel
import torch

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


def create_forecast_items(timestamps: List[str], values: List[float]) -> List[ForecastItem]:
    """Create ForecastItem list from timestamps and values"""
    return [
        ForecastItem(ts=ts, value=float(val), probabilistic_values={})
        for ts, val in zip(timestamps, values)
    ]


# The nine deciles the arena's probabilistic contract asks for (see the repo README,
# "Probabilistic forecasts"). TimesFM 3.0's checkpoint happens to predict exactly
# these, but we resolve them by level rather than by position: the 2.5 head returned
# ten columns with the mean at index 0, so an assumed offset is a silent, whole-model
# quantile shift waiting to happen.
REQUIRED_LEVELS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]


def build_quantile_index(model_levels: List[float]) -> Optional[Dict[str, int]]:
    """Map each required decile key to its column in the model's quantile output.

    Returns None if the model does not predict all nine deciles — in that case the
    service reports point forecasts only rather than inventing the missing levels
    (models #13).
    """
    index = {}
    for level in REQUIRED_LEVELS:
        matches = [i for i, q in enumerate(model_levels) if abs(q - float(level)) < 1e-9]
        if not matches:
            logger.warning(
                f"Model does not predict quantile level {level} "
                f"(has {model_levels}); serving point forecasts only."
            )
            return None
        index[f"q_{level}"] = matches[0]
    return index


def extract_quantiles(
    series_quantiles, horizon: int, q_index: Optional[Dict[str, int]]
) -> List[Dict[str, float]]:
    """Turn one series' (horizon, n_quantiles) block into per-step decile dicts."""
    if q_index is None or series_quantiles is None:
        return [{} for _ in range(horizon)]
    out = []
    for h in range(horizon):
        q_values = series_quantiles[h]
        out.append({key: float(q_values[i]) for key, i in q_index.items()})
    return out


app = FastAPI()
try:
    model = TimesFMModel()
    QUANTILE_INDEX = build_quantile_index(model.quantile_levels)
except Exception as e:
    logger.error(f"Failed to initialize model: {e}")
    model = None
    QUANTILE_INDEX = None


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
        
    if not request.history:
        raise HTTPException(status_code=400, detail="History cannot be empty")
    
    logger.info(f"Received prediction request with horizon {request.horizon}")
    
    freq = request.freq or "h"
    is_batch = isinstance(request.history[0], list)
    
    try:
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
            point_predictions, quantile_predictions = model.predict(history_values, request.horizon)
            
            # Create ForecastItems for each series
            all_forecasts = []
            for i, (pred_values, last_ts) in enumerate(zip(point_predictions, last_timestamps)):
                future_ts = generate_future_timestamps(last_ts, request.horizon, freq)
                
                # quantile_predictions shape: (batch, horizon, n_quantiles)
                series_quantiles = (
                    quantile_predictions[i] if quantile_predictions is not None else None
                )
                quantiles_dict_list = extract_quantiles(
                    series_quantiles, len(future_ts), QUANTILE_INDEX
                )

                forecasts = []
                for h, (ts, q_dict) in enumerate(zip(future_ts, quantiles_dict_list)):
                    # Use q_0.5 (median) as point forecast for consistency
                    point_val = q_dict.get("q_0.5", float(pred_values[h])) if q_dict else float(pred_values[h])
                    forecasts.append(ForecastItem(ts=ts, value=point_val, probabilistic_values=q_dict))
                
                all_forecasts.append(forecasts)
            
            return {"prediction": all_forecasts}
        else:
            # Single series
            history_values = [item.value for item in request.history]
            last_ts = datetime.fromisoformat(request.history[-1].ts.replace('Z', '+00:00').replace('+00:00', ''))
            
            # Get prediction from model
            point_prediction, quantile_prediction = model.predict(history_values, request.horizon)
            
            # Create ForecastItems
            future_ts = generate_future_timestamps(last_ts, request.horizon, freq)
            
            # quantile_prediction shape: (horizon, n_quantiles) for a single series
            quantiles_dict_list = extract_quantiles(
                quantile_prediction, len(future_ts), QUANTILE_INDEX
            )

            forecasts = []
            for h, (ts, q_dict) in enumerate(zip(future_ts, quantiles_dict_list)):
                # Use q_0.5 (median) as point forecast for consistency
                point_val = q_dict.get("q_0.5", float(point_prediction[h])) if q_dict else float(point_prediction[h])
                forecasts.append(ForecastItem(ts=ts, value=point_val, probabilistic_values=q_dict))
            
            return {"prediction": forecasts}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
        
    try:
        prediction = model.predict([1,2,3,4,5], horizon=1)
        # Model loading check
        if prediction is not None:
            return {"status": "healthy", "model": "ready"}
        else:
            raise HTTPException(status_code=503, detail="Model not ready")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")
    

@app.get("/gpu-check")
def gpu_check():
    return {
        "gpu_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }
