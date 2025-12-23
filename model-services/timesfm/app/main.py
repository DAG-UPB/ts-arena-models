from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Dict, Optional
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from .model import TimesFMModel
import torch

class HistoryItem(BaseModel):
    ts: str
    value: float

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


app = FastAPI()
model = TimesFMModel()


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
        predictions = model.predict(history_values, request.horizon)
        
        # Create ForecastItems for each series
        all_forecasts = []
        for pred_values, last_ts in zip(predictions, last_timestamps):
            future_ts = generate_future_timestamps(last_ts, request.horizon, freq)
            forecasts = create_forecast_items(future_ts, pred_values)
            all_forecasts.append(forecasts)
        
        return {"prediction": all_forecasts}
    else:
        # Single series
        history_values = [item.value for item in request.history]
        last_ts = datetime.fromisoformat(request.history[-1].ts.replace('Z', '+00:00').replace('+00:00', ''))
        
        # Get prediction from model
        prediction = model.predict(history_values, request.horizon)
        
        # Create ForecastItems
        future_ts = generate_future_timestamps(last_ts, request.horizon, freq)
        forecasts = create_forecast_items(future_ts, prediction)
        
        return {"prediction": forecasts}


@app.get("/health")
async def health_check():
    try:
        prediction = model.predict([1,2,3,4,5], horizon=1)
        # Model loading check
        if prediction is not None:
            return {"status": "healthy", "model": "ready"}
        else:
            raise HTTPException(status_code=503, detail="Model not ready")
    except Exception:
        raise HTTPException(status_code=503, detail="Service unhealthy")
    

@app.get("/gpu-check")
def gpu_check():
    return {
        "gpu_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }
