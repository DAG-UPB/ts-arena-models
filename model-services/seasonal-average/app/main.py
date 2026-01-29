from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Dict, Optional
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from .model import SeasonalAverageModel

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


def infer_seasonality_from_freq(freq: str) -> int:
    """
    Infer the seasonality based on the frequency of the time series.
    Returns the most common seasonal pattern for each frequency.
    """
    seasonality_map = {
        "1min": 60,      # Hourly pattern (60 minutes)
        "15min": 96,     # Daily pattern (24*60/15 = 96 intervals per day)
        "30min": 48,     # Daily pattern (24*60/30 = 48 intervals per day)
        "h": 24,         # Daily pattern (24 hours per day)
        "D": 7,          # Weekly pattern (7 days per week)
        "W": 52,         # Yearly pattern (52 weeks per year)
        "M": 12,         # Yearly pattern (12 months per year)
    }
    return seasonality_map.get(freq, 24)  # Default to 24 if unknown frequency


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

num_seasons_env = os.environ.get("SEASONAL_SEASONS")
if num_seasons_env:
    num_seasons = int(num_seasons_env)
else:
    num_seasons = 3

model = SeasonalAverageModel(num_seasons=num_seasons)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not request.history:
        raise HTTPException(status_code=400, detail="History cannot be empty")
    
    freq = request.freq or "h"
    seasonality = infer_seasonality_from_freq(freq)
    is_batch = isinstance(request.history[0], list)
    
    if is_batch:
        history_values = []
        last_timestamps = []
        offsets = []
        for series in request.history:
            # Parse last timestamp
            last_ts = datetime.fromisoformat(series[-1].ts.replace('Z', '+00:00').replace('+00:00', ''))
            
            values = [item.value for item in series]
            history_values.append(values)
            last_timestamps.append(last_ts)
            offsets.append(0)
        
        # Get predictions from model
        predictions = model.predict(history_values, request.horizon, seasonality, offset=offsets)
        
        # Create ForecastItems for each series
        all_forecasts = []
        for pred_values, last_ts in zip(predictions, last_timestamps):
            future_ts = generate_future_timestamps(last_ts, request.horizon, freq)
            forecasts = create_forecast_items(future_ts, pred_values)
            all_forecasts.append(forecasts)
        
        return {"prediction": all_forecasts}
    else:
        # Single series
        series = request.history
        last_ts = datetime.fromisoformat(series[-1].ts.replace('Z', '+00:00').replace('+00:00', ''))
        
        history_values = [item.value for item in series]
        
        # Get prediction from model
        prediction = model.predict(history_values, request.horizon, seasonality, offset=0)
        
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
