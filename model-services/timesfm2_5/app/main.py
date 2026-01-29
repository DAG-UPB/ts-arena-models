from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Dict, Optional
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from .model import TimesFMModel
import torch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
try:
    model = TimesFMModel()
except Exception as e:
    logger.error(f"Failed to initialize model: {e}")
    model = None


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
                
                # Extract quantiles if available
                quantiles_dict_list = []
                if quantile_predictions is not None:
                    # quantile_predictions shape: (batch, horizon, 10)
                    # We need to iterate over the horizon for the current series
                    series_quantiles = quantile_predictions[i] # shape (horizon, 10)
                    
                    for h in range(len(future_ts)):
                        q_values = series_quantiles[h]
                        # Map to quantile names: q_0.1, q_0.2, ..., q_0.9
                        q_dict = {
                            "q_0.1": float(q_values[1]),
                            "q_0.2": float(q_values[2]),
                            "q_0.3": float(q_values[3]),
                            "q_0.4": float(q_values[4]),
                            "q_0.5": float(q_values[5]),
                            "q_0.6": float(q_values[6]),
                            "q_0.7": float(q_values[7]),
                            "q_0.8": float(q_values[8]),
                            "q_0.9": float(q_values[9])
                        }
                        quantiles_dict_list.append(q_dict)
                else:
                    quantiles_dict_list = [{}] * len(future_ts)

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
            
            quantiles_dict_list = []
            if quantile_prediction is not None:
                 # quantile_prediction shape: (horizon, 10) for single series
                 for h in range(len(future_ts)):
                    q_values = quantile_prediction[h]
                    q_dict = {
                        "q_0.1": float(q_values[1]),
                        "q_0.2": float(q_values[2]),
                        "q_0.3": float(q_values[3]),
                        "q_0.4": float(q_values[4]),
                        "q_0.5": float(q_values[5]),
                        "q_0.6": float(q_values[6]),
                        "q_0.7": float(q_values[7]),
                        "q_0.8": float(q_values[8]),
                        "q_0.9": float(q_values[9])
                    }
                    quantiles_dict_list.append(q_dict)
            else:
                quantiles_dict_list = [{}] * len(future_ts)

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
