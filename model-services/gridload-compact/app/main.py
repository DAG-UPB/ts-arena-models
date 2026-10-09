from __future__ import annotations

# --- logging: same root-logger setup as the other ts-arena-models services --------------
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
# -------------------------------------------------------------------------------------

from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .model import GridLoadCompact


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


def _fmt(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _items(series: dict) -> List[ForecastItem]:
    items = []
    for i, ts in enumerate(series["ts"]):
        pv = {f"q_{lv}": float(series["quantiles"][lv][i]) for lv in series["quantiles"]}
        items.append(ForecastItem(ts=_fmt(ts), value=float(series["forecasts"][i]), probabilistic_values=pv))
    return items


app = FastAPI()
model = GridLoadCompact()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not request.history:
        raise HTTPException(status_code=400, detail="History cannot be empty")
    if request.horizon < 1:
        raise HTTPException(status_code=400, detail="Horizon must be at least 1")
    freq = request.freq or "h"
    is_batch = isinstance(request.history[0], list)
    hist = [[{"ts": it.ts, "value": it.value} for it in s] for s in request.history] if is_batch \
        else [{"ts": it.ts, "value": it.value} for it in request.history]
    try:
        result = model.predict(hist, request.horizon, freq)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    forecasts = [_items(s) for s in result["series"]]
    return {"prediction": forecasts if is_batch else forecasts[0]}


@app.get("/health")
async def health_check():
    try:
        now = int(time.time())
        data = [{"ts": _fmt(now - i * 3600), "value": float(i + 1)} for i in range(5, 0, -1)]
        result = model.predict(data, 1, "h")
        if result is not None and result.get("series"):
            return {"status": "healthy", "model": "ready"}
        raise HTTPException(status_code=503, detail="Model not ready")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")
