import torch
import timesfm
import numpy as np
import os
import logging
from typing import List, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Set float32 matmul precision as recommended for performance
torch.set_float32_matmul_precision("high")

class TimesFMModel:
    def __init__(self) -> None:
        self.model_id = os.getenv("MODEL_ID", "google/timesfm-2.5-200m-pytorch")
        logger.info(f"Loading model from {self.model_id}...")
        
        # Check if we have a local model path mapped
        # Assuming standard mapping /models/<repo_name>
        local_dir = Path(f"/models/{self.model_id.split('/')[-1]}")
        
        load_path = self.model_id
        if local_dir.exists() and any(local_dir.iterdir()):
             logger.info(f"Found local model at {local_dir}")
             load_path = str(local_dir)
        else:
             logger.info(f"Model not found locally at {local_dir}, using {self.model_id} (will download if not cached)...")

        # Initialize model
        # Using the 200M class as per instructions for 2.5-200m
        try:
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                load_path,
                torch_compile=True
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Default config
        self.max_context = 2048
        self.max_horizon = 256
        
        self.config = timesfm.ForecastConfig(
            max_context=self.max_context,
            max_horizon=self.max_horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
        
        logger.info("Compiling model...")
        try:
            self.model.compile(self.config)
            # self.model.eval() # Removed as it causes AttributeError
            logger.info("Model loaded and compiled.")
        except Exception as e:
            logger.error(f"Failed to compile model: {e}")
            raise

    def predict(self, history: Union[List[float], List[List[float]]], horizon: int):
        if not history:
            raise ValueError("History cannot be empty.")

        is_single_series = not history or not isinstance(history[0], list)
        if is_single_series:
            history = [history]

        # Check if we need to recompile for larger horizon
        if horizon > self.max_horizon:
            logger.info(f"Requested horizon {horizon} > max_horizon {self.max_horizon}. Recompiling...")
            self.max_horizon = horizon
            self.config = timesfm.ForecastConfig(
                max_context=self.max_context,
                max_horizon=self.max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
            self.model.compile(self.config)

        processed_history = []
        for series in history:
            # Truncate if too long
            if len(series) > self.max_context:
                series = series[-self.max_context:]
            processed_history.append(np.array(series))
            
        point_forecast, quantile_forecast = self.model.forecast(
            horizon=horizon,
            inputs=processed_history
        )
        
        # Convert to list
        if hasattr(point_forecast, 'tolist'):
            point_forecast = point_forecast.tolist()
        elif isinstance(point_forecast, np.ndarray):
            point_forecast = point_forecast.tolist()
            
        if quantile_forecast is not None:
             if hasattr(quantile_forecast, 'tolist'):
                quantile_forecast = quantile_forecast.tolist()
             elif isinstance(quantile_forecast, np.ndarray):
                quantile_forecast = quantile_forecast.tolist()

        if is_single_series:
            return point_forecast[0], quantile_forecast[0] if quantile_forecast else None
        return point_forecast, quantile_forecast
