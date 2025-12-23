import os
import threading
import logging
from typing import List, Dict, Optional, Any, Sequence

import pandas as pd
from fastapi import HTTPException

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, TSMixerx, PatchTST, TFT, TiDE
from neuralforecast.auto import AutoNHITS, AutoTSMixerx, AutoPatchTST, AutoTFT, AutoTiDE
from neuralforecast.losses.pytorch import MQLoss

logger = logging.getLogger(__name__)



class NeuralForecastManager:
    """
    Encapsulates training, persistence, loading, and forecasting for various NeuralForecast models.

    - Maintains an in-memory cache of loaded models
    - Saves/loads models under models_dir
    - Provides helpers to convert request items into DataFrames
    - Dynamically selects model type based on MODEL_TYPE environment variable
    """

    def __init__(self, models_dir: str) -> None:
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        # Support MODEL_ID (like other containers) or fallback to MODEL_TYPE
        model_id = os.getenv('MODEL_ID', 'nixtla/nhits')
        self.model_type = model_id.split('/')[-1].lower()
        self._loaded_models: Dict[str, NeuralForecast] = {}
        self._lock = threading.Lock()

    # -----------------------------
    # DataFrame helpers
    # -----------------------------
    @staticmethod
    def to_df(items: Sequence[Any], include_y: bool) -> pd.DataFrame:
        """Convert a list of objects with attributes unique_id, ds, y, exog into a DataFrame."""
        recs: List[Dict[str, Any]] = []
        for it in items:
            # Supports Pydantic objects as well as dict-like structures
            unique_id = getattr(it, "unique_id", None) if not isinstance(it, dict) else it.get("unique_id")
            ds = getattr(it, "ds", None) if not isinstance(it, dict) else it.get("ds")
            y = getattr(it, "y", None) if not isinstance(it, dict) else it.get("y")
            exog = getattr(it, "exog", {}) if not isinstance(it, dict) else it.get("exog", {})
            # Validate required fields
            if unique_id is None:
                raise HTTPException(status_code=400, detail="'unique_id' must not be null")
            if ds is None:
                raise HTTPException(status_code=400, detail="'ds' (timestamp) must not be null")
            # Convert timestamp (raises if invalid)
            ts = pd.to_datetime(ds)
            row: Dict[str, Any] = {"unique_id": unique_id, "ds": ts}
            if include_y:
                if y is None:
                    raise HTTPException(status_code=400, detail="'y' must not be null in training data")
                row["y"] = y
            if isinstance(exog, dict):
                # Do not overwrite reserved columns
                reserved = {"unique_id", "ds", "y"}
                filtered = {k: v for k, v in exog.items() if k not in reserved}
                row.update(filtered)
            recs.append(row)
        df = pd.DataFrame(recs)
        expected_cols = ["unique_id", "ds"] + (["y"] if include_y else [])
        for c in expected_cols:
            if c not in df.columns:
                raise HTTPException(status_code=400, detail=f"Missing column '{c}' in data: {df.head()} from items {items[:3]}")
        return df

    @staticmethod
    def static_df_from(df: pd.DataFrame, stat_cols: List[str]) -> Optional[pd.DataFrame]:
        if not stat_cols:
            return None
        return df.groupby("unique_id", as_index=False)[stat_cols].first()

    @staticmethod
    def static_df_from_rows(rows: Optional[List[Dict[str, Any]]]) -> Optional[pd.DataFrame]:
        if not rows:
            return None
        sdf = pd.DataFrame(rows)
        if "unique_id" not in sdf.columns:
            raise HTTPException(status_code=400, detail="static rows must include 'unique_id'")
        return sdf

    # -----------------------------
    # Persistence / cache
    # -----------------------------
    def _model_path(self, name: str) -> str:
        safe = name.strip().replace("/", "_")
        return os.path.join(self.models_dir, safe)

    def load_or_get(self, name: str) -> NeuralForecast:
        path = self._model_path(name)
        if not os.path.isdir(path):
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found at {path}")
        with self._lock:
            if name in self._loaded_models:
                return self._loaded_models[name]
            nf = NeuralForecast.load(path=path)
            self._loaded_models[name] = nf
            return nf

    def _build_model(
        self,
        h: int,
        input_size: int,
        use_auto: bool,
        futr_exog: List[str],
        hist_exog: List[str],
        stat_exog: List[str],
        num_samples: int,
        n_series: int = 1,
    ):
        if use_auto:
            def config_space(trial):
                return {
                    "input_size": input_size,
                    "max_steps": 400,
                    "learning_rate": 1e-3,
                    "futr_exog_list": futr_exog,
                    "hist_exog_list": hist_exog,
                    "stat_exog_list": stat_exog,
                    "scaler_type": "standard",
                }

            if self.model_type == 'nhits':
                return AutoNHITS(
                    h=h,
                    config=config_space,
                    loss=MQLoss(),
                    backend="optuna",
                    num_samples=max(1, num_samples),
                )
            elif self.model_type == 'tsmixerx':
                return AutoTSMixerx(
                    h=h,
                    config=config_space,
                    loss=MQLoss(),
                    backend="optuna",
                    num_samples=max(1, num_samples),
                )
            elif self.model_type == 'patchtst':
                return AutoPatchTST(
                    h=h,
                    config=config_space,
                    loss=MQLoss(),
                    backend="optuna",
                    num_samples=max(1, num_samples),
                )
            elif self.model_type == 'tft':
                return AutoTFT(
                    h=h,
                    config=config_space,
                    loss=MQLoss(),
                    backend="optuna",
                    num_samples=max(1, num_samples),
                )
            elif self.model_type == 'tide':
                return AutoTiDE(
                    h=h,
                    config=config_space,
                    loss=MQLoss(),
                    backend="optuna",
                    num_samples=max(1, num_samples),
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported auto model type: {self.model_type}")
        else:
            if self.model_type == 'nhits':
                return NHITS(
                    h=h,
                    input_size=input_size,
                    # max_steps=400,
                    # learning_rate=1e-3,
                    futr_exog_list=futr_exog,
                    hist_exog_list=hist_exog,
                    stat_exog_list=stat_exog,
                    scaler_type = "standard",
                )
            elif self.model_type == 'tsmixerx':
                return TSMixerx(
                    h=h,
                    input_size=input_size,
                    n_series=n_series,
                    # max_steps=400,
                    # learning_rate=1e-3,
                    futr_exog_list=futr_exog,
                    hist_exog_list=hist_exog,
                    stat_exog_list=stat_exog,
                    scaler_type = "standard",
                )
            elif self.model_type == 'patchtst':
                return PatchTST(
                    h=h,
                    input_size=input_size,
                    # max_steps=400,
                    # learning_rate=1e-3,
                    futr_exog_list=futr_exog,
                    hist_exog_list=hist_exog,
                    stat_exog_list=stat_exog,
                    scaler_type = "standard",
                )
            elif self.model_type == 'tft':
                return TFT(
                        h=h,
                        input_size=input_size,
                        futr_exog_list=futr_exog,
                        hist_exog_list=hist_exog,
                        stat_exog_list=stat_exog,
                        scaler_type = "standard",
                    )
            elif self.model_type == 'tide':
                return TiDE(
                        h=h,
                        input_size=input_size,
                        futr_exog_list=futr_exog,
                        hist_exog_list=hist_exog,
                        stat_exog_list=stat_exog,
                        scaler_type = "standard",
                    )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported model type: {self.model_type}")

    # -----------------------------
    # High-level operations
    # -----------------------------
    def fit_and_save(
        self,
        *,
        df: pd.DataFrame,
        static_df: Optional[pd.DataFrame],
        freq: str,
        model_name: str,
        h: int,
        input_size: int,
        futr_exog: List[str],
        hist_exog: List[str],
        stat_exog: List[str],
        use_auto: bool,
        num_samples: int,
    ) -> Dict[str, str]:
        # Validierung der Pflichtfelder
        if df[["unique_id", "ds", "y"]].isnull().any().any():
            raise HTTPException(status_code=400, detail="Missing values in unique_id/ds/y")

        n_series = df["unique_id"].nunique()
        model = self._build_model(
            h=h,
            input_size=input_size,
            use_auto=use_auto,
            futr_exog=futr_exog,
            hist_exog=hist_exog,
            stat_exog=stat_exog,
            num_samples=num_samples,
            n_series=n_series,
        )
        nf = NeuralForecast(models=[model], freq=freq)
        nf.fit(df=df, static_df=static_df)

        save_dir = self._model_path(model_name)
        os.makedirs(save_dir, exist_ok=True)
        nf.save(path=save_dir, save_dataset=True, overwrite=True)

        with self._lock:
            self._loaded_models[model_name] = nf

        return {"status": "trained", "model_name": model_name, "path": save_dir}

    def cross_validate(
        self,
        *,
        df: pd.DataFrame,
        static_df: Optional[pd.DataFrame],
        freq: str,
        h: int,
        input_size: int,
        futr_exog: List[str],
        hist_exog: List[str],
        stat_exog: List[str],
        use_auto: bool,
        num_samples: int,
        val_size: int,
        test_size: int,
        step_size: int,
        n_windows: Optional[int],
    ) -> pd.DataFrame:
        n_series = df["unique_id"].nunique()
        model = self._build_model(
            h=h,
            input_size=input_size,
            use_auto=use_auto,
            futr_exog=futr_exog,
            hist_exog=hist_exog,
            stat_exog=stat_exog,
            num_samples=num_samples,
            n_series=n_series,
        )
        nf = NeuralForecast(models=[model], freq=freq)
        cv_df = nf.cross_validation(
            df=df,
            val_size=val_size,
            test_size=test_size,
            step_size=step_size,
            n_windows=n_windows,
        )
        return cv_df

    def predict(self, *, model_name: str, df: pd.DataFrame, futr_df: Optional[pd.DataFrame] = None, static_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        nf = self.load_or_get(model_name)
        return nf.predict(df, futr_df=futr_df, static_df=static_df)

    def list_models(self) -> List[str]:
        names: List[str] = []
        for name in os.listdir(self.models_dir):
            p = os.path.join(self.models_dir, name)
            if os.path.isdir(p):
                names.append(name)
        return names

    def fit_and_predict(
        self,
        *,
        df: pd.DataFrame,
        freq: str,
        h: int,
    ) -> pd.DataFrame:
        # Heuristic for input_size: 3 * horizon, but at least some minimum
        input_size = max(3 * h, 3)
        
        logger.info(f"Building model {self.model_type} with h={h}, input_size={input_size}")

        # Use the generic build model method
        n_series = df["unique_id"].nunique()
        model = self._build_model(
            h=h,
            input_size=input_size,
            use_auto=False,
            futr_exog=[],
            hist_exog=[],
            stat_exog=[],
            num_samples=5,
            n_series=n_series,
        )
        
        # Override max_steps for speed if it's not Auto and supported by the model
        if hasattr(model, 'max_steps'):
             model.max_steps = 100
        
        logger.info("Fitting model...")
        nf = NeuralForecast(models=[model], freq=freq)
        nf.fit(df=df)
        logger.info("Predicting...")
        return nf.predict()
