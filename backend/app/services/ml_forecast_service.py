"""
Production ML Forecast Service

Loads trained model artifacts (LSTM, Transformer, TCN, DLinear, LightGBM) and
generates multi-horizon cryptocurrency price forecasts via the FastAPI backend.

Model artifacts are discovered via models/artifacts/manifest.json (built by
models/src/build_manifest.py). Preprocessing artifacts (scalers, feature names)
are loaded from models/artifacts/preprocessing/{COIN}/.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent  # backend/
_PROJECT_ROOT = _BACKEND_DIR.parent                           # crypto-prediction/
_MODELS_SRC = _PROJECT_ROOT / "models" / "src"
_ARTIFACTS_DIR = _PROJECT_ROOT / "models" / "artifacts"
_FEATURES_DIR = _PROJECT_ROOT / "data" / "features"
_MANIFEST_PATH = _ARTIFACTS_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# Import ML model classes via importlib to avoid collision with backend/app/models/
# ---------------------------------------------------------------------------

ML_MODELS_AVAILABLE = False
_tf = None
_forecaster_classes: Dict[str, Any] = {}
_keras_custom_objects: Dict[str, Any] = {}  # Collected during import
_loaded_ml_modules: Dict[str, Any] = {}


def _import_ml_module(module_filename: str):
    """Import a module from models/src/models/{module_filename} by file path.

    Uses importlib.util to avoid the namespace collision between
    backend/app/models/ (SQLAlchemy) and models/src/models/ (ML).
    Returns the loaded module or None.
    """
    if module_filename in _loaded_ml_modules:
        return _loaded_ml_modules[module_filename]

    import importlib.util

    module_path = _MODELS_SRC / "models" / module_filename
    if not module_path.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        f"ml_models.{module_path.stem}", str(module_path)
    )
    if spec is None or spec.loader is None:
        return None

    mod = importlib.util.module_from_spec(spec)
    # Add models/src to path for intra-module imports
    if str(_MODELS_SRC) not in sys.path:
        sys.path.insert(0, str(_MODELS_SRC))
    try:
        spec.loader.exec_module(mod)  # type: ignore
    except Exception as e:
        logger.warning(f"Failed to load {module_filename}: {e}")
        return None

    _loaded_ml_modules[module_filename] = mod
    return mod


def _import_ml_class(module_filename: str, class_name: str):
    """Import a class from a module file."""
    mod = _import_ml_module(module_filename)
    if mod is None:
        return None
    return getattr(mod, class_name, None)


try:
    import tensorflow as tf
    # Prevent TF from grabbing all GPU/CPU memory at once
    tf.config.set_soft_device_placement(True)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    _tf = tf
except ImportError:
    logger.warning("TensorFlow not available — Keras models disabled")

if _tf is not None:
    # Import forecaster classes and collect custom Keras objects for model loading
    _keras_imports = [
        ("lstm", "enhanced_lstm.py", "EnhancedLSTMForecaster",
         ["AttentionLayer", "ResidualLSTMCell"]),
        ("transformer", "transformer_model.py", "TransformerForecaster",
         ["PositionalEncoding", "MultiHeadSelfAttention", "TransformerBlock",
          "MCDropout", "TransformerSchedule"]),
        ("tcn", "tcn_model.py", "TCNForecaster",
         ["TemporalBlock"]),
        ("dlinear", "dlinear_model.py", "DLinearForecaster",
         []),
    ]
    for _model_type, _filename, _classname, _custom_objs in _keras_imports:
        _cls = _import_ml_class(_filename, _classname)
        if _cls is not None:
            _forecaster_classes[_model_type] = _cls
        # Collect custom objects for keras.models.load_model
        _mod = _loaded_ml_modules.get(_filename)
        if _mod is not None:
            for _obj_name in _custom_objs:
                _obj = getattr(_mod, _obj_name, None)
                if _obj is not None:
                    _keras_custom_objects[_obj_name] = _obj

# LightGBM (no TF dependency)
_cls = _import_ml_class("lightgbm_model.py", "LightGBMForecaster")
if _cls is not None:
    _forecaster_classes["lightgbm"] = _cls

ML_MODELS_AVAILABLE = bool(_forecaster_classes)
logger.info(
    f"ML models available: {ML_MODELS_AVAILABLE} "
    f"({list(_forecaster_classes.keys())})"
)

# ---------------------------------------------------------------------------
# CoinGecko ID → Ticker mapping
# ---------------------------------------------------------------------------

COINGECKO_TO_TICKER = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "litecoin": "LTC",
    "ripple": "XRP",
    "dogecoin": "DOGE",
}

# Keras model type → artifact subdirectory
KERAS_MODEL_DIRS = {
    "lstm": "enhanced_lstm",
    "transformer": "transformer",
    "tcn": "tcn",
    "dlinear": "dlinear",
}

# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ProductionMLForecastService:
    """Load trained models and generate forecasts."""

    def __init__(self):
        self.manifest: Dict[str, Any] = {}
        self._model_cache: Dict[str, Any] = {}          # key: "{coin}_{model_type}"
        self._failed_models: set = set()                 # keys that crashed — don't retry
        self._preprocessing_cache: Dict[str, Any] = {}   # key: coin
        self._input_cache: Dict[str, Tuple[np.ndarray, datetime]] = {}  # key: coin
        self._input_cache_ttl = 300  # 5 minutes
        self._load_manifest()

    # -- manifest -----------------------------------------------------------

    def _load_manifest(self):
        if _MANIFEST_PATH.exists():
            with open(_MANIFEST_PATH) as f:
                self.manifest = json.load(f)
            logger.info(f"Loaded manifest: {list(self.manifest.keys())} coins")
        else:
            logger.warning(f"Manifest not found at {_MANIFEST_PATH}. Run build_manifest.py first.")

    # -- preprocessing ------------------------------------------------------

    def _load_preprocessing(self, coin: str) -> Dict[str, Any]:
        """Load preprocessing artifacts (scalers + feature_names) for a coin."""
        if coin in self._preprocessing_cache:
            return self._preprocessing_cache[coin]

        pp_dir = _ARTIFACTS_DIR / "preprocessing" / coin
        if not pp_dir.exists():
            raise FileNotFoundError(f"Preprocessing dir not found: {pp_dir}")

        feature_scaler = joblib.load(pp_dir / "feature_scaler.joblib")
        target_scaler = joblib.load(pp_dir / "target_scaler.joblib")
        with open(pp_dir / "feature_names.json") as f:
            feature_names = json.load(f)

        result = {
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "feature_names": feature_names,
        }
        self._preprocessing_cache[coin] = result
        return result

    # -- model loading ------------------------------------------------------

    def _load_model_instance(self, coin: str, model_type: str):
        """Load a model using the forecaster class's load_model method.

        For Keras models, we load with compile=False (inference only) to avoid
        needing custom LR schedule classes for deserialization.
        """
        forecaster_cls = _forecaster_classes.get(model_type)
        if forecaster_cls is None:
            raise ImportError(f"Forecaster class not available for {model_type}")

        forecaster = forecaster_cls()

        if model_type == "lightgbm":
            forecaster.load_model(str(_ARTIFACTS_DIR / "lightgbm"))
        else:
            coin_info = self.manifest.get(coin, {})
            artifact_dir_name = KERAS_MODEL_DIRS.get(model_type)
            artifact_ts = coin_info.get("artifact_timestamps", {}).get(
                artifact_dir_name
            )
            if not artifact_ts:
                raise FileNotFoundError(
                    f"No artifact timestamp for {coin}/{artifact_dir_name}"
                )
            h5_path = (
                _ARTIFACTS_DIR
                / artifact_dir_name
                / f"{artifact_dir_name}_vproduction_{artifact_ts}.h5"
            )
            if not h5_path.exists():
                raise FileNotFoundError(f"Model file not found: {h5_path}")

            # Load with compile=False to skip optimizer deserialization
            # (avoids TransformerSchedule / custom LR issues for inference)
            # safe_mode=False needed for TCN's Lambda layers
            forecaster.model = _tf.keras.models.load_model(
                str(h5_path),
                custom_objects=_keras_custom_objects,
                compile=False,
                safe_mode=False,
            )

        logger.info(f"Loaded {model_type} forecaster for {coin}")
        return forecaster

    def _get_model(self, coin: str, model_type: str):
        """Get a model from cache or load it."""
        cache_key = f"{coin}_{model_type}"

        if cache_key in self._failed_models:
            raise RuntimeError(f"Model {cache_key} previously failed to load")

        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        coin_info = self.manifest.get(coin, {})
        available = coin_info.get("models", [])

        # Map model_type to artifact directory name for availability check
        check_name = KERAS_MODEL_DIRS.get(model_type, model_type)
        if check_name not in available:
            raise ValueError(
                f"Model '{model_type}' not available for {coin}. "
                f"Available: {available}"
            )

        try:
            model = self._load_model_instance(coin, model_type)
        except Exception as e:
            self._failed_models.add(cache_key)
            raise

        self._model_cache[cache_key] = model
        return model

    # -- input preparation --------------------------------------------------

    def _prepare_input(self, coin: str, preprocessing: Dict[str, Any]) -> np.ndarray:
        """Prepare inference input from parquet features.

        Returns array of shape (1, 60, n_features+1) — one sliding window.
        The +1 is because models were trained on combined data where column 0
        is the scaled log-return target and columns 1..n are scaled features.
        """
        # Check cache
        if coin in self._input_cache:
            cached_input, cached_at = self._input_cache[coin]
            if (datetime.now() - cached_at).seconds < self._input_cache_ttl:
                return cached_input

        feature_names = preprocessing["feature_names"]
        feature_scaler = preprocessing["feature_scaler"]
        target_scaler = preprocessing["target_scaler"]

        parquet_path = _FEATURES_DIR / f"{coin}_features.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Feature parquet not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)

        # Clean (same as training pipeline)
        if "ticker" in df.columns:
            df = df.drop(columns=["ticker"])
        df = df.select_dtypes(include=[np.number])
        thresh = len(df) * 0.5
        df = df.dropna(axis=1, thresh=int(thresh))
        df = df.ffill()
        df = df.dropna()

        # Cross-asset features for BTC
        if coin == "BTC":
            eth_path = _FEATURES_DIR / "ETH_features.parquet"
            if eth_path.exists():
                eth_df = pd.read_parquet(eth_path)
                if "close" in eth_df.columns:
                    eth_close = eth_df["close"]
                    cross = pd.DataFrame(index=eth_df.index)
                    cross["eth_return_1d"] = eth_close.pct_change(1)
                    cross["eth_return_7d"] = eth_close.pct_change(7)
                    cross["eth_return_30d"] = eth_close.pct_change(30)
                    if "rsi_14" in eth_df.columns:
                        cross["eth_rsi_14"] = eth_df["rsi_14"]
                    cross["eth_volatility_20d"] = (
                        eth_close.pct_change().rolling(20).std()
                    )
                    if "close" in df.columns:
                        ratio = eth_close.reindex(df.index) / df["close"]
                        cross["eth_btc_ratio_sma"] = ratio.rolling(
                            20, min_periods=1
                        ).mean()
                    cross = cross.reindex(df.index).ffill().fillna(0)
                    df = pd.concat([df, cross], axis=1)

        # Check that all required features exist
        missing = [f for f in feature_names if f not in df.columns]
        if missing:
            raise ValueError(
                f"Missing {len(missing)} features for {coin}: {missing[:5]}..."
            )

        seq_len = 60
        tail = df.iloc[-seq_len:]
        if len(tail) < seq_len:
            raise ValueError(
                f"Not enough data for {coin}: need {seq_len} rows, got {len(tail)}"
            )

        # Scale features
        X_scaled = feature_scaler.transform(tail[feature_names].values)

        # Compute scaled log-return target (column 0 in combined array)
        close_prices = tail["close"].values
        log_returns = np.zeros_like(close_prices)
        log_returns[1:] = np.log(close_prices[1:] / close_prices[:-1])
        y_scaled = target_scaler.transform(log_returns.reshape(-1, 1))

        # Combined: [target, features] — matches training format
        combined = np.column_stack([y_scaled, X_scaled])
        X = combined.reshape(1, seq_len, combined.shape[1])

        self._input_cache[coin] = (X, datetime.now())
        return X

    # -- prediction ---------------------------------------------------------

    def _predict_keras(
        self, model, X: np.ndarray, return_confidence: bool = False
    ) -> Tuple[Dict[str, float], Optional[Dict[str, Tuple[float, float]]]]:
        """Run forecaster prediction, return per-horizon scalars.

        Uses the forecaster class's predict() which supports return_confidence
        for MC dropout uncertainty estimation.

        Returns (predictions_dict, confidence_dict_or_None)
        """
        raw = model.predict(X, return_confidence=False)
        preds = self._normalize_keras_output(raw)

        ci = None
        if return_confidence:
            try:
                result = model.predict(X, return_confidence=True)
                if isinstance(result, tuple) and len(result) == 2:
                    _, ci_bounds = result
                    if isinstance(ci_bounds, dict):
                        ci = {k: (float(v[0, 0]), float(v[0, 1])) for k, v in ci_bounds.items()}
                    elif isinstance(ci_bounds, list):
                        ci = {}
                        for i, bounds in enumerate(ci_bounds):
                            ci[f"horizon_{i+1}"] = (float(bounds[0, 0]), float(bounds[0, 1]))
            except Exception as e:
                logger.warning(f"Confidence estimation failed: {e}")

        return preds, ci

    def _normalize_keras_output(self, raw) -> Dict[str, float]:
        """Convert various Keras predict output formats to {key: scalar}."""
        if isinstance(raw, dict):
            return {k: float(v.ravel()[0]) if hasattr(v, 'ravel') else float(v) for k, v in raw.items()}
        if isinstance(raw, list):
            return {f"horizon_{i+1}": float(arr.ravel()[0]) for i, arr in enumerate(raw)}
        if isinstance(raw, np.ndarray):
            if raw.ndim == 1 or (raw.ndim == 2 and raw.shape[1] == 1):
                return {"horizon_1": float(raw.ravel()[0])}
            # Multi-column
            return {f"horizon_{i+1}": float(raw[0, i]) for i in range(raw.shape[1])}
        return {"horizon_1": float(raw)}

    def _predict_lightgbm(
        self, model, X: np.ndarray, return_confidence: bool = False
    ) -> Tuple[Dict[str, float], Optional[Dict[str, Tuple[float, float]]]]:
        """Run LightGBM prediction.

        LightGBM expects 2D input (n_samples, n_features * 5 summary stats).
        We compute summary stats from the 3D sequence.
        """
        # Build summary statistics from sequence (same as training)
        seq = X[0]  # (60, n_features)
        last = seq[-1, :]
        mean = seq.mean(axis=0)
        std = seq.std(axis=0)
        trend = seq[-1, :] - seq[0, :]
        momentum = seq[-1, :] - seq[-5, :] if seq.shape[0] >= 5 else np.zeros(seq.shape[1])
        flat = np.concatenate([last, mean, std, trend, momentum]).reshape(1, -1)

        raw = model.predict(flat)  # shape (1, n_horizons)
        preds = {
            f"horizon_{i+1}": float(raw[0, i]) for i in range(raw.shape[1])
        }

        ci = None
        if return_confidence:
            try:
                _, uncert = model.predict_proba(flat)
                ci = {}
                for i in range(uncert.shape[1]):
                    u = float(uncert[0, i])
                    p = float(raw[0, i])
                    ci[f"horizon_{i+1}"] = (p - 1.96 * u, p + 1.96 * u)
            except Exception as e:
                logger.warning(f"LightGBM confidence failed: {e}")

        return preds, ci

    # -- log-return → price conversion --------------------------------------

    def _to_prices(
        self,
        preds: Dict[str, float],
        ci: Optional[Dict[str, Tuple[float, float]]],
        current_price: float,
        target_scaler,
    ) -> Tuple[Dict[str, float], Optional[Dict[str, Tuple[float, float]]]]:
        """Convert scaled log-return predictions to absolute prices."""
        price_preds = {}
        price_ci = None

        for key, scaled_lr in preds.items():
            # Inverse-transform: scaled log-return → raw log-return
            raw_lr = float(
                target_scaler.inverse_transform(np.array([[scaled_lr]]))[0, 0]
            )
            price_preds[key] = current_price * np.exp(raw_lr)

        if ci is not None:
            price_ci = {}
            for key, (lo, hi) in ci.items():
                raw_lo = float(
                    target_scaler.inverse_transform(np.array([[lo]]))[0, 0]
                )
                raw_hi = float(
                    target_scaler.inverse_transform(np.array([[hi]]))[0, 0]
                )
                price_ci[key] = (
                    current_price * np.exp(raw_lo),
                    current_price * np.exp(raw_hi),
                )

        return price_preds, price_ci

    # -- public API ---------------------------------------------------------

    async def generate_ml_forecast(
        self,
        crypto_id: str,
        current_price: float,
        historical_prices: List[float],
        days: int = 7,
        model_type: str = "lstm",
    ) -> Dict[str, Any]:
        """Generate forecast using trained ML model.

        Returns dict matching the existing API contract.
        """
        coin = COINGECKO_TO_TICKER.get(crypto_id)
        if coin is None:
            raise ValueError(f"Unknown crypto_id: {crypto_id}")

        if not self.manifest:
            raise RuntimeError("No manifest loaded. Run build_manifest.py first.")

        if coin not in self.manifest:
            raise ValueError(f"No trained models for {coin}")

        # Handle ml_ensemble specially
        if model_type == "ml_ensemble":
            return await self._generate_ensemble_forecast(
                coin, current_price, days
            )

        try:
            preprocessing = self._load_preprocessing(coin)
            model = self._get_model(coin, model_type)
            X = self._prepare_input(coin, preprocessing)

            # Run prediction in thread pool to avoid blocking the async event loop
            loop = asyncio.get_event_loop()
            if model_type == "lightgbm":
                preds, ci = await loop.run_in_executor(
                    None, lambda: self._predict_lightgbm(model, X, return_confidence=True)
                )
            else:
                preds, ci = await loop.run_in_executor(
                    None, lambda: self._predict_keras(model, X, return_confidence=True)
                )

            # Convert scaled log-returns to prices
            price_preds, price_ci = self._to_prices(
                preds, ci, current_price, preprocessing["target_scaler"]
            )

            # Build forecast points
            forecasts = self._build_forecast_points(
                price_preds, price_ci, current_price, days
            )

            # Load training metrics from report if available
            metrics = self._load_training_metrics(coin, model_type)

            return {
                "model": model_type,
                "current_price": current_price,
                "generated_at": datetime.now().isoformat(),
                "forecast_horizon_days": days,
                "forecasts": forecasts,
                "historical_data": self._get_historical_data(coin),
                "model_metrics": {
                    "mape": float(metrics.get("mape", 0.0)),
                    "rmse": float(metrics.get("val_rmse_1d", metrics.get("rmse", 0.0))),
                    "r_squared": float(metrics.get("r_squared", 0.0)),
                },
                "status": "success",
                "note": f"ML {model_type} forecast",
                "using_ml": True,
            }

        except Exception as e:
            logger.error(f"ML forecast failed for {coin}/{model_type}: {e}")
            raise

    async def _generate_ensemble_forecast(
        self, coin: str, current_price: float, days: int
    ) -> Dict[str, Any]:
        """Weighted average of all available Keras models for a coin."""
        preprocessing = self._load_preprocessing(coin)
        X = self._prepare_input(coin, preprocessing)
        target_scaler = preprocessing["target_scaler"]

        available = self.manifest.get(coin, {}).get("models", [])
        keras_available = [
            mt for mt in ["dlinear", "tcn", "enhanced_lstm", "transformer"]
            if mt in available
        ]

        if not keras_available:
            raise ValueError(f"No Keras models available for {coin}")

        # Load training report to get validation RMSE for weighting
        report = self._load_training_report(coin)
        model_rmses = {}
        for mt in keras_available:
            # Map artifact dir name to report key
            report_key = "lstm" if mt == "enhanced_lstm" else mt
            model_data = report.get("models", {}).get(report_key, {})
            metrics = model_data.get("metrics", {})
            val_rmse = metrics.get("val_rmse_1d")
            if val_rmse is not None:
                model_rmses[mt] = float(val_rmse)

        # Inverse-squared-RMSE weights (same logic as ensemble training)
        if model_rmses:
            best_rmse = min(model_rmses.values())
            weights = {}
            for mt, rmse in model_rmses.items():
                if rmse > 2.0 * best_rmse:
                    weights[mt] = 0.0
                else:
                    weights[mt] = 1.0 / (rmse ** 2)
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
            else:
                weights = {mt: 1.0 / len(keras_available) for mt in keras_available}
        else:
            weights = {mt: 1.0 / len(keras_available) for mt in keras_available}

        # Predict with each model and combine
        all_preds: Dict[str, Dict[str, float]] = {}
        for mt in keras_available:
            mt_type = "lstm" if mt == "enhanced_lstm" else mt
            try:
                model = self._get_model(coin, mt_type)
                preds, _ = self._predict_keras(model, X)
                all_preds[mt] = preds
            except Exception as e:
                logger.warning(f"Ensemble: {mt} failed for {coin}: {e}")

        if not all_preds:
            raise RuntimeError(f"All models failed for {coin}")

        # Weighted average
        horizon_keys = list(next(iter(all_preds.values())).keys())
        combined = {}
        for hk in horizon_keys:
            total_weight = 0.0
            weighted_sum = 0.0
            for mt, preds in all_preds.items():
                if hk in preds:
                    w = weights.get(mt, 0.0)
                    weighted_sum += w * preds[hk]
                    total_weight += w
            combined[hk] = weighted_sum / total_weight if total_weight > 0 else 0.0

        price_preds, _ = self._to_prices(combined, None, current_price, target_scaler)
        forecasts = self._build_forecast_points(price_preds, None, current_price, days)

        return {
            "model": "ml_ensemble",
            "current_price": current_price,
            "generated_at": datetime.now().isoformat(),
            "forecast_horizon_days": days,
            "forecasts": forecasts,
            "historical_data": self._get_historical_data(coin),
            "model_metrics": {
                "mape": 0.0,
                "rmse": 0.0,
                "r_squared": 0.0,
                "weights": {k: round(v, 4) for k, v in weights.items()},
            },
            "status": "success",
            "note": "ML ensemble forecast",
            "using_ml": True,
        }

    # -- helpers ------------------------------------------------------------

    def _build_forecast_points(
        self,
        price_preds: Dict[str, float],
        price_ci: Optional[Dict[str, Tuple[float, float]]],
        current_price: float,
        days: int,
    ) -> List[Dict[str, Any]]:
        """Build forecast point dicts for the API response.

        Maps horizon predictions (1d, 7d, 30d) to the requested day range,
        interpolating both predicted price AND confidence intervals so each
        day gets unique, expanding bounds.
        """
        horizon_days = [1, 7, 30]
        horizon_prices: Dict[int, float] = {}
        horizon_keys: Dict[int, str] = {}
        for i, (key, price) in enumerate(sorted(price_preds.items())):
            if i < len(horizon_days):
                horizon_prices[horizon_days[i]] = price
                horizon_keys[horizon_days[i]] = key

        # Build absolute CI bounds at each horizon
        # If the model returns CI use it; otherwise derive a margin from the
        # predicted move so that the interval expands realistically.
        def _ci_at_horizon(h: int) -> Tuple[float, float]:
            p = horizon_prices.get(h, current_price)
            if price_ci is not None:
                k = horizon_keys.get(h)
                if k and k in price_ci:
                    return price_ci[k]
            # Fallback: margin grows with horizon distance
            margin_pct = {1: 0.015, 7: 0.04, 30: 0.10}.get(h, 0.05)
            return p * (1 - margin_pct), p * (1 + margin_pct)

        ci_bounds: Dict[int, Tuple[float, float]] = {
            h: _ci_at_horizon(h) for h in horizon_days if h in horizon_prices
        }

        def _interp(v0: float, v1: float, t: float) -> float:
            return v0 + t * (v1 - v0)

        forecasts = []
        for day_offset in range(1, days + 1):
            date = (datetime.now() + timedelta(days=day_offset)).isoformat()

            # --- Interpolate predicted price ---
            if day_offset <= 1 and 1 in horizon_prices:
                pred_price = horizon_prices[1]
            elif day_offset <= 7 and 1 in horizon_prices and 7 in horizon_prices:
                t = (day_offset - 1) / 6.0
                pred_price = _interp(horizon_prices[1], horizon_prices[7], t)
            elif day_offset <= 30 and 7 in horizon_prices and 30 in horizon_prices:
                t = (day_offset - 7) / 23.0
                pred_price = _interp(horizon_prices[7], horizon_prices[30], t)
            else:
                pred_price = horizon_prices.get(30, current_price)

            # --- Interpolate CI bounds (expanding with horizon) ---
            if day_offset <= 1 and 1 in ci_bounds:
                lo, hi = ci_bounds[1]
            elif day_offset <= 7 and 1 in ci_bounds and 7 in ci_bounds:
                t = (day_offset - 1) / 6.0
                lo = _interp(ci_bounds[1][0], ci_bounds[7][0], t)
                hi = _interp(ci_bounds[1][1], ci_bounds[7][1], t)
            elif day_offset <= 30 and 7 in ci_bounds and 30 in ci_bounds:
                t = (day_offset - 7) / 23.0
                lo = _interp(ci_bounds[7][0], ci_bounds[30][0], t)
                hi = _interp(ci_bounds[7][1], ci_bounds[30][1], t)
            else:
                margin_pct = 0.02 + (day_offset / max(days, 1)) * 0.08
                lo = pred_price * (1 - margin_pct)
                hi = pred_price * (1 + margin_pct)

            base_confidence = max(0.5, 0.92 - day_offset * 0.015)

            forecasts.append({
                "date": date,
                "predicted_price": round(float(pred_price), 2),
                "confidence": round(float(base_confidence), 4),
                "confidence_lower": round(float(lo), 2),
                "confidence_upper": round(float(hi), 2),
            })

        return forecasts

    def _get_historical_data(self, coin: str, n_days: int = 30) -> List[Dict[str, Any]]:
        """Return last N days of close prices from parquet as historical_data list."""
        parquet_path = _FEATURES_DIR / f"{coin}_features.parquet"
        if not parquet_path.exists():
            return []
        try:
            df = pd.read_parquet(parquet_path)
            if "close" not in df.columns:
                return []
            tail = df["close"].dropna().tail(n_days)
            return [
                {"date": str(idx.date() if hasattr(idx, "date") else idx),
                 "price": round(float(price), 2),
                 "is_historical": True}
                for idx, price in tail.items()
            ]
        except Exception as e:
            logger.warning(f"Failed to load historical data for {coin}: {e}")
            return []

    def _load_training_report(self, coin: str) -> Dict[str, Any]:
        """Load training report JSON for a coin."""
        ts = self.manifest.get(coin, {}).get("timestamp", "")
        report_dir = _MODELS_SRC / "training_output" / f"{coin}_{ts}"
        report_path = report_dir / "training_report.json"
        if report_path.exists():
            with open(report_path) as f:
                return json.load(f)
        return {}

    def get_all_model_metrics(self) -> Dict[str, Any]:
        """Aggregate training metrics across all coins for all model types.

        Returns a dict keyed by model_type with per-coin metrics, average
        directional accuracy, and test-set RMSE/MAE from ensemble evaluation.
        """
        # report key → API model_type
        MODEL_INFO = {
            "lstm": {
                "report_key": "lstm",
                "label": "LSTM",
                "description": "Bidirectional LSTM with attention mechanism",
                "type": "deep_learning",
                "ensemble_key": "enhanced_lstm",
            },
            "transformer": {
                "report_key": "transformer",
                "label": "Transformer",
                "description": "Multi-head self-attention with causal masking",
                "type": "deep_learning",
                "ensemble_key": "transformer",
            },
            "tcn": {
                "report_key": "tcn",
                "label": "TCN",
                "description": "Temporal Convolutional Network with dilated causal convolutions",
                "type": "deep_learning",
                "ensemble_key": "tcn",
            },
            "dlinear": {
                "report_key": "dlinear",
                "label": "DLinear",
                "description": "Trend/seasonal decomposition baseline model",
                "type": "deep_learning",
                "ensemble_key": "dlinear",
            },
            "lightgbm": {
                "report_key": "lightgbm",
                "label": "LightGBM",
                "description": "Gradient boosting with 150+ engineered features",
                "type": "machine_learning",
                "ensemble_key": "lightgbm",
            },
        }

        result: Dict[str, Any] = {}

        for model_type, info in MODEL_INFO.items():
            per_coin: Dict[str, Any] = {}
            da_1d_values: List[float] = []
            available_coins: List[str] = []

            for coin in self.manifest:
                report = self._load_training_report(coin)
                if not report:
                    continue

                model_data = report.get("models", {}).get(info["report_key"], {})
                if not model_data:
                    continue

                available_coins.append(coin)
                metrics = model_data.get("metrics", {})
                da = model_data.get("directional_accuracy", {})

                coin_metrics: Dict[str, Any] = {}

                # Validation metrics
                for k in ("val_rmse_1d", "val_rmse_7d", "val_rmse_30d"):
                    if k in metrics:
                        coin_metrics[k] = round(float(metrics[k]), 4)

                # LightGBM uses val_mae array
                if "val_mae" in metrics and isinstance(metrics["val_mae"], list):
                    coin_metrics["val_mae"] = [round(float(v), 4) for v in metrics["val_mae"]]

                # Directional accuracy
                if da:
                    coin_metrics["directional_accuracy"] = {
                        k: round(float(v), 4) for k, v in da.items()
                    }
                    if "1d" in da:
                        da_1d_values.append(float(da["1d"]))

                # Test-set metrics from ensemble_evaluation
                ens_metrics = report.get("ensemble_evaluation", {}).get("metrics", {})
                ens_key = info["ensemble_key"]
                if f"{ens_key}_rmse" in ens_metrics:
                    coin_metrics["test_rmse"] = round(float(ens_metrics[f"{ens_key}_rmse"]), 4)
                if f"{ens_key}_mae" in ens_metrics:
                    coin_metrics["test_mae"] = round(float(ens_metrics[f"{ens_key}_mae"]), 4)

                per_coin[coin] = coin_metrics

            avg_da = round(sum(da_1d_values) / len(da_1d_values), 4) if da_1d_values else None

            result[model_type] = {
                "label": info["label"],
                "description": info["description"],
                "type": info["type"],
                "available_coins": available_coins,
                "avg_directional_accuracy_1d": avg_da,
                "per_coin": per_coin,
            }

        # ml_ensemble: aggregate ensemble_evaluation across coins
        ens_per_coin: Dict[str, Any] = {}
        ens_coins: List[str] = []
        for coin in self.manifest:
            report = self._load_training_report(coin)
            ens_eval = report.get("ensemble_evaluation", {}).get("metrics", {})
            if ens_eval:
                ens_coins.append(coin)
                ens_per_coin[coin] = {
                    "ensemble_rmse": round(float(ens_eval.get("ensemble_rmse", 0)), 4),
                    "ensemble_mae": round(float(ens_eval.get("ensemble_mae", 0)), 4),
                    "improvement_pct": round(float(ens_eval.get("improvement_pct", 0)), 2),
                }

        result["ml_ensemble"] = {
            "label": "ML Ensemble",
            "description": "Weighted average of all deep learning models",
            "type": "ensemble",
            "available_coins": ens_coins,
            "avg_directional_accuracy_1d": None,
            "per_coin": ens_per_coin,
        }

        return {"models": result, "ml_available": ML_MODELS_AVAILABLE}

    def get_fallback_price(self, crypto_id: str) -> Optional[float]:
        """Get last close price from parquet as fallback when APIs are down."""
        coin = COINGECKO_TO_TICKER.get(crypto_id)
        if not coin:
            return None
        parquet_path = _FEATURES_DIR / f"{coin}_features.parquet"
        if not parquet_path.exists():
            return None
        try:
            df = pd.read_parquet(parquet_path)
            if "close" in df.columns:
                return float(df["close"].iloc[-1])
        except Exception as e:
            logger.warning(f"Failed to read fallback price for {coin}: {e}")
        return None

    def _load_training_metrics(self, coin: str, model_type: str) -> Dict[str, Any]:
        """Extract validation metrics for a specific model from training report."""
        report = self._load_training_report(coin)
        report_key = model_type
        if model_type in KERAS_MODEL_DIRS:
            report_key = KERAS_MODEL_DIRS[model_type]
        # Training report uses "lstm" not "enhanced_lstm"
        if report_key == "enhanced_lstm":
            report_key = "lstm"
        model_data = report.get("models", {}).get(report_key, {})
        return model_data.get("metrics", {})


# ---------------------------------------------------------------------------
# Global service instance
# ---------------------------------------------------------------------------

ml_forecast_service = ProductionMLForecastService()
