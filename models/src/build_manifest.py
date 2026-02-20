#!/usr/bin/env python3
"""
Build artifact manifest and save preprocessing artifacts for inference.

Scans training_output dirs, maps coins to artifact timestamps, and re-derives
preprocessing (scalers + feature_names) from parquet data using the same logic
as ProductionDataLoader.

Usage:
    python3 models/src/build_manifest.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import RobustScaler

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
ARTIFACTS_DIR = SCRIPT_DIR.parent / "artifacts"
TRAINING_OUTPUT_DIR = SCRIPT_DIR / "training_output"
PREPROCESSING_DIR = ARTIFACTS_DIR / "preprocessing"

# Training config defaults (must match what was used during training)
TRAIN_RATIO = 0.70
MAX_FEATURES = 50

# Model type → artifact directory name
KERAS_MODELS = ["dlinear", "tcn", "enhanced_lstm", "transformer"]


def scan_training_outputs():
    """Scan training_output dirs to get coin → timestamp mapping."""
    mapping = {}
    for d in sorted(TRAINING_OUTPUT_DIR.iterdir()):
        if not d.is_dir():
            continue
        parts = d.name.split("_", 1)
        if len(parts) == 2:
            coin, timestamp = parts[0], parts[1]
            mapping[coin] = timestamp
    return mapping


def verify_artifacts(coin, timestamp, all_timestamps):
    """Check which model types have artifacts for this coin/timestamp.

    Timestamps may differ by ±2 seconds between training output dir and
    model save time, so we match by date prefix and find the closest.
    """
    available = []
    # Find the actual .h5 file for each Keras model type
    # The timestamp in the filename may be off by 1-2 seconds
    date_prefix = timestamp[:8]  # e.g. "20260218"
    artifact_timestamps = {}

    for model_type in KERAS_MODELS:
        model_dir = ARTIFACTS_DIR / model_type
        if not model_dir.exists():
            continue
        # Find all .h5 files matching this date
        candidates = sorted(model_dir.glob(f"{model_type}_vproduction_{date_prefix}*.h5"))
        if not candidates:
            continue
        # Pick the one closest to our timestamp
        target_int = int(timestamp.replace("_", ""))
        best = min(candidates, key=lambda p: abs(
            int(p.stem.split("_vproduction_")[1].replace("_", "")) - target_int
        ))
        # Verify it's not closer to another coin's timestamp
        best_ts = best.stem.split("_vproduction_")[1]
        best_int = int(best_ts.replace("_", ""))
        is_closest_to_us = True
        for other_coin, other_ts in all_timestamps.items():
            if other_coin == coin:
                continue
            other_int = int(other_ts.replace("_", ""))
            if abs(best_int - other_int) < abs(best_int - target_int):
                is_closest_to_us = False
                break
        if is_closest_to_us:
            available.append(model_type)
            artifact_timestamps[model_type] = best_ts

    return available, artifact_timestamps


def derive_preprocessing(coin):
    """Re-derive preprocessing artifacts (scalers + feature_names) from parquet."""
    parquet_path = FEATURES_DIR / f"{coin}_features.parquet"
    if not parquet_path.exists():
        print(f"  WARNING: {parquet_path} not found, skipping")
        return None

    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {coin}: {df.shape[0]} rows, {df.shape[1]} cols")

    # Clean (same as ProductionDataLoader._clean)
    if "ticker" in df.columns:
        df = df.drop(columns=["ticker"])
    df = df.select_dtypes(include=[np.number])
    thresh = len(df) * 0.5
    df = df.dropna(axis=1, thresh=int(thresh))
    df = df.ffill()
    df = df.dropna()

    # Cross-asset features for BTC
    if coin == "BTC":
        eth_path = FEATURES_DIR / "ETH_features.parquet"
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
                cross["eth_volatility_20d"] = eth_close.pct_change().rolling(20).std()
                if "close" in df.columns:
                    ratio = eth_close.reindex(df.index) / df["close"]
                    cross["eth_btc_ratio_sma"] = ratio.rolling(20, min_periods=1).mean()
                cross = cross.reindex(df.index).ffill().fillna(0)
                df = pd.concat([df, cross], axis=1)
                print(f"  Added {cross.shape[1]} cross-asset ETH features for BTC")

    # Chronological split
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    train_df = df.iloc[:train_end]
    print(f"  Train split: {len(train_df)} rows (of {n} total)")

    # Feature selection on train only
    if "close" not in train_df.columns:
        print(f"  ERROR: 'close' column not found for {coin}")
        return None

    candidates = [c for c in train_df.columns if c != "close"]
    X_cand = train_df[candidates].values
    y_target = train_df["close"].values
    mask = np.all(np.isfinite(X_cand), axis=1) & np.isfinite(y_target)
    mi_scores = mutual_info_regression(X_cand[mask], y_target[mask], random_state=42)
    mi_series = pd.Series(mi_scores, index=candidates).sort_values(ascending=False)
    feature_names = mi_series.head(MAX_FEATURES).index.tolist()
    print(f"  Selected {len(feature_names)} features by mutual information")

    # Fit scalers on train only
    feature_scaler = RobustScaler()
    feature_scaler.fit(train_df[feature_names].values)

    # Target scaler: fit on train log-returns
    prices = train_df["close"].values
    lr = np.zeros_like(prices)
    lr[1:] = np.log(prices[1:] / prices[:-1])
    target_scaler = RobustScaler()
    target_scaler.fit(lr.reshape(-1, 1))

    return {
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_names": feature_names,
        "n_features": len(feature_names),
    }


def main():
    print("=" * 60)
    print("Building artifact manifest and preprocessing artifacts")
    print("=" * 60)

    # Step 1: Scan training outputs
    coin_timestamps = scan_training_outputs()
    print(f"\nFound {len(coin_timestamps)} coins in training_output:")
    for coin, ts in coin_timestamps.items():
        print(f"  {coin} → {ts}")

    # Determine which coin was trained last (its LightGBM is the one on disk)
    last_coin = max(coin_timestamps, key=lambda c: coin_timestamps[c])
    print(f"\nLast coin trained: {last_coin} (LightGBM artifacts belong to this coin only)")

    # Step 2: Build manifest and preprocessing for each coin
    manifest = {}
    for coin, timestamp in coin_timestamps.items():
        print(f"\n--- {coin} ---")
        available_models, artifact_timestamps = verify_artifacts(
            coin, timestamp, coin_timestamps
        )
        # LightGBM: only available for the last-trained coin
        lgb_dir = ARTIFACTS_DIR / "lightgbm"
        if coin == last_coin and (lgb_dir / "horizon_1.txt").exists():
            available_models.append("lightgbm")
        print(f"  Available models: {available_models}")

        preprocessing = derive_preprocessing(coin)
        if preprocessing is None:
            continue

        # Save preprocessing artifacts
        coin_dir = PREPROCESSING_DIR / coin
        coin_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessing["feature_scaler"], coin_dir / "feature_scaler.joblib")
        joblib.dump(preprocessing["target_scaler"], coin_dir / "target_scaler.joblib")
        with open(coin_dir / "feature_names.json", "w") as f:
            json.dump(preprocessing["feature_names"], f, indent=2)
        print(f"  Saved preprocessing to {coin_dir}")

        manifest[coin] = {
            "timestamp": timestamp,
            "artifact_timestamps": artifact_timestamps,
            "models": available_models,
            "n_features": preprocessing["n_features"],
        }

    # Step 3: Write manifest
    manifest_path = ARTIFACTS_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
