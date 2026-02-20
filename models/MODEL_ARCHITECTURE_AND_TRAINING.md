# Model Architecture & Training Methodology

## Why Five Models?

Cryptocurrency markets exhibit fundamentally different statistical behaviors depending on time scale, regime, and market conditions. No single model architecture dominates across all scenarios. Our approach trains five complementary architectures and combines them through a CV-stacked ensemble:

| Property | DLinear | TCN | LSTM | Transformer | LightGBM |
|----------|---------|-----|------|-------------|----------|
| **Strength** | Overfitting-proof baseline | Efficient long-range patterns | Sequential memory, local patterns | Global attention patterns | Tabular feature interactions |
| **Weakness** | Limited capacity | Fixed receptive field | Struggles with long sequences | Data-hungry | No native temporal modeling |
| **Best regime** | All (regularizer) | Trending / cyclical | Trending markets | Volatile / regime-shifting | Mean-reverting / feature-rich |
| **Parameters** | ~3,000 | ~15,000 | ~50,000 | ~40,000 | N/A (trees) |
| **Inference speed** | Fast (~5ms) | Fast (~8ms) | Medium (~32ms) | Slow (~45ms) | Fast (~8ms) |

By combining all five, the ensemble captures simple trend/seasonal structure (DLinear), efficient temporal patterns (TCN), short-term momentum (LSTM), long-range structural shifts (Transformer), and complex feature interactions (LightGBM). The ensemble's CV-stacked meta-learner learns the optimal combination from out-of-fold predictions.

---

## Model 1: DLinear (Simple Linear Baseline)

**File:** `src/models/dlinear_model.py`
**Class:** `DLinearForecaster`

### Why DLinear?

DLinear (from "Are Transformers Effective for Time Series Forecasting?", Zeng et al. 2023) demonstrates that a simple linear decomposition model can match or beat Transformers on many time series benchmarks. With only ~3,000 parameters, it is virtually impossible to overfit — making it an ideal regularizing baseline in the ensemble and a sanity check for more complex models.

### Architecture

```
Input (60 timesteps x N features)
  |
  AveragePooling1D(kernel=25, padding=same) → trend component
  Subtract(input, trend) → seasonal component
  |
  Flatten(trend) → Dropout(0.1) → Dense(1) → trend_pred
  Flatten(seasonal) → Dropout(0.1) → Dense(1) → seasonal_pred
  |
  Add(trend_pred, seasonal_pred) → output
  |
  Separate heads for 1d / 7d / 30d horizons
```

### Key Design Choices

- **Trend/seasonal decomposition**: A moving average (AvgPool1D with kernel=25) extracts the trend. The remainder is the seasonal component. Each gets its own linear projection, then they're summed.
- **Extreme simplicity**: Two linear layers per horizon head, ~3,000 total parameters. This makes DLinear a strong lower bound — if complex models can't beat it, they're overfitting.
- **No recurrence or attention**: Fully parallelizable, trains in seconds.
- **Huber loss per horizon**: Same loss weighting scheme as other models (`1/sqrt(h)`).

### Hyperparameters

| Parameter | Default | Optuna Range | Rationale |
|-----------|---------|--------------|-----------|
| Kernel size | 25 | 15-35 | Moving average window for trend extraction |
| Dropout | 0.1 | 0.05-0.3 | Light regularization (few params to regularize) |
| Learning rate | 0.001 | 1e-4 to 1e-2 | Standard Adam learning rate |
| Early stopping patience | 10 | — | Stops early if validation loss plateaus |

---

## Model 2: TCN (Temporal Convolutional Network)

**File:** `src/models/tcn_model.py`
**Class:** `TCNForecaster`

### Why TCN?

TCNs use dilated causal convolutions with residual connections to capture long-range dependencies. Compared to LSTMs, TCNs are fully parallelizable (no sequential hidden state), have a well-defined receptive field, and achieve comparable accuracy with fewer parameters. With 4 blocks and kernel size 3, the receptive field is 49 steps — covering most of the 60-step input window.

### Architecture

```
Input (60 timesteps x N features)
  |
  TemporalBlock(filters=32, kernel=3, dilation=1) + residual
  TemporalBlock(filters=32, kernel=3, dilation=2) + residual
  TemporalBlock(filters=32, kernel=3, dilation=4) + residual
  TemporalBlock(filters=32, kernel=3, dilation=8) + residual
  |
  Lambda(x[:, -1, :])  →  last timestep (causal — only uses past)
  |
  Dropout(0.3) → Dense → Multi-step heads for 1d / 7d / 30d
```

Each TemporalBlock:
```
Conv1D(causal, dilation) → BatchNorm → ReLU → Dropout
Conv1D(causal, dilation) → BatchNorm → ReLU → Dropout
+ Residual connection (1x1 conv if dimensions differ)
```

Receptive field: `2^4 × (3-1) × 2 + 1 = 49 steps`

### Key Design Choices

- **Dilated causal convolutions**: Exponentially increasing dilation rates (1, 2, 4, 8) give each successive block a wider receptive field without increasing parameter count.
- **Causal padding**: Each Conv1D uses `padding='causal'` so the model never sees future data.
- **Residual connections**: Skip connections with 1x1 convolution for dimension matching prevent gradient vanishing.
- **BatchNormalization**: Stabilizes training and acts as a regularizer.
- **~15k parameters**: Well-suited for ~2,500 training samples — significantly smaller than the LSTM.

### Hyperparameters

| Parameter | Default | Optuna Range | Rationale |
|-----------|---------|--------------|-----------|
| Filters | 32 | 16-64 | Channels per temporal block |
| Kernel size | 3 | 2-5 | Conv kernel width |
| Blocks | 4 | 3-6 | Depth of TCN stack (controls receptive field) |
| Dropout | 0.3 | 0.1-0.4 | Applied after each conv + final layer |
| Learning rate | 0.001 | — | With ReduceLROnPlateau (factor=0.5, patience=5) |

---

## Model 3: Enhanced LSTM

**File:** `src/models/enhanced_lstm.py`
**Class:** `EnhancedLSTMForecaster`

### Why LSTM?

LSTMs are the workhorse of sequential prediction. They maintain an internal cell state that acts as memory, allowing them to learn how recent price action, volume spikes, and technical indicator crossovers predict near-future movements. For cryptocurrency markets, where short-term momentum and mean-reversion patterns dominate intraday and daily timeframes, LSTMs are a natural fit.

### Architecture

```
Input (60 timesteps x N features)
  |
  LayerNormalization
  |
  Bidirectional LSTM (64 units, L2=1e-4) + Dropout (0.4)
  |
  Residual LSTM (32 units, L2=1e-4) + LayerNorm
  |
  Attention Layer (learnable weights over all 60 timesteps)
  |
  Dense (32, L2=1e-4) -> LayerNorm -> Dropout
  |              |              |
  Head 1d        Head 7d        Head 30d   (multi-step output)
```

### Key Design Choices

- **Bidirectional first layer**: Captures both forward and backward temporal context. Only the first layer is bidirectional to control parameter count.
- **Residual connections**: Each `ResidualLSTMCell` wraps a standard LSTM with a linear projection skip-connection and layer normalization, preventing gradient vanishing.
- **L2 kernel regularization**: `kernel_regularizer=l2(1e-4)` on all LSTM and Dense layers reduces overfitting for BTC's limited training data.
- **Attention mechanism**: After the LSTM stack, a Bahdanau-style attention layer computes learned weights over all 60 timesteps, letting the model focus on the most predictive time points.
- **Multi-step heads**: Separate output heads for 1-day, 7-day, and 30-day horizons with `1/sqrt(h)` loss weighting.
- **No recurrent dropout**: Set to 0.0 to enable fast CuDNN (NVIDIA) and Metal (Apple Silicon) GPU kernels. Any value > 0 forces a pure Python LSTM fallback that is 10-100x slower.
- **Monte Carlo dropout**: At inference time, dropout remains active across multiple forward passes for calibrated uncertainty estimates.
- **Data augmentation**: Training sequences are augmented with Gaussian jittering (N(0, 0.01)) and random scaling (U(0.97, 1.03)), tripling effective dataset size.

### Hyperparameters

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| Sequence length | 60 | ~2 months of daily data |
| LSTM units | [64, 32] | 2-layer stack — reduced from [128,64,32] to prevent overfitting |
| Dense units | [32] | Single head — smaller to reduce capacity |
| Dropout | 0.4 | Standard inter-layer dropout |
| L2 regularization | 1e-4 | Kernel regularization on LSTM + Dense layers |
| Recurrent dropout | 0.0 | Must be 0 for CuDNN/Metal GPU kernels |
| Learning rate | 0.0005 | Adam optimizer |
| Gradient clipping | 1.0 | Prevents exploding gradients |
| Early stopping patience | 10 | Reduced from 20 to stop earlier |
| Loss function | Huber | Robust to crypto flash crashes |

---

## Model 4: Transformer

**File:** `src/models/transformer_model.py`
**Class:** `TransformerForecaster`

### Why Transformers?

Transformers process the entire input sequence in parallel through self-attention, computing pairwise relationships between all timesteps simultaneously. This gives them two advantages over LSTMs:

1. **Long-range dependencies**: Self-attention directly connects patterns from 60 days ago to today's prediction without sequential gate propagation.
2. **Multi-scale pattern detection**: Different attention heads naturally specialize on different temporal scales.

### Architecture

```
Input (60 timesteps x N features)
  |
  Dense Projection -> d_model (64)
  |
  Positional Encoding (learnable)
  |  Dropout (0.35)
  |
  Transformer Block x2:
  |   Multi-Head Self-Attention (4 heads, causal mask)
  |   Residual Add & LayerNorm
  |   Feed-Forward Network (256 -> GELU -> 64)
  |   Residual Add & LayerNorm
  |
  Global Average Pooling
  |
  Dense (64) -> Dropout -> Output 1d
  Dense (64) -> Dropout -> Output 7d
  Dense (64) -> Dropout -> Output 30d
```

### Key Design Choices

- **Reduced capacity**: d_model=64 (from 128), ff_dim=256 (from 512), 2 layers (from 3) — better suited for ~2,500 training samples.
- **Causal masking**: Lower-triangular mask prevents information leakage from future timesteps.
- **Learnable positional encoding**: Discovers temporal significance patterns.
- **AdamW with weight decay**: `weight_decay=0.01` provides additional L2-style regularization beyond dropout.
- **Higher dropout (0.35)**: Increased from 0.25 to combat overfitting on BTC.
- **Warmup learning rate schedule**: Ramps up during warmup steps, then decays — critical for attention weight initialization.
- **GELU activation**: Smoother gradients than ReLU, better for attention mechanisms.
- **Huber loss per horizon**: Robust to extreme price moves.
- **Data augmentation**: Same jittering + scaling as LSTM, tripling effective training data.

### Hyperparameters

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| d_model | 64 | Reduced from 128 for regularization |
| Attention heads | 4 | Each head = 16 dims |
| Feed-forward dim | 256 | Reduced from 512 |
| Transformer layers | 2 | Reduced from 3 |
| Dropout | 0.35 | Increased from 0.25 |
| Weight decay | 0.01 | AdamW L2 regularization |
| Warmup steps | 500 | Standard transformer schedule |
| Batch size | 32 | Smaller batches for gradient noise regularization |
| Early stopping patience | 10 | Reduced from 15 |
| Causal mask | Yes | Prevents future information leakage |

---

## Model 5: LightGBM

**File:** `src/models/lightgbm_model.py`
**Class:** `LightGBMForecaster`

### Why Gradient Boosting?

Gradient boosted trees excel at learning non-linear feature interactions from tabular data. While deep learning models operate on raw sequences, LightGBM consumes **summary statistics** extracted from the input window:

1. **Feature interactions**: Discovers complex rules like "RSI < 30 AND volume > 2x average AND volatility declining" without needing temporal convolutions.
2. **Speed and interpretability**: Training takes seconds. Feature importance rankings provide direct insight.
3. **Robustness**: Naturally resistant to feature scaling issues, outliers, and irrelevant features.

### Feature Preparation

Instead of flattening the full 60×N window (which creates a p >> n problem), LightGBM extracts 5 summary statistics per feature:

```
Input (60 timesteps x N features)
  |
  last_step  = X[:, -1, :]                    # Most recent values
  means      = X.mean(axis=1)                  # Window averages
  stds       = X.std(axis=1)                   # Window volatility
  trends     = X[:, -1, :] - X[:, 0, :]       # Window trend
  momentum   = X[:, -5:].mean(1) - X[:, -10:-5].mean(1)  # Recent momentum
  |
  X_flat = hstack([last_step, means, stds, trends, momentum])
  # N features × 5 summaries = 5N features (vs 60N with flat reshape)
```

This reduces features from ~3,000 to ~250 (with 50 selected features), keeping the feature-to-sample ratio manageable.

Each horizon gets its own independent GBDT model.

### Hyperparameters

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| n_estimators | 500 | Sufficient trees with early stopping |
| num_leaves | 31 | Reduced from 127 to prevent overfitting |
| max_depth | 5 | Reduced from 8 for shallower trees |
| learning_rate | 0.05 | Standard boosting rate |
| feature_fraction | 0.8 | Column subsampling per tree |
| bagging_fraction | 0.8 | Row subsampling per tree |
| min_child_samples | 50 | Increased from 20 — larger minimum leaf size |
| reg_alpha (L1) | 1.0 | Increased from 0.1 for stronger regularization |
| reg_lambda (L2) | 1.0 | Increased from 0.1 for stronger regularization |
| early_stopping | 50 rounds | Stops when validation loss plateaus |

---

## Model 6: Advanced Ensemble

**File:** `src/models/advanced_ensemble.py`
**Class:** `AdvancedEnsemble`

### Why an Ensemble?

The five base models make different types of errors. DLinear captures the obvious trend, TCN finds periodic patterns, LSTM tracks momentum, the Transformer spots regime changes, and LightGBM exploits feature interactions. An ensemble exploits this error diversity.

Our ensemble uses three complementary techniques:

### 1. CV-Stacked Meta-Learner

Base models are trained on the training split only. The meta-learner trains on their **validation set predictions** (genuinely out-of-sample). A `TimeSeriesSplit(n_splits=3)` cross-validation on these validation predictions provides an unbiased estimate of ensemble quality.

```
Base models trained on X_train
  |
  Get predictions on X_val (genuinely OOF)
  |
  Compute per-model RMSE on validation predictions
  |
  TimeSeriesSplit(n_splits=3) on validation predictions
  |
  If OOF ensemble RMSE < best individual:
    Train final LightGBM meta-learner on ALL validation predictions
    (n_estimators=100, max_depth=3, num_leaves=8, reg_alpha=1.0)
  Else:
    Fall back to performance-based weighted average
  |
  Always: set inverse-squared-RMSE weights from validation RMSE
  Exclude models with RMSE > 2x best individual (weight = 0)
```

**Fallback logic**: If the stacked ensemble can't beat the best individual model on cross-validated validation splits, it falls back to a performance-based weighted average using inverse-squared-RMSE weights. This ensures poorly-performing models (e.g., DLinear on BTC) receive minimal weight without manual tuning.

### 2. Market Regime Detection

The `MarketRegimeDetector` classifies the current market into one of four regimes using price statistics:

| Regime | Detection Logic | Weight Adjustment |
|--------|----------------|--------------------------|
| **Trending** | Strong directional momentum, low noise | LSTM weight increases (momentum-sensitive) |
| **Mean-reverting** | Oscillating around a level, high autocorrelation | LightGBM weight increases (feature-driven) |
| **Volatile** | High variance, large swings | Transformer weight increases (long-range context) |
| **Ranging** | Low volatility, tight bands | Equal weighting |

### 3. Online Weight Adaptation

After each prediction, the ensemble updates model weights based on recent accuracy using exponential moving averages. Models that performed well in the last 50 predictions receive higher weight, enabling real-time adaptation without retraining.

### Default Weights

| Model | Default Weight | Regime: Trending | Regime: Volatile | Regime: Ranging |
|-------|---------------|-----------------|-----------------|----------------|
| DLinear | 0.10 | 0.10 | 0.10 | 0.15 |
| TCN | 0.15 | 0.20 | 0.15 | 0.15 |
| Transformer | 0.25 | 0.25 | 0.30 | 0.20 |
| Enhanced LSTM | 0.25 | 0.30 | 0.20 | 0.20 |
| LightGBM | 0.25 | 0.15 | 0.25 | 0.30 |

---

## Training Pipeline

### Data Flow

```
Raw Parquet (5600+ rows, 150+ features per crypto)
     |
     v
[1] Clean: drop non-numeric, handle NaN, remove >50% missing cols
     |
     v
[2] Chronological Split: 70% train / 15% validation / 15% test
     |
     v
[3] Feature Selection: top 80 features by mutual information (train data only)
     |
     v
[4] Cross-Asset Features: ETH indicators added when training BTC
     |
     v
[5] Target Transform: log-returns (stationary across bull/bear regimes)
     |
     v
[6] Scale: RobustScaler (fit on train split only)
     |
     v
[7] Sequence: sliding window of 60 days, target = log-return at +1/+7/+30 days
     |
     v
[8] Augment: jittering + scaling (deep learning models only, 3x data)
     |
     v
[9] Train: DLinear -> TCN -> LSTM -> Transformer -> LightGBM -> Ensemble
     |
     v
[10] Save: model weights to models/artifacts/ (saved FIRST — crash-safe)
     |
     v
[11] Visualize: 10 PNG diagnostic plots saved to training_output/
     |
     v
[12] Report: training report JSON with metrics, uncertainty, ensemble eval
     |
     v
[13] Log: all metrics, configs, and artifacts to MLflow
```

### Key Pipeline Improvements (Feb 2026)

| Change | Before | After | Impact |
|--------|--------|-------|--------|
| Feature selection | Pearson correlation on full dataset | Mutual information on train split only | Eliminates data leakage |
| Target variable | Scaled absolute close price | Log-returns: `log(close_{t+h} / close_t)` | Stationary, bounded variance |
| Feature count | 60 (correlation) | 80 (mutual information) | Captures nonlinear relationships |
| LightGBM features | Flat reshape (60×50 = 3,000) | Summary stats (5×N ≈ 250) | 12x feature reduction |
| LSTM capacity | [128, 64, 32] + [64, 32] dense | [64, 32] + [32] dense + L2 reg | ~60% fewer parameters |
| Transformer capacity | d_model=128, 3 layers, ff=512 | d_model=64, 2 layers, ff=256 | ~75% fewer parameters |
| Data augmentation | None | Jittering + scaling (3x data) | Triples effective training set |
| Ensemble meta-learner | ElasticNet on in-sample predictions | LightGBM on CV out-of-fold predictions | Prevents stacking on overfit predictions |
| Cross-asset features | None | ETH indicators for BTC training | Macro crypto market context |
| Model count | 3 | 5 (+ DLinear, TCN) | More diversity for ensemble |

### Why Chronological Splits (Not Random)?

Cryptocurrency prices are non-stationary time series. A random train/test split would leak future information into training. We use strict chronological ordering:

- **Train**: oldest 70% of data (learns historical patterns)
- **Validation**: next 15% (tunes hyperparameters, triggers early stopping)
- **Test**: newest 15% (final unbiased evaluation — never seen during training)

### Why Log-Return Targets?

BTC ranges from $3k to $69k — enormous non-stationarity. Predicting absolute price means a model trained on 2020 data ($10k) sees completely different magnitudes in 2024 ($60k). Log-returns (`log(close_{t+h} / close_t)`) are:

- **Stationary** across bull/bear regimes
- **Bounded variance** (typically -0.3 to +0.3 for daily)
- **Comparable across coins** (BTC and ETH have similar return distributions)

For inverse transform: `predicted_price = current_price × exp(predicted_log_return)`.

### Why RobustScaler?

Crypto prices have extreme outliers (flash crashes, parabolic rallies). `RobustScaler` uses the interquartile range instead of standard deviation, making it resistant to these events. The scaler is fit only on training data to prevent data leakage.

### Multi-Step Forecasting

All models predict three horizons simultaneously through separate output heads:

| Horizon | Use Case | Difficulty |
|---------|----------|------------|
| **1 day** | Short-term trading signals | Easiest (most autocorrelation) |
| **7 days** | Swing trading, position sizing | Medium |
| **30 days** | Portfolio allocation, risk management | Hardest (most uncertainty) |

Shorter horizons receive higher loss weights during training (`1/sqrt(h)`) because they are more predictable and more commercially valuable.

### Cross-Asset Features

When training BTC, the pipeline loads ETH feature data and adds 6 cross-asset columns:

| Feature | Description |
|---------|-------------|
| `eth_return_1d` | ETH 1-day return |
| `eth_return_7d` | ETH 7-day return |
| `eth_return_30d` | ETH 30-day return |
| `eth_rsi_14` | ETH 14-day RSI |
| `eth_volatility_20d` | ETH 20-day rolling volatility |
| `eth_btc_ratio_sma` | 20-day SMA of ETH/BTC price ratio |

These provide macro context — ETH often leads or confirms crypto market moves.

### Data Augmentation

Applied to deep learning models (DLinear, TCN, LSTM, Transformer) only:

- **Jittering**: Gaussian noise N(0, 0.01) added to all features
- **Scaling**: Per-sample random factor U(0.97, 1.03) applied to all features
- **Effect**: Triples effective dataset size (~2,500 → ~7,500 sequences)
- **Train-only**: Never applied to validation or test data

---

## Training Visualizations

The training script generates 10 diagnostic plots per cryptocurrency:

| Plot | What It Shows |
|------|---------------|
| `loss_curves.png` | Train vs validation loss per epoch for each model |
| `metrics_progression.png` | Per-horizon loss progression over training epochs |
| `learning_rates.png` | LR schedules (decay, warmup, plateau reduction) |
| `attention_heatmap.png` | LSTM attention weights showing which timesteps the model focuses on |
| `feature_importance.png` | Top 30 LightGBM features ranked by importance |
| `model_comparison.png` | Side-by-side RMSE/MAE bars for all 5 models at each horizon |
| `predictions_vs_actual.png` | Scatter plots with R-squared per model per horizon |
| `residual_analysis.png` | Residual distributions and residual-vs-predicted scatter |
| `ensemble_weights.png` | Contribution of each base model in the ensemble |
| `training_summary.png` | Overview panel: training times, convergence, epochs, final RMSE |

---

## Running Training

**GPU is strongly recommended for production training.** Google Colab (T4/A100) or Apple Silicon Macs with Metal GPU acceleration both work.

```bash
# Full production training for BTC and ETH
python3 models/src/train_production.py --crypto BTC,ETH

# Quick validation run (3 epochs, skip ensemble)
python3 models/src/train_production.py --crypto BTC --epochs 3 --no-ensemble

# With Optuna hyperparameter tuning (all 5 models)
python3 models/src/train_production.py --crypto BTC --tune --tune-trials 20 \
    --tune-models dlinear,tcn,lstm,transformer,lightgbm

# Tune deep learning + LightGBM + walk-forward validation
python3 models/src/train_production.py --crypto BTC --tune --walk-forward

# Full professional pipeline (both coins, tuning, walk-forward)
python3 models/src/train_production.py --crypto BTC,ETH --tune --tune-trials 20 \
    --tune-models dlinear,tcn,lstm,transformer,lightgbm --walk-forward --epochs 150

# Skip specific models
python3 models/src/train_production.py --crypto BTC --no-dlinear --no-tcn

# All available options
python3 models/src/train_production.py --help
```

### CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--crypto` | Comma-separated crypto tickers | BTC |
| `--epochs` | Training epochs for deep learning models | 100 |
| `--batch-size` | Batch size | 32 |
| `--seq-length` | Lookback window in days | 60 |
| `--max-features` | Max features to select | 50 |
| `--no-dlinear` | Skip DLinear model training | Off |
| `--no-tcn` | Skip TCN model training | Off |
| `--no-ensemble` | Skip ensemble training | Off |
| `--output-dir` | Output directory for plots and reports | training_output/ |
| `--tune` | Run Optuna Bayesian hyperparameter optimization | Off |
| `--tune-trials N` | Number of Optuna trials per model | 20 |
| `--tune-timeout N` | Max seconds for tuning per model | 3600 |
| `--tune-models STR` | Comma-separated models to tune | lstm,transformer,lightgbm |
| `--walk-forward` | Walk-forward cross-validation after training | Off |
| `--wf-splits N` | Number of walk-forward splits | 5 |
| `--wf-min-train N` | Min training samples per walk-forward fold | 1000 |
| `--wf-rolling` | Use rolling window instead of expanding | Off |
| `--no-uncertainty` | Disable Monte Carlo dropout uncertainty quantification | Off |
| `--mc-samples N` | MC dropout samples for confidence intervals | 50 |
| `--no-advanced-mlflow` | Disable interactive MLflow curves and model cards | Off |

### Output Structure

```
models/src/training_output/{CRYPTO}_{timestamp}/
    loss_curves.png
    metrics_progression.png
    learning_rates.png
    attention_heatmap.png
    feature_importance.png
    model_comparison.png
    predictions_vs_actual.png
    residual_analysis.png
    ensemble_weights.png
    training_summary.png
    training_report.json

models/artifacts/
    dlinear/           # .h5 model + metadata JSON
    tcn/               # .h5 model + metadata JSON
    enhanced_lstm/     # .weights.h5 + metadata JSON
    transformer/       # .weights.h5 + metadata JSON
    lightgbm/          # .txt model files + config JSON
    advanced_ensemble/ # Ensemble metadata + meta-learner
```

---

## Feature Engineering

The input features (150+ per cryptocurrency) fall into these categories:

| Category | Count | Examples |
|----------|-------|---------|
| Price-derived | ~40 | Returns (1-30d), log returns, momentum, spreads, gaps |
| Moving averages | ~29 | SMA/EMA at 5/10/20/50/100/200 periods, crossovers, MACD |
| Momentum | ~16 | RSI (7/14/21), Stochastic K%/D%, Williams %R, ROC |
| Volatility | ~25 | Bollinger Bands, ATR, rolling std, Keltner Channels |
| Volume | ~10 | OBV, VPT, volume ratios, volume breakout flags |
| Statistical | ~11 | Skewness, kurtosis, z-scores, autocorrelations |
| Time-based | ~14 | Day of week, month, quarter (cyclical sine/cosine encoding) |
| Regime | ~5 | Bull/bear flags, high-volatility regime, market stress |
| Cross-asset | ~6 | ETH returns, RSI, volatility, ETH/BTC ratio (BTC only) |

Feature selection narrows this to the top 80 by mutual information regression (computed on training data only), capturing nonlinear feature-target relationships while preventing data leakage.

---

## MLflow Experiment Tracking

All training runs are automatically logged to MLflow (local file store):

- **Parameters**: model configs, sequence length, epochs, batch size, feature count
- **Metrics**: per-epoch loss, MAE, MSE, learning rate; final test RMSE/MAE/R-squared per horizon
- **Artifacts**: saved model weights, visualization PNGs, training report JSON

Access the MLflow UI:
```bash
mlflow ui --backend-store-uri file://models/src/mlruns_production --port 5000
```

---

## Training Results (February 2026 — Run 2, post-fixes)

Production training run across 5 cryptocurrencies (BTC, ETH, LTC, XRP, DOGE) with Optuna hyperparameter tuning (20 trials per model), walk-forward validation, MC dropout uncertainty quantification, and advanced MLflow logging. Run on Apple Silicon (Metal GPU). Total wall-clock time: ~18 hours.

### Pipeline fixes applied before this run

1. **Directional accuracy**: Fixed `np.sign(np.diff(y))` to `np.sign(y)` for log-return targets (was measuring change-of-change instead of direction, making LightGBM show 5-16% DA)
2. **Ensemble weighting**: Performance-based inverse-squared-RMSE weights replace static regime weights; models > 2x worse than best are excluded; regime weight interpolation disabled when validation weights are active
3. **Transformer MC Dropout**: Added `MCDropout` class (always-on dropout) for output heads to enable Monte Carlo uncertainty on TF-Metal

### Validation RMSE by Model (1d horizon, log-return space)

| Model | BTC | ETH | LTC | XRP | DOGE |
|-------|-----|-----|-----|-----|------|
| DLinear | 1.35 | 0.87 | 0.84 | 1.03 | 1.31 |
| TCN | 0.91 | 0.84 | 0.84 | 1.02 | 1.29 |
| LSTM | 0.91 | 0.83 | 0.84 | 1.02 | 1.28 |
| Transformer | 0.92 | 0.83 | 0.83 | 1.02 | 1.27 |
| LightGBM (MAE) | 0.61 | 0.59 | 0.57 | 0.64 | 0.92 |

### Train/Validation Gap (overfitting ratio, 1d horizon)

| Model | BTC | ETH | LTC | XRP | DOGE |
|-------|-----|-----|-----|-----|------|
| DLinear | 1.31x | 1.27x | 1.33x | 1.28x | 1.30x |
| TCN | 1.91x | 1.32x | 1.30x | 1.31x | 1.25x |
| LSTM | 1.93x | 1.35x | 1.30x | 1.32x | 1.30x |
| Transformer | 1.92x | 1.37x | 1.32x | 1.33x | 1.31x |

All models show healthy train/val gaps (< 2x), a major improvement from the initial BTC runs that had 2.7-5.8x gaps before the overfitting fixes (log-return targets, mutual info feature selection, L2 regularization, data augmentation, reduced capacity).

### Best Individual Model (test RMSE) by Coin

| Coin | Best Model | Test RMSE | 2nd Best | Ensemble RMSE | Ensemble vs Best |
|------|-----------|-----------|----------|---------------|------------------|
| BTC | LightGBM | 0.716 | LSTM (0.718) | 0.739 | -3.3% |
| ETH | LSTM | 0.948 | LightGBM (0.950) | 0.958 | -1.1% |
| LTC | TCN | 0.818 | Transformer (0.820) | 0.821 | -0.3% |
| XRP | LSTM | 0.900 | LightGBM (0.909) | 0.933 | -3.6% |
| DOGE | LSTM | 1.085 | LightGBM (1.087) | 1.097 | -1.1% |

The ensemble uses performance-based inverse-squared-RMSE weights computed from validation predictions. Models with RMSE > 2x the best are excluded (DLinear excluded for BTC). The meta-learner consistently falls back to the weighted average (LightGBM meta-learner can't beat the best individual on cross-validated validation splits). The ensemble is within -0.3% to -3.6% of the best individual model — a significant improvement from the previous run's -0.4% to -4.4%.

### Ensemble Weights by Coin

| Model | BTC | ETH | LTC | XRP | DOGE |
|-------|-----|-----|-----|-----|------|
| DLinear | 12.5% | 19.1% | 20.3% | 20.5% | 19.3% |
| TCN | 21.6% | 20.4% | 19.8% | 20.0% | 20.2% |
| Transformer | 21.6% | 19.4% | 20.1% | 19.8% | 19.8% |
| LSTM | 22.3% | 20.6% | 19.8% | 19.6% | 20.4% |
| LightGBM | 22.0% | 20.5% | 20.0% | 20.1% | 20.3% |

For ETH, LTC, XRP, DOGE: all 5 models have similar RMSE, so weights are near-uniform (~20% each). For BTC: DLinear receives lower weight (12.5%) but is not excluded because its validation RMSE (1.35) is within 2x of the best (0.91).

### Directional Accuracy (1-day horizon)

| Model | BTC | ETH | LTC | XRP | DOGE |
|-------|-----|-----|-----|-----|------|
| DLinear | 51.7% | **51.5%** | **54.0%** | 49.6% | 49.6% |
| TCN | **52.6%** | 47.1% | 46.7% | 50.4% | **54.0%** |
| LSTM | 52.2% | 48.9% | 47.8% | 51.1% | 50.4% |
| Transformer | 50.9% | 49.6% | 49.6% | **51.8%** | **57.0%** |
| LightGBM | **53.3%** | 47.4% | 50.4% | 50.0% | 53.7% |

Directional accuracy ranges from 47-57% across all models and coins. This is now correctly computed as `sign(log_return)` (positive = price up). LightGBM and TCN tend to have the highest DA for BTC; Transformer leads for DOGE (57.0%). The modest above-random performance is consistent with the efficient market hypothesis for liquid crypto assets.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs (max) | 150 |
| Early stopping patience | 10 |
| Batch size | 32 |
| Sequence length | 60 days |
| Max features | 50 (mutual information) |
| Train/val/test split | 70% / 15% / 15% |
| Optuna trials | 20 per model |
| MC dropout samples | 50 |

### Epochs Trained (early stopped)

| Model | BTC | ETH | LTC | XRP | DOGE |
|-------|-----|-----|-----|-----|------|
| DLinear | 33 | 30 | 17 | 45 | 22 |
| TCN | 14 | 19 | 19 | 47 | 19 |
| LSTM | 37 | 35 | 53 | 87 | 42 |
| Transformer | 15 | 30 | 31 | 21 | 15 |

LSTM consistently trains the longest before early stopping triggers. XRP requires the most epochs across all models (47-87), suggesting its return distribution is harder to fit.

### Training Time (Apple Silicon M3 Max, Metal GPU)

| Model | BTC | ETH | LTC | XRP | DOGE |
|-------|-----|-----|-----|-----|------|
| DLinear | 2.2 min | 0.4 min | 0.3 min | 0.6 min | 0.4 min |
| TCN | 10.7 min | 5.5 min | 5.6 min | 12.1 min | 34.5 min |
| LSTM | 51.9 min | 15.7 min | 26.4 min | 50.1 min | 30.2 min |
| Transformer | 17.2 min | 9.4 min | 14.2 min | 10.1 min | 5.0 min |
| LightGBM | 0.02 min | 0.01 min | 0.02 min | 0.02 min | 0.01 min |
| **Total** | **~82 min** | **~31 min** | **~47 min** | **~73 min** | **~70 min** |

Times include Optuna hyperparameter tuning (20 trials per model, subprocess-isolated). DOGE TCN took 34.5 min due to its tuned configuration requiring more compute per epoch.

### Optuna-Tuned Hyperparameters

| Parameter | BTC | ETH | LTC | XRP | DOGE |
|-----------|-----|-----|-----|-----|------|
| DLinear kernel | 17 | 27 | 35 | 27 | 27 |
| DLinear dropout | 0.17 | 0.09 | 0.22 | 0.09 | 0.09 |
| TCN filters | — | 64 | — | 16 | 32 |
| TCN blocks | — | 4 | — | 4 | 3 |
| LSTM layers x units | — | — | — | — | 3x128 bidir |
| LSTM dropout | — | — | — | — | 0.45 |
| Transformer d_model | 128 | — | 64 | 128 | — |
| Transformer layers | 3 | — | 3 | 3 | — |
| Transformer dropout | 0.31 | — | 0.13 | 0.31 | — |
| LightGBM leaves | 26 | 39 | 64 | 18 | 10 |
| LightGBM estimators | 223 | 351 | 146 | 174 | 452 |
| LightGBM depth | 9 | 4 | 5 | 7 | 5 |

"—" indicates default parameters were used (Optuna tuning not run or reused from another coin).

### Uncertainty Quantification (MC Dropout, 50 samples)

| Model | BTC CI | ETH CI | LTC CI | XRP CI | DOGE CI |
|-------|--------|--------|--------|--------|---------|
| DLinear | 25.4 | 1.67 | 1.07 | 8.12 | 2.80 |
| TCN | 0.63 | 1.07 | 0.83 | 1.68 | 2.66 |
| LSTM | 0.06 | 0.18 | 0.06 | 0.06 | 0.005 |
| Transformer | 0.0 | 0.0 | ~0.0 | 0.0 | 0.001 |

TCN provides the most useful uncertainty estimates (meaningful CI widths that correlate with prediction difficulty). LSTM CIs are narrow but non-zero. Transformer MCDropout activates on DOGE (0.001 CI) but remains near-zero for other coins — the output-head-only MCDropout has minimal effect when the transformer body produces deterministic representations. DLinear CIs are wide and noisy due to the model's high base variance.

---

## Hardware Requirements

| Setup | DLinear/TCN Epoch | LSTM Epoch | Transformer Epoch | Full Training (5 models) |
|-------|-------------------|-----------|-------------------|--------------------------|
| CPU only | ~1-3 sec | ~5-10 min | ~3-8 min | Not recommended |
| Apple M1/M2/M3 (Metal) | ~2-5 sec | ~15-30 sec | ~10-20 sec | ~30-60 min per coin |
| Google Colab T4 | ~1-3 sec | ~5-15 sec | ~3-10 sec | ~15-30 min per coin |
| A100 GPU | ~1-2 sec | ~2-5 sec | ~1-3 sec | ~5-10 min per coin |

**Apple Silicon Metal GPU Setup:**
```bash
# Requires Python 3.12 (tensorflow-metal does not support 3.13+)
python3.12 -m venv tf-gpu-env
source tf-gpu-env/bin/activate
pip install tensorflow==2.18 tensorflow-metal
pip install lightgbm optuna scikit-learn pandas pyarrow mlflow matplotlib seaborn

# Verify GPU detection
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
