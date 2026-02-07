# Model Architecture & Training Methodology

## Why Three Models?

Cryptocurrency markets exhibit fundamentally different statistical behaviors depending on time scale, regime, and market conditions. No single model architecture dominates across all scenarios. Our approach trains three complementary architectures and combines them through an intelligent ensemble:

| Property | LSTM | Transformer | LightGBM |
|----------|------|-------------|----------|
| **Strength** | Sequential memory, local patterns | Long-range dependencies, global patterns | Tabular feature interactions |
| **Weakness** | Struggles with long sequences | Data-hungry, computationally heavy | No native temporal modeling |
| **Best regime** | Trending markets | Volatile / regime-shifting | Mean-reverting / feature-rich |
| **Inference speed** | Medium (~32ms) | Slow (~45ms) | Fast (~8ms) |

By combining all three, the ensemble captures short-term momentum (LSTM), long-range structural shifts (Transformer), and complex feature interactions (LightGBM) simultaneously. The ensemble's market regime detector dynamically re-weights contributions based on current conditions.

---

## Model 1: Enhanced LSTM

**File:** `src/models/enhanced_lstm.py`
**Class:** `EnhancedLSTMForecaster`

### Why LSTM?

LSTMs are the workhorse of sequential prediction. They maintain an internal cell state that acts as memory, allowing them to learn how recent price action, volume spikes, and technical indicator crossovers predict near-future movements. For cryptocurrency markets, where short-term momentum and mean-reversion patterns dominate intraday and daily timeframes, LSTMs are a natural fit.

### Architecture

```
Input (60 timesteps x 61 features)
  |
  LayerNormalization
  |
  Bidirectional LSTM (128 units) + Dropout (0.4)
  |
  Residual LSTM (64 units) + LayerNorm + Dropout (0.4)
  |
  Residual LSTM (32 units) + LayerNorm
  |
  Attention Layer (learnable weights over all 60 timesteps)
  |
  Dense (64) -> LayerNorm -> Dropout -> Dense (32) -> LayerNorm -> Dropout
  |              |              |
  Head 1d        Head 7d        Head 30d   (multi-step output)
```

### Key Design Choices

- **Bidirectional first layer**: Captures both forward and backward temporal context. Only the first layer is bidirectional to control parameter count while the residual layers process the combined representation sequentially.
- **Residual connections**: Each `ResidualLSTMCell` wraps a standard LSTM with a linear projection skip-connection and layer normalization. This prevents gradient vanishing in the three-layer stack and allows the model to learn incremental refinements rather than complete transformations at each layer.
- **Attention mechanism**: After the LSTM stack, a Bahdanau-style attention layer computes learned weights over all 60 timesteps. This lets the model focus on the most predictive time points (e.g., a sudden volume spike 12 days ago or a support level test 45 days ago) rather than relying solely on the final hidden state.
- **Multi-step heads**: Separate output heads for 1-day, 7-day, and 30-day horizons. Shorter horizons receive higher loss weights (`1/sqrt(h)`) because near-term predictions are more actionable and more feasible.
- **Recurrent dropout (0.2)**: Applied within the LSTM recurrence at each timestep, providing stronger regularization than standard dropout alone. Requires GPU (CuDNN) for acceptable training speed.
- **Monte Carlo dropout**: At inference time, dropout remains active across 100 forward passes. The variance of predictions provides a calibrated uncertainty estimate without requiring a separate model.

### Hyperparameters (Production)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sequence length | 60 | ~2 months of daily data; captures medium-term patterns |
| LSTM units | [128, 64, 32] | Three-layer stack with tapering width; reduced from [256,128,64] to combat overfitting on ~3700 training samples |
| Dense units | [64, 32] | Two dense layers before output heads for non-linear feature combination |
| Dropout | 0.4 | Aggressive regularization for noisy financial data with limited samples |
| Recurrent dropout | 0.2 | Per-timestep regularization within LSTM recurrence |
| Learning rate | 0.0005 | Adam with early stopping; lower LR for more stable convergence |
| Gradient clipping | 1.0 | Prevents exploding gradients common in financial time series |
| Batch size | 64 | Larger batches for GPU utilization and gradient stability |
| Epochs | 150 | With early stopping (patience=20) typically converges in 40-80 epochs |
| Loss function | Huber | Less sensitive to outliers than MSE; robust to crypto flash crashes |

---

## Model 2: Transformer

**File:** `src/models/transformer_model.py`
**Class:** `TransformerForecaster`

### Why Transformers?

Transformers process the entire input sequence in parallel through self-attention, computing pairwise relationships between all timesteps simultaneously. This gives them two advantages over LSTMs for crypto prediction:

1. **Long-range dependencies**: Self-attention can directly connect a price pattern from 60 days ago to today's prediction without information passing through 60 sequential gates.
2. **Multi-scale pattern detection**: Different attention heads naturally specialize on different temporal scales — some learn daily patterns, others weekly cycles, others regime transitions.

### Architecture

```
Input (60 timesteps x 61 features)
  |
  Dense Projection -> d_model (128)
  |
  Positional Encoding (learnable)
  |  Dropout (0.25)
  |
  Transformer Block x3:
  |   Multi-Head Self-Attention (4 heads, causal mask)
  |   Residual Add & LayerNorm
  |   Feed-Forward Network (512 -> GELU -> 128)
  |   Residual Add & LayerNorm
  |
  Global Average Pooling
  |
  Dense (128) -> Dropout -> Output 1d
  Dense (128) -> Dropout -> Output 7d
  Dense (128) -> Dropout -> Output 30d
```

### Key Design Choices

- **Causal masking**: The self-attention uses a lower-triangular mask so that each timestep can only attend to itself and earlier timesteps. This prevents information leakage from the future during training.
- **Learnable positional encoding**: Unlike fixed sinusoidal encoding, learnable embeddings allow the model to discover that "3 days before a prediction" has different significance than "45 days before."
- **Warmup learning rate schedule**: The standard Transformer schedule (`1/sqrt(d_model) * min(1/sqrt(step), step * warmup^{-1.5})`) starts with a very low rate, ramps up during 500 warmup steps, then decays. This is critical because the attention weights are initially random and large gradients early on can destabilize training.
- **GELU activation**: Used in the feed-forward network instead of ReLU. GELU provides smoother gradients which helps with the attention mechanism's sensitivity.
- **Huber loss per horizon**: Less sensitive to outliers than MSE, which matters because crypto prices occasionally have extreme moves that would otherwise dominate the gradient.

### Hyperparameters (Production)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 128 | Reduced from 256 to combat overfitting on ~3700 training samples; 128/4 = 32 dims per head |
| Attention heads | 4 | Multi-scale pattern learning; each head sees 32-dimensional projections |
| Feed-forward dim | 512 | 4x d_model (standard ratio from "Attention Is All You Need") |
| Transformer layers | 3 | Reduced from 4; fewer layers prevent overfitting on limited data |
| Warmup steps | 500 | Faster warmup for smaller model; reaches peak LR sooner |
| Dropout | 0.25 | Increased from 0.1; stronger regularization needed for small dataset |
| Optimizer | AdamW | Weight decay (0.01) provides additional regularization vs standard Adam |
| Batch size | 64 | Larger batches for GPU utilization |
| Epochs | 150 | With early stopping (patience=15) typically converges in 30-60 epochs |

---

## Model 3: LightGBM

**File:** `src/models/lightgbm_model.py`
**Class:** `LightGBMForecaster`

### Why Gradient Boosting?

Gradient boosted trees excel at learning non-linear feature interactions from tabular data. While LSTM and Transformer operate on raw sequences, LightGBM consumes the **flattened feature matrix** (60 timesteps x features). This gives it a fundamentally different view of the data:

1. **Feature interactions**: It can discover that "RSI < 30 AND volume > 2x average AND BTC correlation > 0.8" is a strong buy signal, without needing to learn temporal convolutions.
2. **Speed and interpretability**: Training takes seconds to minutes, not hours. Feature importance rankings provide direct insight into which indicators drive predictions.
3. **Robustness to noise**: Tree ensembles are naturally resistant to feature scaling issues, outliers, and irrelevant features.

LightGBM serves as a strong baseline and a diversity driver in the ensemble. Its errors are typically uncorrelated with the deep learning models, which is exactly what makes an ensemble effective.

### Architecture

```
Input (60 timesteps x 60 features) -> Flatten to 3600 features
  |
  GBDT for 1-day horizon  (up to 1000 trees, depth 8)
  GBDT for 7-day horizon  (up to 1000 trees, depth 8)
  GBDT for 30-day horizon (up to 1000 trees, depth 8)
```

Each horizon gets its own independent model because the optimal tree structure for 1-day prediction is very different from 30-day prediction.

### Hyperparameters (Production)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_estimators | 1000 | Large tree budget; early stopping prevents overfitting |
| num_leaves | 127 | High complexity capacity for the 3600-dimensional flattened input |
| max_depth | 8 | Limits individual tree complexity to prevent memorization |
| learning_rate | 0.03 | Low rate + many trees = smooth convergence and better generalization |
| feature_fraction | 0.85 | Column subsampling per tree for ensemble diversity |
| bagging_fraction | 0.8 | Row subsampling reduces variance |
| min_child_samples | 20 | Minimum leaf size prevents fitting to noise |
| reg_alpha | 0.1 | L1 regularization for feature selection |
| reg_lambda | 0.1 | L2 regularization for weight smoothing |
| early_stopping | 50 rounds | Stops when validation MAE plateaus |

---

## Model 4: Advanced Ensemble

**File:** `src/models/advanced_ensemble.py`
**Class:** `AdvancedEnsemble`

### Why an Ensemble?

The three base models make different types of errors. The LSTM might miss a structural regime change that the Transformer catches. LightGBM might nail a feature-interaction signal that the deep learning models overlook. An ensemble exploits this error diversity.

Our ensemble goes beyond simple averaging with three techniques:

### 1. Market Regime Detection

The `MarketRegimeDetector` classifies the current market into one of four regimes using price statistics:

| Regime | Detection Logic | Weight Adjustment |
|--------|----------------|--------------------------|
| **Trending** | Strong directional momentum, low noise | LSTM weight increases (momentum-sensitive) |
| **Mean-reverting** | Oscillating around a level, high autocorrelation | LightGBM weight increases (feature-driven) |
| **Volatile** | High variance, large swings | Transformer weight increases (long-range context) |
| **Ranging** | Low volatility, tight bands | Equal weighting |

### 2. Meta-Learner (Stacking)

An ElasticNet regression is trained on the base model predictions as features, with actual prices as targets. This learns the optimal linear combination and can discover that, for example, "when LSTM and Transformer disagree, trust LightGBM."

### 3. Online Weight Adaptation

After each prediction, the ensemble updates model weights based on recent accuracy using exponential moving averages. Models that performed well in the last 50 predictions receive higher weight, enabling real-time adaptation without retraining.

### Default Weights

| Model | Weight | Rationale |
|-------|--------|-----------|
| Transformer | 0.40 | Strongest on structural patterns and regime changes |
| LSTM | 0.35 | Best for trending / momentum-driven markets |
| LightGBM | 0.25 | Fast inference, stable baseline, uncorrelated errors |

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
[2] Feature Selection: top 60 features by absolute correlation with close price
     |
     v
[3] Scale: RobustScaler (fit on train split only)
     |
     v
[4] Split: 70% train / 15% validation / 15% test (strict chronological order)
     |
     v
[5] Sequence: sliding window of 60 days, target = close at +1/+7/+30 days
     |
     v
[6] Train: LSTM -> Transformer -> LightGBM -> Ensemble (sequential)
     |
     v
[7] Evaluate: per-horizon RMSE / MAE / R-squared on held-out test set
     |
     v
[8] Visualize: 10 PNG diagnostic plots saved to training_output/
     |
     v
[9] Save: model weights to models/artifacts/ + training report JSON
     |
     v
[10] Log: all metrics, configs, and artifacts to MLflow
```

### Why Chronological Splits (Not Random)?

Cryptocurrency prices are non-stationary time series. A random train/test split would leak future information into training, producing artificially inflated metrics. We use strict chronological ordering:

- **Train**: oldest 70% of data (learns historical patterns)
- **Validation**: next 15% (tunes hyperparameters, triggers early stopping)
- **Test**: newest 15% (final unbiased evaluation — never seen during training)

### Why RobustScaler?

Crypto prices have extreme outliers (flash crashes, parabolic rallies). `RobustScaler` uses the interquartile range instead of standard deviation, making it far more resistant to these events than `StandardScaler` or `MinMaxScaler`. Critically, the scaler is fit only on training data to prevent data leakage.

### Multi-Step Forecasting

All models predict three horizons simultaneously through separate output heads:

| Horizon | Use Case | Difficulty |
|---------|----------|------------|
| **1 day** | Short-term trading signals | Easiest (most autocorrelation) |
| **7 days** | Swing trading, position sizing | Medium |
| **30 days** | Portfolio allocation, risk management | Hardest (most uncertainty) |

Shorter horizons receive higher loss weights during training because they are more predictable and more commercially valuable.

---

## Training Visualizations

The training script generates 10 diagnostic plots per cryptocurrency:

| Plot | What It Shows |
|------|---------------|
| `loss_curves.png` | Train vs validation loss per epoch for each model |
| `metrics_progression.png` | MAE/MSE per horizon over training epochs |
| `learning_rates.png` | LR schedules (decay, warmup, plateau reduction) |
| `attention_heatmap.png` | LSTM attention weights showing which timesteps the model focuses on |
| `feature_importance.png` | Top 30 LightGBM features ranked by importance |
| `model_comparison.png` | Side-by-side RMSE/MAE bars for all models at each horizon |
| `predictions_vs_actual.png` | Scatter plots with R-squared per model per horizon |
| `residual_analysis.png` | Residual distributions and residual-vs-predicted scatter |
| `ensemble_weights.png` | Contribution of each base model in the ensemble |
| `training_summary.png` | Overview panel: training times, convergence, epochs, final RMSE |

---

## Running Training

**GPU is required for production training.** Google Colab (T4/A100) is recommended. On a T4 GPU, full training takes approximately 15-30 minutes per cryptocurrency.

```bash
# Full production training for BTC and ETH
python models/src/train_production.py --crypto BTC,ETH

# Quick test run (3 epochs, skip ensemble)
python models/src/train_production.py --crypto BTC --epochs 3 --no-ensemble

# With Optuna hyperparameter tuning
python models/src/train_production.py --crypto BTC --tune --tune-trials 20

# Tune LightGBM only (fast) + walk-forward validation
python models/src/train_production.py --crypto BTC --tune --tune-models lightgbm --walk-forward

# Full professional pipeline
python models/src/train_production.py --crypto BTC --tune --walk-forward --epochs 150

# All available options
python models/src/train_production.py --help
```

### Advanced Pipeline Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--tune` | Run Optuna Bayesian hyperparameter optimization before training | Off |
| `--tune-trials N` | Number of Optuna trials per model | 20 |
| `--tune-timeout N` | Max seconds for tuning per model | 3600 |
| `--tune-models STR` | Comma-separated models to tune | lstm,transformer,lightgbm |
| `--walk-forward` | Walk-forward cross-validation after training | Off |
| `--wf-splits N` | Number of walk-forward splits | 5 |
| `--no-uncertainty` | Disable Monte Carlo dropout uncertainty quantification | On |
| `--mc-samples N` | MC dropout samples for confidence intervals | 50 |
| `--no-advanced-mlflow` | Disable interactive MLflow curves and model cards | On |

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

Feature selection narrows this to the top 60 by absolute Pearson correlation with the close price, removing noise features and reducing overfitting risk.

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

## Hardware Requirements

| Setup | LSTM Epoch | Transformer Epoch | Full Training |
|-------|-----------|-------------------|---------------|
| CPU (M1 Mac) | ~5-10 min | ~3-8 min | Not recommended |
| Google Colab T4 | ~5-15 sec | ~3-10 sec | ~15-30 min per coin |
| A100 GPU | ~2-5 sec | ~1-3 sec | ~5-10 min per coin |

The LSTM with `recurrent_dropout > 0` requires GPU for the CuDNN kernel. On CPU, recurrent dropout forces a pure Python fallback that is orders of magnitude slower.
