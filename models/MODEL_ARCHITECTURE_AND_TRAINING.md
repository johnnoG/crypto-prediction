# Model Architecture & Training Methodology

## Why Three Models?

Cryptocurrency markets exhibit fundamentally different statistical behaviors depending on time scale, regime, and market conditions. No single model architecture dominates across all scenarios. Our approach trains three complementary architectures and combines them through an intelligent ensemble:

| Property | LSTM | Transformer | LightGBM |
|----------|------|-------------|----------|
| **Strength** | Sequential memory, local patterns | Long-range dependencies, global patterns | Tabular feature interactions |
| **Weakness** | Struggles with long sequences | Data-hungry, computationally heavy | No native temporal modeling |
| **Best regime** | Trending markets | Volatile / regime-shifting | Mean-reverting / feature-rich |
| **Inference speed** | Medium (~32ms) | Slow (~45ms) | Fast (~8ms) |
| **Parameters** | ~306K | ~2.5M | ~50K |

By combining all three, the ensemble captures short-term momentum (LSTM), long-range structural shifts (Transformer), and complex feature interactions (LightGBM) simultaneously. The ensemble's market regime detector dynamically re-weights contributions based on current conditions.

---

## Model 1: Enhanced LSTM

**File:** `src/models/enhanced_lstm.py`
**Class:** `EnhancedLSTMForecaster`

### Why LSTM?

LSTMs are the workhorse of sequential prediction. They maintain an internal cell state that acts as memory, allowing them to learn how recent price action, volume spikes, and technical indicator crossovers predict near-future movements. For cryptocurrency markets, where short-term momentum and mean-reversion patterns dominate intraday and daily timeframes, LSTMs are a natural fit.

### Architecture

```
Input (60 timesteps x 51 features)
  |
  LayerNormalization
  |
  Bidirectional LSTM (128 units) + Residual Connection
  |  Dropout (0.2)
  |
  Residual LSTM (64 units)
  |  Dropout (0.2)
  |
  Residual LSTM (32 units)
  |
  Attention Layer (learnable position encoding)
  |
  Dense (64) -> LayerNorm -> Dense (32) -> LayerNorm
  |         |         |
  Head 1d   Head 7d   Head 30d   (multi-step output)
```

### Key Design Choices

- **Bidirectional first layer**: Captures both forward and backward temporal context. Only the first layer is bidirectional to control parameter count.
- **Residual connections**: Each LSTM cell wraps a standard LSTM with a linear projection skip-connection. This prevents gradient vanishing in deeper stacks and allows the model to learn incremental refinements.
- **Attention mechanism**: After the LSTM stack, an attention layer computes learned weights over all 60 timesteps. This lets the model focus on the most predictive time points (e.g., a sudden volume spike 12 days ago) rather than treating all timesteps equally.
- **Multi-step heads**: Separate output heads for 1-day, 7-day, and 30-day horizons. Shorter horizons receive higher loss weights (`1/sqrt(h)`) because near-term predictions are more actionable and more feasible.
- **Monte Carlo dropout**: At inference time, dropout remains active across multiple forward passes (default 50 samples). The variance of predictions provides a calibrated uncertainty estimate without requiring a separate model.

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sequence length | 60 | ~2 months of daily data; captures medium-term patterns |
| LSTM units | [128, 64, 32] | Tapering width forces compression of learned representations |
| Learning rate | 0.001 | Standard Adam starting point with ReduceLROnPlateau |
| Gradient clipping | 1.0 | Prevents exploding gradients common in financial time series |
| Early stopping patience | 20 | Generous patience since crypto data is noisy |
| Teacher forcing ratio | 0.5 | Balances training stability with inference-time robustness |

---

## Model 2: Transformer

**File:** `src/models/transformer_model.py`
**Class:** `TransformerForecaster`

### Why Transformers?

Transformers process the entire input sequence in parallel through self-attention, computing pairwise relationships between all timesteps simultaneously. This gives them two advantages over LSTMs for crypto prediction:

1. **Long-range dependencies**: Self-attention can directly connect a price pattern from 60 days ago to today's prediction without information passing through 60 sequential gates.
2. **Multi-scale pattern detection**: Different attention heads naturally specialize on different temporal scales - some learn daily patterns, others weekly cycles, others regime transitions.

### Architecture

```
Input (60 timesteps x 51 features)
  |
  Dense Projection -> d_model (256)
  |
  Positional Encoding (learnable)
  |  Dropout (0.1)
  |
  Transformer Block x4:
  |   Multi-Head Self-Attention (8 heads, causal mask)
  |   Add & LayerNorm
  |   Feed-Forward (1024 -> 256)
  |   Add & LayerNorm
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
- **Warmup learning rate schedule**: The standard Transformer schedule (`1/sqrt(d_model) * min(1/sqrt(step), step * warmup^{-1.5})`) starts low, ramps up during warmup, then decays. This is critical because the attention weights are initially random and large gradients early on can destabilize training.
- **Reduced dimensions (d_model=256, 4 layers)**: Full-scale Transformers (512d, 6 layers) are designed for NLP datasets with millions of sequences. Our dataset (~5000 samples) requires a smaller model to avoid overfitting.
- **Huber loss per horizon**: Huber loss is less sensitive to outliers than MSE, which matters because crypto prices occasionally have extreme moves that would otherwise dominate the gradient.

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 256 | Balanced capacity for ~5K sample dataset |
| Attention heads | 8 | 256/8 = 32 dims per head; allows multi-scale pattern learning |
| Feed-forward dim | 1024 | 4x d_model (standard ratio) |
| Transformer layers | 4 | Deeper than needed for sequential data, but attention makes each layer efficient |
| Warmup steps | 500 | ~15 epochs of warmup at batch_size=32 |
| Dropout | 0.1 | Light regularization; early stopping handles overfitting |

---

## Model 3: LightGBM

**File:** `src/models/lightgbm_model.py`
**Class:** `LightGBMForecaster`

### Why Gradient Boosting?

Gradient boosted trees excel at learning non-linear feature interactions from tabular data. While LSTM and Transformer operate on raw sequences, LightGBM consumes the **flattened feature matrix** (60 timesteps x 50 features = 3000 input features). This gives it a fundamentally different view of the data:

1. **Feature interactions**: It can discover that "RSI < 30 AND volume > 2x average AND BTC correlation > 0.8" is a strong buy signal, without needing to learn temporal convolutions.
2. **Speed and interpretability**: Training takes minutes, not hours. Feature importance rankings provide direct insight into which indicators drive predictions.
3. **Robustness to noise**: Tree ensembles are naturally resistant to feature scaling issues, outliers, and irrelevant features.

LightGBM serves as a strong baseline and a diversity driver in the ensemble. Its errors are typically uncorrelated with the deep learning models, which is exactly what makes an ensemble effective.

### Architecture

```
Input (60 timesteps x 50 features) -> Flatten to 3000 features
  |
  GBDT for 1-day horizon (up to 500 trees)
  GBDT for 7-day horizon (up to 500 trees)
  GBDT for 30-day horizon (up to 500 trees)
```

Each horizon gets its own independent model because the optimal tree structure for 1-day prediction is very different from 30-day prediction.

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| num_leaves | 63 | Moderate complexity; prevents overfitting on ~3700 training samples |
| learning_rate | 0.05 | Low rate + many trees = smooth convergence |
| feature_fraction | 0.9 | Mild column subsampling for diversity |
| bagging_fraction | 0.8 | Row subsampling reduces variance |
| early_stopping | 20 rounds | Stops adding trees when validation MAE plateaus |

---

## Model 4: Advanced Ensemble

**File:** `src/models/advanced_ensemble.py`
**Class:** `AdvancedEnsemble`

### Why an Ensemble?

The three base models make different types of errors. The LSTM might miss a structural regime change that the Transformer catches. LightGBM might nail a feature-interaction signal that the deep learning models overlook. An ensemble exploits this error diversity.

Our ensemble goes beyond simple averaging with three techniques:

### 1. Market Regime Detection

The `MarketRegimeDetector` classifies the current market into one of four regimes using price statistics:

| Regime | Detection Logic | Typical Weight Adjustment |
|--------|----------------|--------------------------|
| **Trending** | Strong directional momentum, low noise | LSTM weight increases (momentum-sensitive) |
| **Mean-reverting** | Oscillating around a level, high autocorrelation | LightGBM weight increases (feature-driven) |
| **Volatile** | High variance, large swings | Transformer weight increases (long-range context) |
| **Ranging** | Low volatility, tight bands | Equal weighting |

### 2. Meta-Learner (Stacking)

An ElasticNet regression is trained on the base model predictions as features, with actual prices as targets. This learns the optimal linear combination and can discover that, for example, "when LSTM and Transformer disagree, trust LightGBM."

### 3. Online Weight Adaptation

After each prediction, the ensemble updates model weights based on recent accuracy using exponential moving averages. Models that performed well in the last 50 predictions receive higher weight, enabling real-time adaptation.

### Default Weights

| Model | Weight | Rationale |
|-------|--------|-----------|
| Transformer | 0.40 | Best overall accuracy in backtesting |
| LSTM | 0.35 | Strong on trending / momentum markets |
| LightGBM | 0.25 | Fast inference, stable baseline, error diversity |

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
[2] Feature Selection: top 50 features by correlation with close price
     |
     v
[3] Scale: RobustScaler (fit on train only) - handles crypto outliers
     |
     v
[4] Split: 70% train / 15% validation / 15% test (chronological, no shuffle)
     |
     v
[5] Sequence: sliding window of 60 days, target = close at +1/+7/+30 days
     |
     v
[6] Train: LSTM -> Transformer -> LightGBM -> Ensemble
     |
     v
[7] Evaluate: per-horizon RMSE/MAE/R2 on held-out test set
     |
     v
[8] Visualize: 10 PNG plots saved to training_output/
     |
     v
[9] Save: model weights to models/artifacts/
```

### Why Chronological Splits (Not Random)?

Cryptocurrency prices are non-stationary time series. A random train/test split would leak future information into training, producing artificially inflated metrics. We use strict chronological ordering:

- **Train**: oldest 70% of data (learns historical patterns)
- **Validation**: next 15% (tunes hyperparameters, triggers early stopping)
- **Test**: newest 15% (final unbiased evaluation)

### Why RobustScaler?

Crypto prices have extreme outliers (flash crashes, parabolic rallies). `RobustScaler` uses the interquartile range instead of standard deviation, making it far more resistant to these events than `StandardScaler` or `MinMaxScaler`.

### Multi-Step Forecasting

All models predict three horizons simultaneously:

| Horizon | Use Case | Difficulty |
|---------|----------|------------|
| **1 day** | Short-term trading signals | Easiest (most autocorrelation) |
| **7 days** | Swing trading, position sizing | Medium |
| **30 days** | Portfolio allocation, risk management | Hardest (most uncertainty) |

Shorter horizons receive higher loss weights during training because they are more predictable and more commercially valuable.

---

## Training Visualizations

The training script generates the following plots for each cryptocurrency:

| Plot | What It Shows |
|------|---------------|
| `loss_curves.png` | Train vs validation loss per epoch for each model |
| `metrics_progression.png` | MAE/MSE per horizon over training epochs |
| `learning_rates.png` | Learning rate schedules (decay, warmup, plateau reduction) |
| `attention_heatmap.png` | LSTM attention weights showing which timesteps the model focuses on |
| `feature_importance.png` | Top 30 LightGBM features ranked by importance |
| `model_comparison.png` | Side-by-side RMSE/MAE bars for all models at each horizon |
| `predictions_vs_actual.png` | Scatter plots with R-squared per model per horizon |
| `residual_analysis.png` | Residual distributions and residual-vs-predicted scatter |
| `ensemble_weights.png` | Contribution of each base model in the ensemble |
| `training_summary.png` | Overview panel: training times, convergence, epochs, final RMSE |

---

## Running Training

```bash
# Train on BTC and ETH (full training)
python models/src/train_production.py --crypto BTC,ETH

# Quick test run (3 epochs)
python models/src/train_production.py --crypto BTC --epochs 3

# Skip ensemble (faster, individual models only)
python models/src/train_production.py --crypto BTC --no-ensemble

# Custom configuration
python models/src/train_production.py --crypto BTC,ETH --epochs 50 --batch-size 64 --seq-length 30
```

### Output

```
models/src/training_output/{CRYPTO}_{timestamp}/
    *.png             # All visualization plots
    training_report.json   # Machine-readable metrics

models/artifacts/
    enhanced_lstm/    # Saved LSTM weights (.h5) + metadata
    transformer/      # Saved Transformer weights (.h5) + metadata
    lightgbm/         # Saved LightGBM trees (.txt) + config
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

Feature selection narrows this to the top 50 by absolute Pearson correlation with the close price. This removes noise features and reduces overfitting risk.

---

## MLflow Experiment Tracking

All training runs are logged to MLflow (local file store by default):

- **Parameters**: model configs, sequence length, epochs, feature count
- **Metrics**: per-epoch loss, MAE, MSE, learning rate; final test RMSE/MAE/R2
- **Artifacts**: saved model weights, visualization PNGs, training report JSON

Access the MLflow UI:
```bash
mlflow ui --backend-store-uri file://models/src/mlruns_production --port 5000
```
