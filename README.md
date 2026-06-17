# SPY LSTM Trend Prediction

> Comparing deep learning architectures for next-day S&P 500 direction forecasting across 14 years of market data.

📊 **[Live Research Dashboard](https://heetm77.github.io/spy-lstm-trend-prediction/reports/dashboard.html)**

---

## Overview

This project investigates whether LSTM-based deep learning models can predict the next-day directional movement of the S&P 500 ETF (SPY) better than random chance. Five architectures are trained, evaluated, and backtested on 14 years of daily data (2010–2024).

**Key finding:** All models achieved ROC-AUC scores near 0.50–0.53, consistent with the Efficient Market Hypothesis — next-day SPY direction is largely unpredictable from price history and technical indicators alone. The project documents this rigorously rather than overfitting to noise.

---

## Models

| Architecture | Parameters | Notes |
|---|---|---|
| LSTM 1-Layer | ~23K | Baseline recurrent model |
| LSTM 2-Layer | ~38K | Stacked with halved units |
| LSTM 5-Layer | ~95K | Deep recurrent, prone to overfitting |
| GRU | ~19K | Gated recurrent, fastest convergence |
| CNN-LSTM | ~41K | Convolutional feature extraction + LSTM |

---

## Feature Sets

**V1 (24 features):** OHLCV, RSI, MACD, Bollinger Bands, SMA crossovers, volume ratios, lagged returns

**V2 (33 features):** V1 + VIX (fear index), sector ETF relative strength (XLK, XLF, XLE, XLV, XLI), volume-return correlation

---

## Results

### Test Set Performance (V1 Features)

| Model | Accuracy | ROC-AUC | F1 |
|---|---|---|---|
| LSTM 1-Layer | 56.3% | 0.519 | 0.042 |
| LSTM 2-Layer | 55.0% | **0.532** | 0.105 |
| LSTM 5-Layer | 55.4% | 0.525 | 0.169 |
| GRU | 56.2% | 0.518 | 0.024 |
| CNN-LSTM | 55.9% | 0.508 | 0.041 |

### Backtest (LSTM 1-Layer, default threshold)

| Metric | Strategy | SPY Buy & Hold |
|---|---|---|
| Cumulative Return | 21.78% | 86.11% |
| Sharpe Ratio | 0.59 | 1.10 |
| Max Drawdown | -13.57% | -19.00% |
| Days in Market | 106 / 947 | 947 / 947 |

The strategy's lower drawdown reflects its conservative signal — it was in the market only 11% of the time.

---

## Project Structure

```
├── src/
│   ├── create_sequences.py       # V1 sequence generation
│   ├── create_sequences_v2.py    # V2 sequence generation (33 features)
│   ├── models.py                 # All 5 model architectures
│   ├── train.py                  # Single model training
│   ├── train_all.py              # Train all models (V1)
│   ├── train_all_v2.py           # Train all models (V2)
│   ├── evaluate.py               # Single model evaluation
│   ├── evaluate_all.py           # Compare all models (V1)
│   ├── evaluate_all_v2.py        # Compare all models (V2)
│   ├── backtest.py               # Strategy backtesting
│   ├── tune_threshold.py         # Decision threshold analysis
│   ├── add_features.py           # Feature engineering pipeline
│   └── build_dashboard.py        # Interactive HTML dashboard
├── reports/
│   ├── dashboard.html            # Interactive results dashboard
│   ├── v2/                       # V2 model results
│   └── *.csv                     # Metrics, predictions, histories
└── data/
    └── processed/
        ├── sequences/            # V1 numpy sequences
        └── sequences_v2/         # V2 numpy sequences
```

## Setup

```bash
git clone https://github.com/Heetm77/spy-lstm-trend-prediction.git
cd spy-lstm-trend-prediction
python -m venv .venv
.venv\Scripts\activate
pip install tensorflow yfinance pandas numpy scikit-learn
```

### Reproduce Results

```bash
# Download data and engineer features
python src/add_features.py

# Build sequences
python src/create_sequences_v2.py

# Train all models
python src/train_all_v2.py

# Evaluate
python src/evaluate_all_v2.py

# Backtest
python src/backtest.py
```

---

## Key Observations

1. **Market efficiency holds** — ROC-AUC scores of 0.50–0.53 across all architectures suggest next-day SPY direction contains minimal exploitable signal in price/volume data alone.

2. **Overfitting is the dominant failure mode** — V2 models with 33 features showed training accuracy reaching 65%+ while validation accuracy stayed near 50%, a clear sign of overfitting to noise.

3. **Depth doesn't help** — The 5-layer LSTM performed similarly to the 1-layer baseline, suggesting the problem is data quality, not model capacity.

4. **Lower drawdown is a silver lining** — The strategy's max drawdown of -13.57% vs -19.00% for buy-and-hold shows that even a weak signal can improve risk-adjusted exposure when used conservatively.

---

## Future Work

- Alternative labels (5-day forward returns, volatility prediction)
- Attention mechanisms and Transformer architectures
- Macro features (Fed funds rate, yield curve, dollar index)
- Regime detection (bull/bear market conditioning)
- Ensemble methods combining multiple weak classifiers

---

## Tech Stack

`Python` · `TensorFlow/Keras` · `scikit-learn` · `pandas` · `yfinance` · `Chart.js`

---

*Independent research project · 2024–2025*