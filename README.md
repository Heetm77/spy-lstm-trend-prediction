# Deep Learning for Asset Trend Prediction and Backtesting with Trading Strategy

## Project Overview

This project uses deep learning to predict strong daily uptrends in the SPY ETF and evaluates the predictions through a backtested trading strategy.

The goal is to build an LSTM-based binary classification model that predicts whether SPY will experience a strong next-day upward move. A strong uptrend is defined as a next-day return greater than 0.20%.

The model predictions are converted into trading signals and compared against a buy-and-hold SPY benchmark.

## Business Problem

Financial markets generate large amounts of price, volume, technical, and macroeconomic data. This project explores whether deep learning models can identify short-term bullish patterns in SPY and improve strategy performance compared to passive investing.

The project focuses on:

- Predicting strong daily uptrends in SPY
- Engineering technical and macroeconomic financial features
- Reducing a high-dimensional feature set
- Comparing multiple LSTM-based model architectures
- Backtesting a trading strategy based on model predictions

## Dataset

The project uses historical financial market data, including:

- SPY ETF open, high, low, close, adjusted close, and volume data
- Technical indicators
- Treasury yield data
- Rolling return features
- Rolling volatility features
- Momentum and trend indicators

Potential data sources include:

- Yahoo Finance through `yfinance`
- Treasury yield tickers such as `^TNX`, `^IRX`, `^FVX`, and `^TYX`

## Target Variable

The target variable is a binary classification label.

A strong uptrend is defined as:

```python
target = 1 if next_day_return > 0.002 else 0