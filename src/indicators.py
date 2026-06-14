import pandas as pd
import numpy as np

def add_basic_price_features(df):
    df = df.copy()

    df["daily_return"] = df["close"].pct_change()
    df["log_return"] = np.log1p(df["daily_return"])
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    df["open_close_range"] = (df["close"] - df["open"]) / df["open"]

    return df


def add_moving_average_features(df):
    df = df.copy()

    windows = [5, 10, 20, 50, 100, 200]

    for window in windows:
        df[f"sma_{window}"] = df["close"].rolling(window).mean()
        df[f"ema_{window}"] = df["close"].ewm(span=window, adjust=False).mean()
        df[f"close_to_sma_{window}"] = df["close"] / df[f"sma_{window}"] - 1

    return df


def add_volatility_features(df):
    df = df.copy()

    windows = [5, 10, 21, 63]

    for window in windows:
        df[f"volatility_{window}"] = df["daily_return"].rolling(window).std()
        df[f"rolling_return_{window}"] = df["close"].pct_change(window)

    return df


def add_momentum_features(df):
    df = df.copy()

    windows = [5, 10, 21, 63]

    for window in windows:
        df[f"momentum_{window}"] = df["close"] / df["close"].shift(window) - 1

    return df


def add_rsi(df, window=14):
    df = df.copy()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    df[f"rsi_{window}"] = 100 - (100 / (1 + rs))

    return df


def add_macd(df):
    df = df.copy()

    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()

    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    return df


def add_bollinger_bands(df, window=20):
    df = df.copy()

    rolling_mean = df["close"].rolling(window).mean()
    rolling_std = df["close"].rolling(window).std()

    df["bb_middle"] = rolling_mean
    df["bb_upper"] = rolling_mean + (2 * rolling_std)
    df["bb_lower"] = rolling_mean - (2 * rolling_std)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    return df


def add_all_indicators(df):
    df = add_basic_price_features(df)
    df = add_moving_average_features(df)
    df = add_volatility_features(df)
    df = add_momentum_features(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)

    return df