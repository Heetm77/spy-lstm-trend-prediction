from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

FEATURES_PATH = Path("data/processed/spy_features_v2.csv")
SEQUENCE_OUTPUT_DIR = Path("data/processed/sequences_v2")
SEQUENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK = 21
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

EXCLUDE_COLS = [
    "target", "open", "high", "low", "close", "volume",
    "sma_5", "sma_10", "sma_21", "sma_50", "sma_200",
    "bb_upper", "bb_lower", "volume_sma_21",
    "xlk_close", "xlf_close", "xle_close", "xlv_close", "xli_close",
    "vix_sma_10", "macd", "macd_signal",
]


def make_sequences(X, y, dates, lookback):
    Xs, ys, ds = [], [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
        ds.append(dates[i])
    return np.array(Xs), np.array(ys), np.array(ds)


def main():
    df = pd.read_csv(FEATURES_PATH, index_col="date", parse_dates=True)
    df = df.sort_index()

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    X = df[feature_cols].values
    y = df["target"].values
    dates = df.index.to_numpy()

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    scaler = RobustScaler()
    X_train_raw = scaler.fit_transform(X[:train_end])
    X_val_raw = scaler.transform(X[train_end:val_end])
    X_test_raw = scaler.transform(X[val_end:])

    X_train, y_train, train_dates = make_sequences(X_train_raw, y[:train_end], dates[:train_end], LOOKBACK)
    X_val, y_val, val_dates = make_sequences(X_val_raw, y[train_end:val_end], dates[train_end:val_end], LOOKBACK)
    X_test, y_test, test_dates = make_sequences(X_test_raw, y[val_end:], dates[val_end:], LOOKBACK)

    np.save(SEQUENCE_OUTPUT_DIR / "X_train.npy", X_train)
    np.save(SEQUENCE_OUTPUT_DIR / "y_train.npy", y_train)
    np.save(SEQUENCE_OUTPUT_DIR / "X_val.npy", X_val)
    np.save(SEQUENCE_OUTPUT_DIR / "y_val.npy", y_val)
    np.save(SEQUENCE_OUTPUT_DIR / "X_test.npy", X_test)
    np.save(SEQUENCE_OUTPUT_DIR / "y_test.npy", y_test)
    np.save(SEQUENCE_OUTPUT_DIR / "test_dates.npy", test_dates)
    np.save(SEQUENCE_OUTPUT_DIR / "train_dates.npy", train_dates)
    np.save(SEQUENCE_OUTPUT_DIR / "val_dates.npy", val_dates)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape:   {X_val.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"Sequences saved to {SEQUENCE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()