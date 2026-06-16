from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


PROCESSED_DATA_PATH = Path("data/processed/spy_features.csv")
SELECTED_FEATURES_PATH = Path("models/selected_features.csv")
SEQUENCE_DATA_DIR = Path("data/processed/sequences")

LOOKBACK_DAYS = 21
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15


def load_selected_features(path=SELECTED_FEATURES_PATH):
    selected_features = pd.read_csv(path)
    return selected_features["feature"].tolist()


def load_processed_data(path=PROCESSED_DATA_PATH):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def chronological_split(df):
    train_end = int(len(df) * TRAIN_SIZE)
    val_end = int(len(df) * (TRAIN_SIZE + VAL_SIZE))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def scale_splits(train_df, val_df, test_df, feature_cols):
    scaler = MinMaxScaler()

    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_scaled, val_scaled, test_scaled


def create_sequences(df, feature_cols, lookback=LOOKBACK_DAYS):
    X, y, dates = [], [], []

    feature_values = df[feature_cols].values
    target_values = df["target"].values
    date_values = df["date"].values

    for i in range(lookback, len(df)):
        X.append(feature_values[i - lookback:i])
        y.append(target_values[i])
        dates.append(date_values[i])

    return np.array(X), np.array(y), np.array(dates)


def save_sequences(X_train, y_train, X_val, y_val, X_test, y_test, test_dates):
    SEQUENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    np.save(SEQUENCE_DATA_DIR / "X_train.npy", X_train)
    np.save(SEQUENCE_DATA_DIR / "y_train.npy", y_train)
    np.save(SEQUENCE_DATA_DIR / "X_val.npy", X_val)
    np.save(SEQUENCE_DATA_DIR / "y_val.npy", y_val)
    np.save(SEQUENCE_DATA_DIR / "X_test.npy", X_test)
    np.save(SEQUENCE_DATA_DIR / "y_test.npy", y_test)
    np.save(SEQUENCE_DATA_DIR / "test_dates.npy", test_dates)


def main():
    feature_cols = load_selected_features()
    df = load_processed_data()

    train_df, val_df, test_df = chronological_split(df)
    train_scaled, val_scaled, test_scaled = scale_splits(
        train_df,
        val_df,
        test_df,
        feature_cols,
    )

    X_train, y_train, _ = create_sequences(train_scaled, feature_cols)
    X_val, y_val, _ = create_sequences(val_scaled, feature_cols)
    X_test, y_test, test_dates = create_sequences(test_scaled, feature_cols)

    save_sequences(X_train, y_train, X_val, y_val, X_test, y_test, test_dates)

    print("Sequence files saved to data/processed/sequences/")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")


if __name__ == "__main__":
    main()