from pathlib import Path

import pandas as pd

from indicators import add_all_indicators


RAW_DATA_PATH = Path("data/raw/spy.csv")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "spy_features.csv"


def load_spy_data(path=RAW_DATA_PATH):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date")
    df = df.set_index("date")

    return df


def create_target(df, threshold=0.002):
    df = df.copy()

    df["next_day_return"] = df["close"].pct_change().shift(-1)
    df["target"] = (df["next_day_return"] > threshold).astype(int)

    return df

def clean_dataset(df):
    df = df.copy()

    df = df.replace([float("inf"), float("-inf")], pd.NA)

    missing_counts = df.isna().sum()
    fully_missing_cols = missing_counts[missing_counts == len(df)].index.tolist()

    if fully_missing_cols:
        print("Dropping fully missing columns:")
        print(fully_missing_cols)
        df = df.drop(columns=fully_missing_cols)

    df = df.dropna()

    return df

def build_processed_dataset():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_spy_data()
    df = add_all_indicators(df)
    df = create_target(df)
    df = clean_dataset(df)

    df.to_csv(PROCESSED_DATA_PATH)

    print(f"Processed dataset saved to {PROCESSED_DATA_PATH}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("Target distribution:")
    print(df["target"].value_counts(normalize=True))


if __name__ == "__main__":
    build_processed_dataset()