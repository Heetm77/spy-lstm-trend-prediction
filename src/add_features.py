from pathlib import Path

import pandas as pd
import yfinance as yf

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_data():
    print("Downloading SPY...")
    spy = yf.download("SPY", start="2010-01-01", end="2024-12-31", auto_adjust=True)
    spy.columns = spy.columns.get_level_values(0)
    spy.index.name = "date"
    spy.columns = [c.lower() for c in spy.columns]
    spy.to_csv(RAW_DIR / "spy.csv")

    print("Downloading VIX...")
    vix = yf.download("^VIX", start="2010-01-01", end="2024-12-31", auto_adjust=True)
    vix.columns = vix.columns.get_level_values(0)
    vix.index.name = "date"
    vix.columns = [c.lower() for c in vix.columns]
    vix[["close"]].rename(columns={"close": "vix_close"}).to_csv(RAW_DIR / "vix.csv")

    print("Downloading sector ETFs...")
    sectors = ["XLK", "XLF", "XLE", "XLV", "XLI"]
    for ticker in sectors:
        df = yf.download(ticker, start="2010-01-01", end="2024-12-31", auto_adjust=True)
        df.columns = df.columns.get_level_values(0)
        df.index.name = "date"
        df.columns = [c.lower() for c in df.columns]
        df[["close"]].rename(columns={"close": f"{ticker.lower()}_close"}).to_csv(
            RAW_DIR / f"{ticker.lower()}.csv"
        )
        print(f"  Downloaded {ticker}")

    return spy


def build_features(spy):
    df = spy.copy()
    df.index = pd.to_datetime(df.index)

    # Core returns
    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_21d"] = df["close"].pct_change(21)

    # Moving averages
    for w in [5, 10, 21, 50, 200]:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
        df[f"close_vs_sma_{w}"] = df["close"] / df[f"sma_{w}"] - 1

    # Volatility
    df["volatility_21d"] = df["return_1d"].rolling(21).std()
    df["volatility_5d"] = df["return_1d"].rolling(5).std()

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - (100 / (1 + gain / loss))

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # Volume features
    df["volume_sma_21"] = df["volume"].rolling(21).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_21"]
    df["volume_return_corr"] = df["return_1d"].rolling(21).corr(df["volume_ratio"])

    # VIX
    vix = pd.read_csv(RAW_DIR / "vix.csv", index_col="date", parse_dates=True)
    df = df.join(vix, how="left")
    df["vix_close"] = df["vix_close"].ffill()
    df["vix_change"] = df["vix_close"].pct_change()
    df["vix_sma_10"] = df["vix_close"].rolling(10).mean()
    df["vix_vs_sma"] = df["vix_close"] / df["vix_sma_10"] - 1

    # Sector ETFs
    sectors = ["xlk", "xlf", "xle", "xlv", "xli"]
    for ticker in sectors:
        sec = pd.read_csv(RAW_DIR / f"{ticker}.csv", index_col="date", parse_dates=True)
        df = df.join(sec, how="left")
        col = f"{ticker}_close"
        df[col] = df[col].ffill()
        df[f"{ticker}_return_5d"] = df[col].pct_change(5)
        df[f"{ticker}_vs_spy"] = df[col].pct_change(5) - df["return_5d"]

    # Lagged returns
    for lag in [1, 2, 3, 5]:
        df[f"return_lag_{lag}"] = df["return_1d"].shift(lag)

    # Label: next day up
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    return df


def main():
    spy = download_data()
    df = build_features(spy)
    df = df.dropna()

    out_path = PROCESSED_DIR / "spy_features_v2.csv"
    df.to_csv(out_path)
    print(f"\nSaved {len(df)} rows, {len(df.columns)} features to {out_path}")
    print(f"Feature columns: {[c for c in df.columns if c != 'target']}")


if __name__ == "__main__":
    main()