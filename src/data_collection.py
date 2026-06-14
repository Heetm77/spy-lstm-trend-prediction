from pathlib import Path

import pandas as pd
import yfinance as yf


RAW_DATA_DIR = Path("data/raw")
START_DATE = "2000-01-01"

TICKERS = {
    "SPY": "spy.csv",
    "^TNX": "treasury_10y.csv",
    "^IRX": "treasury_13w.csv",
    "^FVX": "treasury_5y.csv",
    "^TYX": "treasury_30y.csv",
}


def download_single_ticker(ticker):
    df = yf.download(
        ticker,
        period="max",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    df.index.name = "date"

    df = df.loc[df.index >= START_DATE]

    return df


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for ticker, filename in TICKERS.items():
        print(f"Downloading {ticker}...")
        df = download_single_ticker(ticker)

        output_path = RAW_DATA_DIR / filename
        df.to_csv(output_path)

        print(f"Saved {ticker} to {output_path} with {len(df)} rows")

    print("Raw market data saved to data/raw/")


if __name__ == "__main__":
    main()