from pathlib import Path

import pandas as pd
import yfinance as yf


RAW_DATA_DIR = Path("data/raw")


def download_market_data(tickers, start_date="2000-01-01", end_date=None):
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )

    return data


def save_single_ticker(data, ticker, output_path):
    if isinstance(data.columns, pd.MultiIndex):
        ticker_data = data[ticker].copy()
    else:
        ticker_data = data.copy()

    ticker_data.columns = [col.lower().replace(" ", "_") for col in ticker_data.columns]
    ticker_data.index.name = "date"
    ticker_data.to_csv(output_path)


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    tickers = ["SPY", "^TNX", "^IRX", "^FVX", "^TYX"]

    data = download_market_data(tickers)

    save_single_ticker(data, "SPY", RAW_DATA_DIR / "spy.csv")
    save_single_ticker(data, "^TNX", RAW_DATA_DIR / "treasury_10y.csv")
    save_single_ticker(data, "^IRX", RAW_DATA_DIR / "treasury_13w.csv")
    save_single_ticker(data, "^FVX", RAW_DATA_DIR / "treasury_5y.csv")
    save_single_ticker(data, "^TYX", RAW_DATA_DIR / "treasury_30y.csv")

    print("Raw market data saved to data/raw/")


if __name__ == "__main__":
    main()