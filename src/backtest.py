from pathlib import Path

import numpy as np
import pandas as pd


RAW_SPY_PATH = Path("data/raw/spy.csv")
PREDICTIONS_PATH = Path("reports/lstm_1_layer_predictions.csv")
BACKTEST_RESULTS_PATH = Path("reports/backtest_results.csv")
BACKTEST_TIMESERIES_PATH = Path("reports/backtest_timeseries.csv")


def load_backtest_data():
    spy = pd.read_csv(RAW_SPY_PATH, parse_dates=["date"])
    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])

    df = predictions.merge(
        spy[["date", "close"]],
        on="date",
        how="left",
    )

    df = df.sort_values("date").reset_index(drop=True)

    return df


def calculate_strategy_returns(df):
    df = df.copy()

    df["market_return"] = df["close"].pct_change()

    # Shift signal by one day to avoid look-ahead bias.
    df["signal"] = df["prediction"].shift(1).fillna(0)

    df["strategy_return"] = df["signal"] * df["market_return"]

    df["benchmark_cumulative"] = (1 + df["market_return"]).cumprod()
    df["strategy_cumulative"] = (1 + df["strategy_return"]).cumprod()

    return df


def calculate_max_drawdown(cumulative_returns):
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1
    return drawdown.min()


def calculate_performance_metrics(df, periods_per_year=252):
    strategy_returns = df["strategy_return"].dropna()
    benchmark_returns = df["market_return"].dropna()

    strategy_cumulative_return = df["strategy_cumulative"].iloc[-1] - 1
    benchmark_cumulative_return = df["benchmark_cumulative"].iloc[-1] - 1

    strategy_volatility = strategy_returns.std() * np.sqrt(periods_per_year)
    benchmark_volatility = benchmark_returns.std() * np.sqrt(periods_per_year)

    strategy_sharpe = (
        np.sqrt(periods_per_year) * strategy_returns.mean() / strategy_returns.std()
        if strategy_returns.std() != 0
        else 0
    )

    benchmark_sharpe = (
        np.sqrt(periods_per_year) * benchmark_returns.mean() / benchmark_returns.std()
        if benchmark_returns.std() != 0
        else 0
    )

    metrics = {
        "strategy_cumulative_return": strategy_cumulative_return,
        "benchmark_cumulative_return": benchmark_cumulative_return,
        "strategy_sharpe": strategy_sharpe,
        "benchmark_sharpe": benchmark_sharpe,
        "strategy_volatility": strategy_volatility,
        "benchmark_volatility": benchmark_volatility,
        "strategy_max_drawdown": calculate_max_drawdown(df["strategy_cumulative"]),
        "benchmark_max_drawdown": calculate_max_drawdown(df["benchmark_cumulative"]),
        "number_of_trading_days": len(df),
        "number_of_strategy_days_in_market": int(df["signal"].sum()),
    }

    return metrics


def main():
    df = load_backtest_data()
    df = calculate_strategy_returns(df)

    metrics = calculate_performance_metrics(df)

    pd.DataFrame([metrics]).to_csv(BACKTEST_RESULTS_PATH, index=False)
    df.to_csv(BACKTEST_TIMESERIES_PATH, index=False)

    print("Backtest results:")
    for key, value in metrics.items():
        if "return" in key or "drawdown" in key or "volatility" in key:
            print(f"{key}: {value:.2%}")
        elif "sharpe" in key:
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    print(f"Backtest metrics saved to {BACKTEST_RESULTS_PATH}")
    print(f"Backtest timeseries saved to {BACKTEST_TIMESERIES_PATH}")


if __name__ == "__main__":
    main()