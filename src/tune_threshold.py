from pathlib import Path

import numpy as np
import pandas as pd


PREDICTIONS_PATH = Path("reports/lstm_1_layer_predictions.csv")
THRESHOLD_RESULTS_PATH = Path("reports/threshold_tuning_results.csv")


def main():
    df = pd.read_csv(PREDICTIONS_PATH)
    y_true = df["actual"].values
    y_prob = df["predicted_probability"].values

    thresholds = np.arange(0.30, 0.71, 0.01)
    results = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(y_true)

        results.append({
            "threshold": round(t, 2),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "days_in_market": int(y_pred.sum()),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(THRESHOLD_RESULTS_PATH, index=False)

    print(results_df.to_string(index=False))
    print(f"\nBest F1 threshold: {results_df.loc[results_df['f1_score'].idxmax(), 'threshold']}")
    print(f"Best recall threshold: {results_df.loc[results_df['recall'].idxmax(), 'threshold']}")


if __name__ == "__main__":
    main()