from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from tensorflow.keras.models import load_model

SEQUENCE_DATA_DIR = Path("data/processed/sequences_v2")
MODELS_DIR = Path("models/v2")
REPORTS_DIR = Path("reports/v2")

MODEL_NAMES = ["lstm_1_layer", "lstm_2_layer", "lstm_5_layer", "gru", "cnn_lstm"]


def load_test_data():
    X_test = np.load(SEQUENCE_DATA_DIR / "X_test.npy")
    y_test = np.load(SEQUENCE_DATA_DIR / "y_test.npy")
    test_dates = np.load(SEQUENCE_DATA_DIR / "test_dates.npy", allow_pickle=True)
    return X_test, y_test, test_dates


def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def main():
    X_test, y_test, test_dates = load_test_data()
    all_metrics = []

    for name in MODEL_NAMES:
        model_path = MODELS_DIR / f"best_{name}.keras"
        if not model_path.exists():
            print(f"Skipping {name}: not found.")
            continue

        print(f"Evaluating {name}...")
        model = load_model(model_path)
        y_prob = model.predict(X_test, verbose=0).ravel()

        metrics = evaluate(y_test, y_prob)
        metrics["model"] = name
        all_metrics.append(metrics)

        pd.DataFrame({
            "date": test_dates,
            "actual": y_test,
            "predicted_probability": y_prob,
            "prediction": (y_prob >= 0.5).astype(int),
        }).to_csv(REPORTS_DIR / f"{name}_predictions.csv", index=False)

    summary = pd.DataFrame(all_metrics).set_index("model")
    summary.to_csv(REPORTS_DIR / "all_models_metrics.csv")

    print("\n=== V2 Model Comparison ===")
    print(summary[["accuracy", "precision", "recall", "f1_score", "roc_auc"]].to_string())


if __name__ == "__main__":
    main()