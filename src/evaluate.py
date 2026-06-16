from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model


SEQUENCE_DATA_DIR = Path("data/processed/sequences")
MODEL_PATH = Path("models/best_lstm_1_layer.keras")
METRICS_PATH = Path("reports/lstm_1_layer_metrics.csv")
PREDICTIONS_PATH = Path("reports/lstm_1_layer_predictions.csv")


def load_test_data():
    X_test = np.load(SEQUENCE_DATA_DIR / "X_test.npy")
    y_test = np.load(SEQUENCE_DATA_DIR / "y_test.npy")
    test_dates = np.load(SEQUENCE_DATA_DIR / "test_dates.npy", allow_pickle=True)

    return X_test, y_test, test_dates


def evaluate_predictions(y_true, y_probability, threshold=0.5):
    y_pred = (y_probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probability),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }

    return metrics, y_pred


def main():
    X_test, y_test, test_dates = load_test_data()

    model = load_model(MODEL_PATH)

    y_probability = model.predict(X_test).ravel()
    metrics, y_pred = evaluate_predictions(y_test, y_probability)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(METRICS_PATH, index=False)

    predictions_df = pd.DataFrame(
        {
            "date": test_dates,
            "actual": y_test,
            "predicted_probability": y_probability,
            "prediction": y_pred,
        }
    )
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    print("Evaluation metrics:")
    print(metrics_df.T)
    print(f"Metrics saved to {METRICS_PATH}")
    print(f"Predictions saved to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()