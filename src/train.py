from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

from models import build_lstm_1_layer


SEQUENCE_DATA_DIR = Path("data/processed/sequences")
MODEL_OUTPUT_PATH = Path("models/best_lstm_1_layer.keras")
TRAINING_HISTORY_PATH = Path("reports/lstm_1_layer_history.csv")


def load_sequence_data():
    X_train = np.load(SEQUENCE_DATA_DIR / "X_train.npy")
    y_train = np.load(SEQUENCE_DATA_DIR / "y_train.npy")
    X_val = np.load(SEQUENCE_DATA_DIR / "X_val.npy")
    y_val = np.load(SEQUENCE_DATA_DIR / "y_val.npy")

    return X_train, y_train, X_val, y_val


def get_class_weights(y_train):
    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    return dict(zip(classes, weights))


def train_lstm_1_layer():
    X_train, y_train, X_val, y_val = load_sequence_data()

    input_shape = (X_train.shape[1], X_train.shape[2])

    model = build_lstm_1_layer(
        input_shape=input_shape,
        units=64,
        dropout=0.2,
        learning_rate=0.001,
    )

    class_weight = get_class_weights(y_train)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            MODEL_OUTPUT_PATH,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    TRAINING_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history.history).to_csv(TRAINING_HISTORY_PATH, index=False)

    print(f"Best model saved to {MODEL_OUTPUT_PATH}")
    print(f"Training history saved to {TRAINING_HISTORY_PATH}")


if __name__ == "__main__":
    train_lstm_1_layer()