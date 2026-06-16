from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from models import (
    build_lstm_1_layer,
    build_lstm_2_layer,
    build_lstm_5_layer,
    build_gru,
    build_cnn_lstm,
)

SEQUENCE_DATA_DIR = Path("data/processed/sequences")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

MODEL_BUILDERS = {
    "lstm_1_layer": build_lstm_1_layer,
    "lstm_2_layer": build_lstm_2_layer,
    "lstm_5_layer": build_lstm_5_layer,
    "gru": build_gru,
    "cnn_lstm": build_cnn_lstm,
}


def load_data():
    X_train = np.load(SEQUENCE_DATA_DIR / "X_train.npy")
    y_train = np.load(SEQUENCE_DATA_DIR / "y_train.npy")
    X_val = np.load(SEQUENCE_DATA_DIR / "X_val.npy")
    y_val = np.load(SEQUENCE_DATA_DIR / "y_val.npy")
    return X_train, y_train, X_val, y_val


def get_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return dict(zip(classes, weights))


def train_model(name, builder, X_train, y_train, X_val, y_val, class_weight):
    print(f"\nTraining {name}...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = builder(input_shape=input_shape)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(
            MODELS_DIR / f"best_{name}.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(REPORTS_DIR / f"{name}_history.csv", index=False)
    print(f"Saved {name} history.")
    return hist_df


def main():
    X_train, y_train, X_val, y_val = load_data()
    class_weight = get_class_weights(y_train)

    for name, builder in MODEL_BUILDERS.items():
        train_model(name, builder, X_train, y_train, X_val, y_val, class_weight)

    print("\nAll models trained.")

if __name__ == "__main__":
    main()