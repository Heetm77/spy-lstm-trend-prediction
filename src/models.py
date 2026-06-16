from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def compile_model(model, learning_rate=0.001):
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_lstm_1_layer(input_shape, units=64, dropout=0.2, learning_rate=0.001):
    model = Sequential(
        [
            LSTM(units, input_shape=input_shape),
            Dropout(dropout),
            Dense(1, activation="sigmoid"),
        ]
    )

    return compile_model(model, learning_rate)


def build_lstm_2_layer(input_shape, units=64, dropout=0.2, learning_rate=0.001):
    model = Sequential(
        [
            LSTM(units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout),
            LSTM(units // 2),
            Dropout(dropout),
            Dense(1, activation="sigmoid"),
        ]
    )

    return compile_model(model, learning_rate)


def build_lstm_5_layer(input_shape, units=64, dropout=0.2, learning_rate=0.001):
    model = Sequential(
        [
            LSTM(units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout),
            LSTM(units, return_sequences=True),
            Dropout(dropout),
            LSTM(units // 2, return_sequences=True),
            Dropout(dropout),
            LSTM(units // 2, return_sequences=True),
            Dropout(dropout),
            LSTM(units // 4),
            Dropout(dropout),
            Dense(1, activation="sigmoid"),
        ]
    )

    return compile_model(model, learning_rate)


def build_gru(input_shape, units=64, dropout=0.2, learning_rate=0.001):
    model = Sequential(
        [
            GRU(units, input_shape=input_shape),
            Dropout(dropout),
            Dense(1, activation="sigmoid"),
        ]
    )

    return compile_model(model, learning_rate)


def build_cnn_lstm(input_shape, units=64, dropout=0.2, learning_rate=0.001):
    model = Sequential(
        [
            Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            LSTM(units),
            Dropout(dropout),
            Dense(1, activation="sigmoid"),
        ]
    )

    return compile_model(model, learning_rate)