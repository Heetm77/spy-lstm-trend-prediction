from tensorflow.keras.layers import Conv1D, Dense, Dropout, GRU, Input, LSTM, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def compile_model(model, learning_rate=0.001):
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm_1_layer(input_shape, units=64, dropout=0.3, learning_rate=0.0005):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units, kernel_regularizer=l2(1e-4)),
        Dropout(dropout),
        Dense(16, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(dropout),
        Dense(1, activation="sigmoid"),
    ])
    return compile_model(model, learning_rate)


def build_lstm_2_layer(input_shape, units=64, dropout=0.3, learning_rate=0.0005):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units, return_sequences=True, kernel_regularizer=l2(1e-4)),
        Dropout(dropout),
        LSTM(units // 2, kernel_regularizer=l2(1e-4)),
        Dropout(dropout),
        Dense(16, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(dropout),
        Dense(1, activation="sigmoid"),
    ])
    return compile_model(model, learning_rate)


def build_lstm_5_layer(input_shape, units=64, dropout=0.3, learning_rate=0.0005):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units, return_sequences=True, kernel_regularizer=l2(1e-4)),
        Dropout(dropout),
        LSTM(units, return_sequences=True, kernel_regularizer=l2(1e-4)),
        Dropout(dropout),
        LSTM(units // 2, return_sequences=True, kernel_regularizer=l2(1e-4)),
        Dropout(dropout),
        LSTM(units // 2, return_sequences=True, kernel_regularizer=l2(1e-4)),
        Dropout(dropout),
        LSTM(units // 4, kernel_regularizer=l2(1e-4)),
        Dropout(dropout),
        Dense(1, activation="sigmoid"),
    ])
    return compile_model(model, learning_rate)


def build_gru(input_shape, units=64, dropout=0.3, learning_rate=0.0005):
    model = Sequential([
        Input(shape=input_shape),
        GRU(units, kernel_regularizer=l2(1e-4)),
        Dropout(dropout),
        Dense(16, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(dropout),
        Dense(1, activation="sigmoid"),
    ])
    return compile_model(model, learning_rate)


def build_cnn_lstm(input_shape, units=64, dropout=0.3, learning_rate=0.0005):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation="relu", kernel_regularizer=l2(1e-4)),
        MaxPooling1D(pool_size=2),
        Dropout(dropout),
        LSTM(units, kernel_regularizer=l2(1e-4)),
        Dropout(dropout),
        Dense(16, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(dropout),
        Dense(1, activation="sigmoid"),
    ])
    return compile_model(model, learning_rate)