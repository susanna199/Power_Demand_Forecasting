# src/models.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, TimeDistributed

def build_lstm_model(input_shape):
    """
    Builds a stacked LSTM model.
    'input_shape' should be (n_past, n_features).
    """
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25, activation='relu'),
        Dense(units=1) # Output layer: predicting 1 value
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_gru_model(input_shape):
    """
    Builds a stacked GRU model.
    'input_shape' should be (n_past, n_features).
    """
    model = Sequential([
        GRU(units=100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25, activation='relu'),
        Dense(units=1) # Output layer
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

