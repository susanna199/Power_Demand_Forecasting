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

def build_bidirectional_lstm_model(input_shape):
    """
    Builds a stacked Bidirectional LSTM model.
    'input_shape' should be (n_past, n_features).
    """
    model = Sequential([
        # Wrap the first LSTM layer with Bidirectional
        # This layer processes the sequence forwards and backwards
        Bidirectional(LSTM(units=100, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        
        # The second layer can also be Bidirectional
        Bidirectional(LSTM(units=50, return_sequences=False)),
        Dropout(0.2),
        
        # The rest of the model remains the same
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

def build_bidirectional_gru_model(input_shape):
    """
    Builds a stacked Bidirectional GRU model. (New Function)
    """
    model = Sequential([
        # Wrap the GRU layers with Bidirectional
        Bidirectional(GRU(units=100, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(GRU(units=50, return_sequences=False)),
        Dropout(0.2),
        Dense(units=25, activation='relu'),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_cnn_lstm_model(input_shape):
    """
    Builds a hybrid 1D CNN-LSTM model. (Corrected Architecture)
    The Conv1D layer acts as a feature extractor on the input sequence.
    """
    model = Sequential([
        # The Conv1D layer expects a 3D input (batch, steps, features)
        # and will correctly slide over the 30 time steps.
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),

        # The output of the CNN part is a sequence of feature maps,
        # which is then fed into the LSTM layer.
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        
        Dense(units=25, activation='relu'),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model