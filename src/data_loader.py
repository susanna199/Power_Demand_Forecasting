# src/data_loader.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Import variables from your configuration file
from config import PROCESSED_DATA_FILE, TARGET_COLUMN, N_PAST, N_FUTURE

def load_and_preprocess_data():
    """
    Loads the processed data, performs final feature engineering, and selects features.
    """
    # Load the dataset created by build_dataset.py
    df = pd.read_csv(PROCESSED_DATA_FILE, parse_dates=['date'], index_col='date')
    
    # --- FIX: Explicitly select only the numeric columns needed for modeling ---
    # This prevents string columns like 'Region' or 'States' from causing errors.
    numeric_features_to_keep = ['temp_avg_karnataka']
    df = df[[TARGET_COLUMN] + numeric_features_to_keep].copy()

    # Feature Engineering for cyclical time-based features
    # These are better for NNs than simple month numbers
    df['month_sin'] = np.sin(2 * np.pi * df.index.month/12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month/12)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear/365)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear/365)
    
    # Handle any potential missing values that might have slipped through
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df

def create_sequences(data):
    """
    Creates time-series sequences suitable for LSTM/GRU models.
    'data' should be a numpy array.
    """
    X, y = [], []
    for i in range(len(data) - N_PAST - N_FUTURE + 1):
        X.append(data[i : i + N_PAST])
        # The target is the first column (index 0) of the future steps
        y.append(data[i + N_PAST : i + N_PAST + N_FUTURE, 0]) 
    return np.array(X), np.array(y)

def get_scaled_data_and_sequences():
    """
    The main function of this module. It orchestrates loading, processing, scaling,
    and sequencing the data.
    """
    df = load_and_preprocess_data()
    
    # Chronological split for time series data
    train_size = int(len(df) * 0.8)
    df_train = df[:train_size]
    df_test = df[train_size:]

    # Scale the data using MinMaxScaler
    # It's important to fit the scaler ONLY on the training data
    scaler = MinMaxScaler()
    data_train_scaled = scaler.fit_transform(df_train)
    data_test_scaled = scaler.transform(df_test)

    # Create sequences from the scaled data
    X_train, y_train = create_sequences(data_train_scaled)
    X_test, y_test = create_sequences(data_test_scaled)

    # We need a separate scaler object just for the target variable so we can
    # inverse_transform the predictions back to their original scale.
    target_scaler = MinMaxScaler()
    target_scaler.min_, target_scaler.scale_ = scaler.min_[0], scaler.scale_[0]

    return X_train, y_train, X_test, y_test, target_scaler
