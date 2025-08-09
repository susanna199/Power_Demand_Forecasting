# src/predict.py

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Import project modules
from config import (
    PROCESSED_DATA_FILE,
    SAVED_MODELS_DIR,
    N_PAST,
    TARGET_COLUMN
)
from data_loader import load_and_preprocess_data
from utils import save_forecast_plot

# --- CONFIGURATION ---
MODEL_TO_USE = "lstm"  # Choose the saved model you want to use for forecasting
DAYS_TO_FORECAST = 90  # How many days into the future to predict
# ---------------------

def forecast_future(model, df, scaler):
    """
    Uses a trained model to forecast future values iteratively.
    """
    print(f"Starting future forecast for {DAYS_TO_FORECAST} days...")
    
    # Get the last N_PAST days from the dataset to start the prediction
    last_sequence = df.values[-N_PAST:]
    
    # Scale this initial sequence
    last_sequence_scaled = scaler.transform(last_sequence)
    
    # Reshape for model input
    current_batch = last_sequence_scaled.reshape((1, N_PAST, df.shape[1]))
    
    future_predictions = []
    
    for i in range(DAYS_TO_FORECAST):
        # Get the prediction for the next day
        next_pred_scaled = model.predict(current_batch)[0]
        
        # Store the scaled prediction
        future_predictions.append(next_pred_scaled)
        
        # --- Create the new row for the next prediction's input ---
        # We need to generate the features for the next day
        last_known_date = df.index[-1] + pd.Timedelta(days=i)
        next_date = last_known_date + pd.Timedelta(days=1)
        
        # Use last year's temperature as a proxy
        temp_last_year = df.loc[str(next_date - pd.DateOffset(years=1))]['temp_avg_karnataka']
        
        # Create other time features
        month_sin = np.sin(2 * np.pi * next_date.month/12)
        month_cos = np.cos(2 * np.pi * next_date.month/12)
        day_sin = np.sin(2 * np.pi * next_date.dayofyear/365)
        day_cos = np.cos(2 * np.pi * next_date.dayofyear/365)
        
        # Assemble the new feature row (order must match the training data)
        # The first value is the prediction itself, the rest are the features
        new_row_features = np.array([[
            next_pred_scaled[0], 
            temp_last_year, 
            month_sin, 
            month_cos, 
            day_sin, 
            day_cos
        ]])

        # Scale the new features using the same scaler
        new_row_scaled = scaler.transform(new_row_features)

        # Append the new row to the current batch and remove the first row
        current_batch = np.append(current_batch[:, 1:, :], [new_row_scaled], axis=1)
        
    return np.array(future_predictions)

def main():
    """
    Main function to load a model and generate a future forecast.
    """
    print("--- Starting Future Forecasting ---")
    
    # 1. Load the saved model
    model_path = os.path.join(SAVED_MODELS_DIR, f'{MODEL_TO_USE}_model.keras')
    print(f"\n[Step 1/4] Loading model from {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
    except IOError:
        print(f"Error: Model file not found. Please run train.py to train the '{MODEL_TO_USE}' model first.")
        return

    # 2. Load and process the full dataset to get the scaler and last sequence
    print("\n[Step 2/4] Loading and processing full dataset...")
    df = load_and_preprocess_data()
    
    # Fit a scaler on the entire dataset to be used for forecasting
    scaler = MinMaxScaler().fit(df)
    
    # 3. Generate the forecast
    print("\n[Step 3/4] Generating future predictions...")
    predictions_scaled = forecast_future(model, df, scaler)
    
    # We need to inverse transform the predictions
    # Create a dummy array with the same shape as the scaler expects
    dummy_array = np.zeros((len(predictions_scaled), df.shape[1]))
    dummy_array[:, 0] = predictions_scaled.flatten() # Put predictions in the first column
    
    # Inverse transform
    predictions = scaler.inverse_transform(dummy_array)[:, 0]

    # 4. Create and save the forecast plot
    print("\n[Step 4/4] Saving the forecast plot...")
    # Create a date range for the forecast
    last_date = df.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=DAYS_TO_FORECAST)
    forecast_df = pd.DataFrame({'forecast': predictions}, index=forecast_dates)
    
    # Get the last 100 days of historical data for context in the plot
    historical_context = df.tail(100)
    
    save_forecast_plot(historical_context, forecast_df, MODEL_TO_USE)
    
    print("\n--- Forecasting Finished ---")

if __name__ == '__main__':
    main()
