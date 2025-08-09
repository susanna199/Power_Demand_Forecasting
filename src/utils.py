# src/utils.py

import matplotlib.pyplot as plt
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score # New imports

from config import RESULTS_PLOTS_DIR, RESULTS_METRICS_DIR

def save_plot(y_test_actual, predictions, model_name):
    """
    Saves a plot comparing actual vs. predicted values on the test set.
    """
    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
    
    plt.figure(figsize=(18, 8))
    plt.plot(y_test_actual, label='Actual Power Generation', color='blue', marker='.', alpha=0.7)
    plt.plot(predictions, label='Predicted Power Generation', color='red', linestyle='--')
    plt.title(f'{model_name.upper()} - Forecast vs Actuals (Test Set)')
    plt.xlabel('Time (Days)')
    plt.ylabel('Generation (MU)')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(RESULTS_PLOTS_DIR, f'{model_name}_validation_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Validation plot saved to: {plot_path}")

def save_metrics(y_test_actual, predictions, model_name):
    """
    Calculates and saves performance metrics (MAE, RMSE, MSE, R2) to a JSON file.
    """
    os.makedirs(RESULTS_METRICS_DIR, exist_ok=True)
    
    # Calculate all metrics
    mae = np.mean(np.abs(predictions - y_test_actual))
    mse = mean_squared_error(y_test_actual, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_actual, predictions)
    
    metrics = {
        'mean_absolute_error': mae,
        'mean_squared_error': mse, # Added MSE
        'root_mean_squared_error': rmse,
        'r_squared': r2 # Added R-squared
    }
    
    metrics_path = os.path.join(RESULTS_METRICS_DIR, f'{model_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Metrics saved to: {metrics_path}")
    print(f"  - R-squared: {r2:.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - MSE: {mse:.4f}")
    print(f"  - MAE: {mae:.4f}")

def save_forecast_plot(historical_df, forecast_df, model_name):
    """
    Saves a plot showing historical data and the future forecast.
    """
    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
    
    plt.figure(figsize=(18, 8))
    # Plot historical data (last 100 days for context)
    plt.plot(historical_df.index, historical_df['Energy Met (MU)'], label='Historical Demand', color='blue')
    # Plot forecasted data
    plt.plot(forecast_df.index, forecast_df['forecast'], label='Future Forecast', color='red', linestyle='--')
    
    plt.title(f'{model_name.upper()} - Future Power Demand Forecast')
    plt.xlabel('Date')
    plt.ylabel('Generation (MU)')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(RESULTS_PLOTS_DIR, f'{model_name}_future_forecast.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Future forecast plot saved to: {plot_path}")
