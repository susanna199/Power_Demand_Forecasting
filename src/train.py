# src/train.py

import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from data_loader import get_original_dates

# Import project modules
from config import (
    EPOCHS,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    SAVED_MODELS_DIR,
    RESULTS_PLOTS_DIR,
    PROCESSED_DATA_FILE,
    N_PAST
)
from data_loader import get_scaled_data_and_sequences, load_and_preprocess_data
from models import (
    build_lstm_model,
    build_gru_model,
    build_cnn_lstm_model,
    build_bidirectional_lstm_model,
    build_bidirectional_gru_model
)
from utils import save_metrics

# --- CHOOSE YOUR MODEL HERE ---
MODEL_TO_TRAIN = "gru"
# ------------------------------

def save_training_history_plot(history, model_name):
    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name.upper()} - Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(RESULTS_PLOTS_DIR, f'{model_name}_training_history.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Training history plot saved to: {plot_path}")

def save_plot(y_test_actual, predictions, model_name, dates_test):
    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
    dates_test = pd.to_datetime(dates_test)
    plt.figure(figsize=(10, 6))
    plt.plot(dates_test, y_test_actual, label="Actual", color='blue')
    plt.plot(dates_test, predictions, label="Predicted", color='red', linestyle='dashed')
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.title(f"{model_name.upper()} Predictions vs Actual")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_PLOTS_DIR, f"{model_name}_predictions.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Prediction plot saved to: {plot_path}")

def main():
    print(f"--- Starting Model Training Pipeline for: {MODEL_TO_TRAIN.upper()} ---")

    # 1. Load and prepare data
    print("\n[Step 1/5] Loading and preparing data...")
    X_train, y_train, X_test, y_test, target_scaler = get_scaled_data_and_sequences()
    df_full = load_and_preprocess_data()
    print("Data loaded and sequenced successfully.")

    # Align dates for test set
    dates_full = get_original_dates(PROCESSED_DATA_FILE)
    train_size = int(len(dates_full) * 0.8)
    dates_test = dates_full[train_size:]
    dates_test = dates_test[N_PAST:N_PAST + len(y_test)]  # shift for lookback

    # Trim all arrays to match length
    min_len = min(len(dates_test), len(y_test))
    dates_test = dates_test[:min_len]
    y_test = y_test[:min_len]
    X_test = X_test[:min_len]

    # 2. Build the model
    print(f"\n[Step 2/5] Building the {MODEL_TO_TRAIN.upper()} model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    if MODEL_TO_TRAIN == "lstm":
        model = build_lstm_model(input_shape)
    elif MODEL_TO_TRAIN == "gru":
        model = build_gru_model(input_shape)
    else:
        raise ValueError("Invalid model type specified in MODEL_TO_TRAIN.")
    model.summary()

    # 3. Train the model
    print(f"\n[Step 3/5] Training the {MODEL_TO_TRAIN.upper()} model...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping],
        verbose=1
    )
    save_training_history_plot(history, MODEL_TO_TRAIN)

    # 4. Evaluate the model on the test set
    print(f"\n[Step 4/5] Evaluating the {MODEL_TO_TRAIN.upper()} model...")
    predictions_scaled = model.predict(X_test)
    predictions = target_scaler.inverse_transform(predictions_scaled)
    y_test_actual = target_scaler.inverse_transform(y_test)
    save_plot(y_test_actual, predictions, MODEL_TO_TRAIN, dates_test)
    save_metrics(y_test_actual, predictions, MODEL_TO_TRAIN)

    # 5. Save the trained model
    print(f"\n[Step 5/5] Saving the trained {MODEL_TO_TRAIN.upper()} model...")
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    model_save_path = os.path.join(SAVED_MODELS_DIR, f'{MODEL_TO_TRAIN}_model.keras')
    model.save(model_save_path)
    print(f"Model saved successfully to: {model_save_path}")

    print("\n--- Pipeline Finished ---")

if __name__ == '__main__':
    tf.random.set_seed(42)
    np.random.seed(42)
    main()
