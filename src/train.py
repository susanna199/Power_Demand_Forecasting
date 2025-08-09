# src/train.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Import project modules
from config import (
    EPOCHS, 
    BATCH_SIZE, 
    VALIDATION_SPLIT,
    SAVED_MODELS_DIR
)
from data_loader import get_scaled_data_and_sequences
from models import (
    build_lstm_model, 
    build_gru_model, 
    build_cnn_lstm_model, 
    build_bidirectional_lstm_model,
    build_bidirectional_gru_model
)
from utils import save_plot, save_metrics

# --- CHOOSE YOUR MODEL HERE ---
# Options: "lstm", "gru", "cnn_lstm", "bidirectional_lstm", "bidirectional_gru"
MODEL_TO_TRAIN = "gru" 
# -----------------------------

def main():
    """
    Main function to run the model training and evaluation pipeline.
    """
    print(f"--- Starting Model Training Pipeline for: {MODEL_TO_TRAIN.upper()} ---")
    
    # 1. Load and prepare data
    print("\n[Step 1/5] Loading and preparing data...")
    X_train, y_train, X_test, y_test, target_scaler = get_scaled_data_and_sequences()
    print("Data loaded and sequenced successfully.")
    
    # 2. Build the model
    print(f"\n[Step 2/5] Building the {MODEL_TO_TRAIN.upper()} model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    if MODEL_TO_TRAIN == "lstm":
        model = build_lstm_model(input_shape)
    elif MODEL_TO_TRAIN == "gru":
        model = build_gru_model(input_shape)
    elif MODEL_TO_TRAIN == "bidirectional_lstm":
        model = build_bidirectional_lstm_model(input_shape)
    elif MODEL_TO_TRAIN == "bidirectional_gru":
        model = build_bidirectional_gru_model(input_shape)
    elif MODEL_TO_TRAIN == "cnn_lstm":
        model = build_cnn_lstm_model(input_shape)
    else:
        raise ValueError("Invalid model type specified in MODEL_TO_TRAIN.")
        
    model.summary()

    # 3. Train the model
    print(f"\n[Step 3/5] Training the {MODEL_TO_TRAIN.upper()} model...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 4. Evaluate the model on the test set
    print(f"\n[Step 4/5] Evaluating the {MODEL_TO_TRAIN.upper()} model...")
    predictions_scaled = model.predict(X_test)
    
    # Inverse transform predictions and actuals to their original scale
    predictions = target_scaler.inverse_transform(predictions_scaled)
    y_test_actual = target_scaler.inverse_transform(y_test)
    
    # Save plots and metrics
    save_plot(y_test_actual, predictions, MODEL_TO_TRAIN)
    save_metrics(y_test_actual, predictions, MODEL_TO_TRAIN)
    
    # 5. Save the trained model
    print(f"\n[Step 5/5] Saving the trained {MODEL_TO_TRAIN.upper()} model...")
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    model_save_path = os.path.join(SAVED_MODELS_DIR, f'{MODEL_TO_TRAIN}_model.keras')
    model.save(model_save_path)
    print(f"Model saved successfully to: {model_save_path}")
    
    print("\n--- Pipeline Finished ---")

if __name__ == '__main__':
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    main()
