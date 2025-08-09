# src/config.py

import os

# --- Project Root ---
# This is the correct, robust way to define the project root for .py scripts.
# It finds the directory containing the 'src' folder.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Paths ---
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Raw data files
RAW_POWER_STATES_FILE = os.path.join(RAW_DATA_DIR, 'Daily_Power_Gen_States_march_23.csv')

# Processed data file (Using a more descriptive name for clarity)
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'final_karnataka_power_and_temp.csv')

# --- NASA POWER API Configuration ---
NASA_API_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
START_DATE = "2013-01-01"
END_DATE = "2023-03-31" # Match the end of your power data
# Coordinates for Karnataka (as determined previously)
KARNATAKA_COORDS = [
    (12.0, 74.5), (12.0, 75.0), (12.0, 75.5), (12.0, 76.0), (12.0, 76.5), (12.0, 77.0), (12.0, 77.5), (12.0, 78.0), (12.0, 78.5),
    (12.5, 74.5), (12.5, 75.0), (12.5, 75.5), (12.5, 76.0), (12.5, 76.5), (12.5, 77.0), (12.5, 77.5), (12.5, 78.0), (12.5, 78.5),
    (13.0, 74.5), (13.0, 75.0), (13.0, 75.5), (13.0, 76.0), (13.0, 76.5), (13.0, 77.0), (13.0, 77.5), (13.0, 78.0), (13.0, 78.5),
    (13.5, 74.5), (13.5, 75.0), (13.5, 75.5), (13.5, 76.0), (13.5, 76.5), (13.5, 77.0), (13.5, 77.5), (13.5, 78.0), (13.5, 78.5),
    (14.0, 74.5), (14.0, 75.0), (14.0, 75.5), (14.0, 76.0), (14.0, 76.5), (14.0, 77.0), (14.0, 77.5), (14.0, 78.0), (14.0, 78.5),
    (14.5, 74.5), (14.5, 75.0), (14.5, 75.5), (14.5, 76.0), (14.5, 76.5), (14.5, 77.0), (14.5, 77.5), (14.5, 78.0), (14.5, 78.5),
    (15.0, 74.5), (15.0, 75.0), (15.0, 75.5), (15.0, 76.0), (15.0, 76.5), (15.0, 77.0), (15.0, 77.5), (15.0, 78.0), (15.0, 78.5),
    (15.5, 74.5), (15.5, 75.0), (15.5, 75.5), (15.5, 76.0), (15.5, 76.5), (15.5, 77.0), (15.5, 77.5), (15.5, 78.0), (15.5, 78.5)
]

# --- Feature Engineering ---
# IMPORTANT: Choose your target column based on your EDA.
# Let's assume you chose 'Energy Met (MU)'
TARGET_COLUMN = 'Energy Met (MU)'
# Drop redundant or useless columns
COLUMNS_TO_DROP = ['Max.Demand Met during the day(MW)', 'Shortage during maximum Demand(MW)']

# --- Model Training ---
# Sequence parameters
N_PAST = 30  # Number of past days to use for prediction
N_FUTURE = 1   # Number of future days to predict

# Model parameters
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2 # Use 20% of the training data for validation

# Paths for saving models and results
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, 'saved_models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
RESULTS_PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
RESULTS_METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
