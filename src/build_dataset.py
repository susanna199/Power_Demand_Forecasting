# src/build_dataset.py

import pandas as pd
import requests
from tqdm import tqdm
import os
import time

# Import configuration from your config file
from config import (
    RAW_POWER_STATES_FILE,
    PROCESSED_DATA_FILE,
    PROCESSED_DATA_DIR,
    NASA_API_BASE_URL,
    START_DATE,
    END_DATE,
    KARNATAKA_COORDS
)

def fetch_nasa_temperature_data(coords, start_date, end_date):
    """
    Fetches temperature data for a list of coordinates from the NASA POWER API
    and returns a DataFrame with the daily average temperature.
    """
    all_temps_df = pd.DataFrame()
    print(f"Fetching temperature data for {len(coords)} locations...")

    for lat, lon in tqdm(coords, desc="Fetching NASA Data"):
        params = {
            "parameters": "T2M",
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": start_date.replace('-', ''),
            "end": end_date.replace('-', ''),
            "format": "JSON"
        }

        response = requests.get(NASA_API_BASE_URL, params=params)

        if response.status_code == 200:
            data = response.json()
            # API returns -999 for missing values, which we replace with NaN
            temps = {k: v if v != -999 else float('nan') for k, v in data['properties']['parameter']['T2M'].items()}
            temp_df = pd.DataFrame(temps.items(), columns=['date', f'temp_{lat}_{lon}'])
            temp_df['date'] = pd.to_datetime(temp_df['date'], format='%Y%m%d')

            if all_temps_df.empty:
                all_temps_df = temp_df
            else:
                all_temps_df = pd.merge(all_temps_df, temp_df, on='date', how='outer')
        else:
            print(f"Warning: Failed to fetch data for {lat}, {lon}. Status: {response.status_code}")

        time.sleep(0.5) # Be respectful to the API server

    # Calculate the average temperature across all coordinates
    temp_cols = [col for col in all_temps_df.columns if col.startswith('temp_')]
    all_temps_df['temp_avg_karnataka'] = all_temps_df[temp_cols].mean(axis=1)

    return all_temps_df[['date', 'temp_avg_karnataka']]

def preprocess_power_data(df):
    """
    Performs all preprocessing on the raw power dataframe.
    """
    print("Preprocessing raw power data...")
    # Rename 'Date' column for consistency
    df.rename(columns={'Date': 'date'}, inplace=True)

    # Convert date column to datetime objects
    df['date'] = pd.to_datetime(df['date'])

    # Fill missing values using the mean of the respective columns
    # Note: This uses the mean of the entire dataset before filtering.
    df['Shortage during maximum Demand(MW)'] = df['Shortage during maximum Demand(MW)'].fillna(df['Shortage during maximum Demand(MW)'].mean())
    df['Energy Met (MU)'] = df['Energy Met (MU)'].fillna(df['Energy Met (MU)'].mean())

    # Create time-based features
    df['DayName'] = df['date'].dt.day_name()
    df['MonthName'] = df['date'].dt.month_name()
    df['Year'] = df['date'].dt.year
    df['Quarter'] = df['date'].dt.quarter
    df['Month'] = df['date'].dt.month

    # Create 'Season' feature based on the month
    # Winter: Dec, Jan, Feb | Spring: Mar, Apr, May | Summer: Jun, Jul, Aug | Autumn: Sep, Oct, Nov
    seasons = {
        1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn",
        11: "Autumn", 12: "Winter"
    }
    df['Season'] = df['Month'].apply(lambda x: seasons[x])

    # Filter rows for Karnataka state
    df_karnataka = df[df['States'].str.strip().str.lower() == 'karnataka'].copy()

    print("Preprocessing complete.")
    return df_karnataka

def main():
    """
    Main function to build and save the final processed dataset.
    """
    print("Starting dataset build process...")

    # 1. Load raw power data
    print(f"Loading raw power data from {RAW_POWER_STATES_FILE}")
    try:
        df_power_raw = pd.read_csv(RAW_POWER_STATES_FILE)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {RAW_POWER_STATES_FILE}")
        return

    # 2. Preprocess the raw data and filter for Karnataka
    df_karnataka = preprocess_power_data(df_power_raw)

    # 3. Fetch temperature data
    df_temp = fetch_nasa_temperature_data(KARNATAKA_COORDS, START_DATE, END_DATE)

    # 4. Merge power data with temperature data
    print("Merging power and temperature datasets...")
    df_merged = pd.merge(df_karnataka, df_temp, on='date', how='inner')

    # 5. Ensure processed directory exists and save the final dataset
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df_merged.to_csv(PROCESSED_DATA_FILE, index=False)

    print(f"\nSuccessfully created and saved final dataset to {PROCESSED_DATA_FILE}")
    print("Final dataset head:")
    print(df_merged.head())

if __name__ == '__main__':
    main()
