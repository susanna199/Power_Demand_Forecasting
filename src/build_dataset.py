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
            temps = {k: v if v != -999 else float('nan') for k, v in data['properties']['parameter']['T2M'].items()}
            temp_df = pd.DataFrame(temps.items(), columns=['date', f'temp_{lat}_{lon}'])
            temp_df['date'] = pd.to_datetime(temp_df['date'], format='%Y%m%d')
            
            if all_temps_df.empty:
                all_temps_df = temp_df
            else:
                all_temps_df = pd.merge(all_temps_df, temp_df, on='date', how='outer')
        else:
            print(f"Warning: Failed to fetch data for {lat}, {lon}. Status: {response.status_code}")
        
        time.sleep(0.5)

    temp_cols = [col for col in all_temps_df.columns if col.startswith('temp_')]
    all_temps_df['temp_avg_karnataka'] = all_temps_df[temp_cols].mean(axis=1)
    
    return all_temps_df[['date', 'temp_avg_karnataka']]

def main():
    """
    Main function to build and save the final processed dataset.
    """
    print("--- Starting Dataset Build Process ---")

    # 1. Load raw power data
    print(f"\n[Step 1/6] Loading raw power data from {RAW_POWER_STATES_FILE}")
    try:
        # FIX: Changed parse_dates to use 'date' (lowercase) which is the correct column name.
        # The subsequent rename operation is no longer necessary.
        df_power_raw = pd.read_csv(RAW_POWER_STATES_FILE, parse_dates=['date'])
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {RAW_POWER_STATES_FILE}")
        return
    except ValueError:
        # This will catch the error if the column isn't named 'date' either.
        print(f"Error: Could not find a date column named 'date' in the raw CSV file.")
        print("Please ensure the date column in your CSV is named correctly.")
        return

    # 2. Filter for Karnataka and fetch temperature data
    df_karnataka = df_power_raw[df_power_raw['States'].str.strip().str.lower() == 'karnataka'].copy()
    df_temp = fetch_nasa_temperature_data(KARNATAKA_COORDS, START_DATE, END_DATE)

    # 3. Merge the two datasets
    df_merged = pd.merge(df_karnataka, df_temp, on='date', how='inner')

    # 4. Handle duplicate and missing dates
    print("\n[Step 2/6] Aggregating duplicate dates...")
    df_daily = df_merged.groupby('date', as_index=False).agg({
        'Region': 'first',
        'States': 'first',
        'Max.Demand Met during the day(MW)': 'mean',
        'Shortage during maximum Demand(MW)': 'mean',
        'Energy Met (MU)': 'mean',
        'temp_avg_karnataka': 'mean'
    })

    print("\n[Step 3/6] Creating a full date range to identify missing dates...")
    full_dates = pd.date_range(start=df_daily['date'].min(), end=df_daily['date'].max())
    df_full = pd.merge(pd.DataFrame({'date': full_dates}), df_daily, on='date', how='left')
    print(f"Found and prepared {df_full.isnull().any(axis=1).sum()} missing days for imputation.")

    # 5. Add time-based features and fill missing values
    print("\n[Step 4/6] Adding time-based features (Month, Season, etc.)...")
    df_full['Month'] = df_full['date'].dt.month
    seasons = {
        1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer", 5: "Summer",
        6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon", 10: "Post-Monsoon",
        11: "Post-Monsoon", 12: "Winter"
    }
    df_full['Season'] = df_full['Month'].apply(lambda x: seasons[x])
    df_full['DayName'] = df_full['date'].dt.day_name()
    df_full['MonthName'] = df_full['date'].dt.month_name()
    df_full['Year'] = df_full['date'].dt.year
    df_full['Quarter'] = df_full['date'].dt.quarter

    print("\n[Step 5/6] Filling missing values using interpolation...")
    for col in ['Region', 'States']:
        df_full[col] = df_full[col].fillna(method='ffill').fillna(method='bfill')

    num_cols = ['Max.Demand Met during the day(MW)', 'Shortage during maximum Demand(MW)', 'Energy Met (MU)', 'temp_avg_karnataka']
    df_full[num_cols] = df_full[num_cols].interpolate(method='linear')

    # 6. Save the final, cleaned dataset to the processed directory
    print("\n[Step 6/6] Saving the final dataset...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df_full.to_csv(PROCESSED_DATA_FILE, index=False)
    
    print(f"\n--- Dataset Build Complete ---")
    print(f"Successfully created and saved final dataset to: {PROCESSED_DATA_FILE}")
    print("Final dataset head:")
    print(df_full.head())

if __name__ == '__main__':
    main()
