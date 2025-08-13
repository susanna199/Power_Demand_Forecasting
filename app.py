# IMPORTANT: Save this script as `app.py` in the root of your `Power_Demand_Forecasting`
# project directory, NOT inside the `src` folder.

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Power Demand Analysis in Karnataka",
    page_icon="",
    layout="wide"
)

# --- Constants ---
N_PAST = 30
TARGET_COLUMN = 'Energy Met (MU)'
MODEL_PATH = os.path.join('saved_models', 'gru_model.keras')
DATA_PATH = os.path.join('data', 'processed', 'final_karnataka_power_and_temp.csv')

# --- Caching Functions ---
@st.cache_resource
def load_model():
    """Loads the pre-trained GRU model."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except IOError:
        st.error(f"Error: Model file not found at {MODEL_PATH}.")
        return None

@st.cache_data
def load_full_data():
    """Loads the full processed dataset for EDA and plotting."""
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=['date'], index_col='date')
        return df
    except FileNotFoundError:
        st.error(f"Error: Processed data file not found at {DATA_PATH}.")
        return None

@st.cache_data
def get_model_predictions(_model, df_full):
    """
    Generates predictions on the test set for plotting.
    The _model argument is used to bust the cache if the model changes.
    """
    # 1. Prepare data for the model (same logic as in data_loader.py)
    df = df_full.copy()
    df['month_sin'] = np.sin(2 * np.pi * df.index.month/12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month/12)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear/365)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear/365)
    
    final_cols = [
        TARGET_COLUMN, 'temp_avg_karnataka',
        'month_sin', 'month_cos', 'day_sin', 'day_cos'
    ]
    df_model_input = df[final_cols]

    # 2. Split and scale data
    train_size = int(len(df_model_input) * 0.8)
    df_train = df_model_input[:train_size]
    df_test = df_model_input[train_size:]

    scaler = MinMaxScaler().fit(df_train)
    data_test_scaled = scaler.transform(df_test)
    
    # 3. Create sequences
    X_test = []
    for i in range(len(data_test_scaled) - N_PAST):
        X_test.append(data_test_scaled[i : i + N_PAST])
    X_test = np.array(X_test)
    
    # 4. Predict and inverse transform
    predictions_scaled = _model.predict(X_test, verbose=0)
    
    dummy_array = np.zeros((len(predictions_scaled), df_model_input.shape[1]))
    dummy_array[:, 0] = predictions_scaled.flatten()
    predictions = scaler.inverse_transform(dummy_array)[:, 0]
    
    # 5. Create final DataFrame
    pred_dates = df_test.index[N_PAST:]
    test_predictions_df = pd.DataFrame({'date': pred_dates, 'prediction': predictions})
    
    return test_predictions_df

# --- UI Layout ---
st.title("Power Demand Analysis in Karnataka")

# --- Sidebar Navigation ---
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Exploratory Data Analysis (EDA)", "GRU Model Evaluation"])
    st.markdown("---")
    st.info("This dashboard presents the analysis and results of the power demand forecasting project.")

# Load data and model
df = load_full_data()
model = load_model()

if df is not None:
    # --- EDA Page ---
    if page == "Exploratory Data Analysis (EDA)":
        st.header("Exploratory Data Analysis")
        st.markdown("Understanding the patterns and relationships within the dataset.")

        # 1. Full Time Series Plot
        st.subheader("Daily Power Demand (2013-2023)")
        fig_ts = px.line(df, y=TARGET_COLUMN, labels={'date': 'Date', TARGET_COLUMN: 'Energy Met (MU)'})
        st.plotly_chart(fig_ts, use_container_width=True)

        # 2. Monthly Patterns
        st.subheader("Average Monthly Demand and Shortage")
        # --- FIX: Changed 'Energy Met (MU)' to 'Max.Demand Met during the day(MW)' ---
        monthly_avg = df.groupby(df.index.month_name())[['Max.Demand Met during the day(MW)', 'Shortage during maximum Demand(MW)']].mean()
        # Sort months chronologically for plotting
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        monthly_avg = monthly_avg.reindex(month_order)
        
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(
            x=monthly_avg.index,
            y=monthly_avg['Max.Demand Met during the day(MW)'],
            name='Max.Demand Met (MW)' # --- FIX: Updated label ---
        ))
        fig_monthly.add_trace(go.Bar(
            x=monthly_avg.index,
            y=monthly_avg['Shortage during maximum Demand(MW)'],
            name='Shortage (MW)'
        ))
        fig_monthly.update_layout(barmode='group', xaxis_title="Month", yaxis_title="Average Value")
        st.plotly_chart(fig_monthly, use_container_width=True)

        # 3. Correlation Heatmap
        st.subheader("Correlation Heatmap")
        corr_cols = ['Energy Met (MU)', 'Max.Demand Met during the day(MW)', 'temp_avg_karnataka', 'Shortage during maximum Demand(MW)']
        corr_matrix = df[corr_cols].corr()
        fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # --- GRU Model Evaluation Page ---
    elif page == "GRU Model Evaluation":
        st.header("GRU Model Evaluation")
        st.markdown("Assessing the performance of the trained GRU model on the unseen test data.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R-Squared (R²)", "92.9%")
        with col2:
            st.metric("Mean Absolute Error", "8.26 MU")
        with col3:
            st.metric("Root Mean Squared Error", "10.51 MU")
        
        st.markdown("---")
        
        if model is not None:
            # Generate predictions
            test_predictions_df = get_model_predictions(model, df)
            
            # Chart view selection
            st.subheader("Performance on Test Set")
            chart_view = st.radio(
                "Select Chart View",
                ("Test Set Only", "Full Time Series"),
                horizontal=True
            )

            # Split data for plotting
            train_size = int(len(df) * 0.8)
            df_train = df[:train_size]
            df_test = df[train_size:]

            fig = go.Figure()

            if chart_view == "Test Set Only":
                fig.add_trace(go.Scatter(x=df_test.index, y=df_test[TARGET_COLUMN], mode='lines', name='Actual Test Data', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=test_predictions_df['date'], y=test_predictions_df['prediction'], mode='lines', name='Predicted Test Data', line=dict(color='orange', dash='dash')))
            
            else: # Full Time Series
                fig.add_trace(go.Scatter(x=df_train.index, y=df_train[TARGET_COLUMN], mode='lines', name='Training Data', line=dict(color='royalblue')))
                fig.add_trace(go.Scatter(x=df_test.index, y=df_test[TARGET_COLUMN], mode='lines', name='Actual Test Data', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=test_predictions_df['date'], y=test_predictions_df['prediction'], mode='lines', name='Predicted Test Data', line=dict(color='orange', dash='dash')))

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Energy Met (MU)",
                legend_title="Data Type"
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Could not load the GRU model to generate predictions.")

else:
    st.error("Failed to load the dataset. Please ensure the file is located at `data/processed/final_karnataka_power_and_temp.csv`.")
