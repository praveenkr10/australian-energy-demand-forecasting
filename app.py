import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


DATA_PATH = Path("data/energy_demand.csv")
METRICS_PATH = Path("outputs/model_metrics.json")
PREDICTIONS_PATH = Path("outputs/test_predictions.csv")
MODEL_PATH = Path("models/best_model.joblib")


st.set_page_config(
    page_title="Australian Energy Demand Forecasting",
    page_icon="⚡",
    layout="wide",
)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    predictions = pd.read_csv(PREDICTIONS_PATH)
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])

    with open(METRICS_PATH, "r") as file:
        metrics = json.load(file)

    return data, predictions, metrics


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("⚡ Australian Energy Demand Forecasting")
st.caption("Neural network time-series forecasting project using Australian NEM-style regional demand data.")

required_files = [DATA_PATH, METRICS_PATH, PREDICTIONS_PATH, MODEL_PATH]

if not all(path.exists() for path in required_files):
    st.error(
        "Required files are missing. Run `python src/generate_data.py` and `python src/train_model.py` first."
    )
    st.stop()


data, predictions, metrics = load_data()
model_bundle = load_model()

regions = sorted(data["region"].unique())

st.sidebar.header("Dashboard Controls")
selected_region = st.sidebar.selectbox("Select region", regions)
selected_model = st.sidebar.selectbox(
    "Prediction model",
    ["Neural Network", "Random Forest", "Linear Regression"],
)

prediction_column = f"{selected_model}_prediction"

filtered_data = data[data["region"] == selected_region].copy()
filtered_predictions = predictions[predictions["region"] == selected_region].copy()

latest_demand = filtered_data.sort_values("timestamp").iloc[-1]["demand_mw"]
average_demand = filtered_data["demand_mw"].mean()
peak_demand = filtered_data["demand_mw"].max()

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.metric("Selected Region", selected_region)
metric_col2.metric("Latest Demand", f"{latest_demand:,.0f} MW")
metric_col3.metric("Average Demand", f"{average_demand:,.0f} MW")
metric_col4.metric("Peak Demand", f"{peak_demand:,.0f} MW")

st.divider()

st.subheader("Demand Trend")

trend_df = filtered_data[["timestamp", "demand_mw"]].tail(500).set_index("timestamp")
st.line_chart(trend_df)

st.subheader("Forecast vs Actual")

forecast_df = filtered_predictions[
    ["timestamp", "demand_mw", prediction_column]
].tail(500).set_index("timestamp")

st.line_chart(forecast_df)

st.subheader("Model Performance")

metrics_df = pd.DataFrame(metrics["metrics"])
st.dataframe(metrics_df, use_container_width=True)

best_model = metrics["best_model"]
st.info(f"Best model selected by RMSE: **{best_model}**")

st.subheader("Regional Demand Comparison")

region_summary = (
    data.groupby("region")["demand_mw"]
    .agg(["mean", "max", "min"])
    .reset_index()
    .rename(
        columns={
            "mean": "average_demand_mw",
            "max": "peak_demand_mw",
            "min": "minimum_demand_mw",
        }
    )
)

st.dataframe(region_summary, use_container_width=True)

st.subheader("Scenario Prediction")

st.write(
    "Use this form to estimate electricity demand for a selected region and scenario using the saved best model."
)

input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    region_input = st.selectbox("Region", regions, key="region_input")
    hour = st.slider("Hour of day", 0, 23, 18)
    day_of_week = st.slider("Day of week", 0, 6, 2)
    month = st.slider("Month", 1, 12, 1)

with input_col2:
    is_weekend = 1 if day_of_week in [5, 6] else 0
    temperature_c = st.slider("Temperature (°C)", 0.0, 45.0, 25.0)
    renewable_share = st.slider("Renewable share", 0.0, 0.9, 0.35)
    price_aud_mwh = st.slider("Price (AUD/MWh)", 0.0, 300.0, 100.0)

with input_col3:
    region_history = data[data["region"] == region_input].sort_values("timestamp")
    lag_1_default = float(region_history["demand_mw"].iloc[-1])
    lag_24_default = float(region_history["demand_mw"].iloc[-24])
    lag_168_default = float(region_history["demand_mw"].iloc[-168])
    rolling_24_default = float(region_history["demand_mw"].tail(24).mean())

    lag_1_demand = st.number_input("Lag 1 demand", value=round(lag_1_default, 2))
    lag_24_demand = st.number_input("Lag 24 demand", value=round(lag_24_default, 2))
    lag_168_demand = st.number_input("Lag 168 demand", value=round(lag_168_default, 2))
    rolling_24_demand = st.number_input("Rolling 24h demand", value=round(rolling_24_default, 2))

if st.button("Predict Demand"):
    input_df = pd.DataFrame(
        [
            {
                "region": region_input,
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "is_weekend": is_weekend,
                "temperature_c": temperature_c,
                "renewable_share": renewable_share,
                "price_aud_mwh": price_aud_mwh,
                "lag_1_demand": lag_1_demand,
                "lag_24_demand": lag_24_demand,
                "lag_168_demand": lag_168_demand,
                "rolling_24_demand": rolling_24_demand,
            }
        ]
    )

    prediction = model_bundle["model"].predict(input_df)[0]
    st.success(f"Predicted electricity demand: {prediction:,.0f} MW")

st.divider()

st.subheader("Project Notes")

st.write(
    """
    This project demonstrates time-series feature engineering, lag features, rolling demand features,
    model comparison, and neural network regression for energy demand forecasting. The included dataset
    is a reproducible Australian NEM-style synthetic dataset designed for portfolio demonstration.
    The pipeline can be adapted to official AEMO or Open Electricity data.
    """
)
