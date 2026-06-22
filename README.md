# Australian Energy Demand Forecasting

A neural network time-series forecasting project that predicts electricity demand across Australian NEM-style regions using lag features, rolling averages, weather-style variables, renewable share, price signals and regional patterns.

## Project Overview

This project demonstrates how machine learning and neural networks can be used to forecast electricity demand. It uses a reproducible Australian NEM-style synthetic dataset with regional demand patterns for NSW, VIC, SA and QLD.

The pipeline can later be adapted to official Australian energy market datasets from AEMO or Open Electricity.

## Features

- Synthetic Australian NEM-style electricity demand dataset
- Regional demand patterns for NSW1, VIC1, SA1 and QLD1
- Time-series feature engineering
- Lag features for previous hour, previous day and previous week demand
- Rolling 24-hour demand feature
- Model comparison using Linear Regression, Random Forest and Neural Network
- Evaluation metrics including MAE, RMSE, MAPE and R2
- Streamlit dashboard
- Scenario-based demand prediction
- Automated tests
- GitHub-ready documentation

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- MLPRegressor neural network
- Streamlit
- joblib
- pytest

## How the Project Works

1. Generate synthetic Australian NEM-style electricity demand data.
2. Engineer time-based, lag and rolling features.
3. Train multiple forecasting models.
4. Compare models using MAE, RMSE, MAPE and R2.
5. Save the best model.
6. Display demand trends, forecasts and model metrics in a Streamlit dashboard.
7. Allow users to run scenario-based demand predictions.

## Run Locally

Create virtual environment and install dependencies:

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Generate dataset:

python src/generate_data.py

Train models:

python src/train_model.py

Run dashboard:

streamlit run app.py

Run tests:

python -m pytest

## Dataset Note

The included dataset is synthetic and generated for reproducible portfolio demonstration. It is designed to reflect common electricity demand drivers such as region, time of day, weekday/weekend behaviour, temperature, renewable share, price and lagged demand.

For a production version, this project can be adapted to official AEMO NEM data or Open Electricity API data.

## Portfolio Value

This project demonstrates neural network regression, time-series forecasting, feature engineering, model benchmarking, energy analytics, dashboarding, testing and clean documentation.

## Disclaimer

This project is for educational and portfolio purposes only. It is not intended for operational energy market forecasting or trading decisions.
