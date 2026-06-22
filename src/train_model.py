import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/energy_demand.csv")
MODEL_PATH = Path("models/best_model.joblib")
METRICS_PATH = Path("outputs/model_metrics.json")
PREDICTIONS_PATH = Path("outputs/test_predictions.csv")


FEATURE_COLUMNS = [
    "region",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "temperature_c",
    "renewable_share",
    "price_aud_mwh",
    "lag_1_demand",
    "lag_24_demand",
    "lag_168_demand",
    "rolling_24_demand",
]


TARGET_COLUMN = "demand_mw"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run: python src/generate_data.py"
        )

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["region", "timestamp"]).copy()

    for lag in [1, 24, 168]:
        df[f"lag_{lag}_demand"] = df.groupby("region")["demand_mw"].shift(lag)

    df["rolling_24_demand"] = (
        df.groupby("region")["demand_mw"]
        .shift(1)
        .rolling(window=24)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df = df.dropna().reset_index(drop=True)

    return df


def build_pipeline(model):
    categorical_features = ["region"]

    numeric_features = [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "temperature_c",
        "renewable_share",
        "price_aud_mwh",
        "lag_1_demand",
        "lag_24_demand",
        "lag_168_demand",
        "rolling_24_demand",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", StandardScaler(), numeric_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def evaluate_model(name: str, model, X_train, X_test, y_train, y_test):
    pipeline = build_pipeline(model)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    r2 = r2_score(y_test, predictions)

    metrics = {
        "model": name,
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "mape": round(float(mape), 2),
        "r2": round(float(r2), 4),
    }

    return pipeline, predictions, metrics


def main():
    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    df = load_data()
    df = add_time_series_features(df)

    split_timestamp = df["timestamp"].quantile(0.8)

    train_df = df[df["timestamp"] <= split_timestamp].copy()
    test_df = df[df["timestamp"] > split_timestamp].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=80,
            max_depth=14,
            random_state=42,
            n_jobs=-1,
        ),
        "Neural Network": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=120,
            early_stopping=True,
            random_state=42,
        ),
    }

    all_metrics = []
    prediction_frame = test_df[["timestamp", "region", "demand_mw"]].copy()

    best_pipeline = None
    best_metric = float("inf")
    best_model_name = None

    for name, model in models.items():
        print(f"Training {name}...")
        pipeline, predictions, metrics = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        all_metrics.append(metrics)
        prediction_frame[f"{name}_prediction"] = predictions.round(2)

        if metrics["rmse"] < best_metric:
            best_metric = metrics["rmse"]
            best_pipeline = pipeline
            best_model_name = name

    joblib.dump(
        {
            "model": best_pipeline,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "best_model_name": best_model_name,
        },
        MODEL_PATH,
    )

    result = {
        "project": "Australian Energy Demand Forecasting",
        "dataset": "Synthetic Australian NEM-style regional electricity demand dataset",
        "target": TARGET_COLUMN,
        "best_model": best_model_name,
        "metrics": all_metrics,
        "notes": [
            "Chronological split used to avoid random time-series leakage.",
            "Neural Network model uses MLPRegressor with lag and rolling demand features.",
            "Dataset can be replaced with AEMO/Open Electricity data for production use.",
        ],
    }

    with open(METRICS_PATH, "w") as file:
        json.dump(result, file, indent=2)

    prediction_frame.to_csv(PREDICTIONS_PATH, index=False)

    print("\nTraining complete.")
    print(f"Best model: {best_model_name}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
