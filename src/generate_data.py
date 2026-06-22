import numpy as np
import pandas as pd
from pathlib import Path


OUTPUT_PATH = Path("data/energy_demand.csv")


REGION_CONFIG = {
    "NSW1": {
        "base_demand": 8200,
        "temperature_mean": 19,
        "temperature_amp": 9,
        "volatility": 420,
    },
    "VIC1": {
        "base_demand": 6100,
        "temperature_mean": 16,
        "temperature_amp": 8,
        "volatility": 360,
    },
    "SA1": {
        "base_demand": 1600,
        "temperature_mean": 20,
        "temperature_amp": 10,
        "volatility": 180,
    },
    "QLD1": {
        "base_demand": 7100,
        "temperature_mean": 23,
        "temperature_amp": 7,
        "volatility": 380,
    },
}


def create_region_data(region: str, config: dict, timestamps: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({"timestamp": timestamps})
    df["region"] = region

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    day_of_year = df["timestamp"].dt.dayofyear

    seasonal_temperature = (
        config["temperature_mean"]
        + config["temperature_amp"] * np.sin(2 * np.pi * (day_of_year - 15) / 365)
    )

    daily_temperature = 4 * np.sin(2 * np.pi * (df["hour"] - 14) / 24)

    df["temperature_c"] = seasonal_temperature + daily_temperature + rng.normal(0, 1.8, len(df))

    morning_peak = np.exp(-((df["hour"] - 8) ** 2) / 10)
    evening_peak = np.exp(-((df["hour"] - 18) ** 2) / 12)
    daily_pattern = 1 + 0.09 * morning_peak + 0.15 * evening_peak

    weekend_effect = np.where(df["is_weekend"] == 1, 0.92, 1.0)

    summer_cooling = np.maximum(df["temperature_c"] - 24, 0) * 95
    winter_heating = np.maximum(13 - df["temperature_c"], 0) * 70

    renewable_base = 0.28 + 0.12 * np.sin(2 * np.pi * (df["hour"] - 11) / 24)
    renewable_season = 0.07 * np.sin(2 * np.pi * (day_of_year - 70) / 365)
    df["renewable_share"] = np.clip(
        renewable_base + renewable_season + rng.normal(0, 0.04, len(df)),
        0.05,
        0.75,
    )

    demand = (
        config["base_demand"] * daily_pattern * weekend_effect
        + summer_cooling
        + winter_heating
        + rng.normal(0, config["volatility"], len(df))
    )

    df["demand_mw"] = np.maximum(demand, config["base_demand"] * 0.55).round(2)

    price = (
        65
        + 0.009 * df["demand_mw"]
        - 35 * df["renewable_share"]
        + rng.normal(0, 18, len(df))
    )

    df["price_aud_mwh"] = np.maximum(price, 0).round(2)

    return df


def generate_dataset() -> pd.DataFrame:
    timestamps = pd.date_range(
        start="2024-01-01 00:00:00",
        end="2025-12-31 23:00:00",
        freq="h",
    )

    frames = []

    for index, (region, config) in enumerate(REGION_CONFIG.items(), start=1):
        frames.append(create_region_data(region, config, timestamps, seed=42 + index))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["region", "timestamp"]).reset_index(drop=True)

    return df


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset created: {OUTPUT_PATH}")
    print(f"Rows: {len(df):,}")
    print(df.head())


if __name__ == "__main__":
    main()
