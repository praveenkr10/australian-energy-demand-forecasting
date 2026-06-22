import pandas as pd

from src.generate_data import generate_dataset
from src.train_model import add_time_series_features


def test_generate_dataset_has_expected_columns():
    df = generate_dataset()

    expected_columns = {
        "timestamp",
        "region",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "temperature_c",
        "renewable_share",
        "price_aud_mwh",
        "demand_mw",
    }

    assert expected_columns.issubset(set(df.columns))
    assert len(df) > 1000
    assert df["demand_mw"].min() > 0


def test_add_time_series_features():
    df = generate_dataset()
    featured_df = add_time_series_features(df)

    expected_columns = {
        "lag_1_demand",
        "lag_24_demand",
        "lag_168_demand",
        "rolling_24_demand",
    }

    assert expected_columns.issubset(set(featured_df.columns))
    assert not featured_df[list(expected_columns)].isna().any().any()
