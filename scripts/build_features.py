import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    FEATURES_PATH,
    MERGED_DATASET_PATH,
    PRECIP_COLUMN,
    TEMP_COLUMN,
    TRAILING_WINDOW,
    YEAR_START,
    YIELD_COLUMN,
)


def add_lag_features(df):
    df["prior_year_yield"] = df.groupby("province")[YIELD_COLUMN].shift(1)
    df["yield_3yr_avg"] = (
        df.groupby("province")[YIELD_COLUMN]
        .apply(lambda s: s.shift(1).rolling(TRAILING_WINDOW).mean())
        .reset_index(level=0, drop=True)
    )
    return df


def add_trend_and_anomaly_features(df):
    df["year_trend"] = df["year"] - YEAR_START
    df["precip_anomaly"] = df[PRECIP_COLUMN] - df.groupby("province")[PRECIP_COLUMN].transform("mean")
    df["temp_anomaly"] = df[TEMP_COLUMN] - df.groupby("province")[TEMP_COLUMN].transform("mean")
    return df


def add_province_dummies(df):
    dummies = pd.get_dummies(df["province"], prefix="province")
    return pd.concat([df, dummies], axis=1)


def main():
    df = pd.read_csv(MERGED_DATASET_PATH)

    if YIELD_COLUMN not in df.columns:
        raise KeyError(
            f"'{YIELD_COLUMN}' not found in {MERGED_DATASET_PATH}. "
            f"Actual columns: {list(df.columns)}"
        )

    df = df.sort_values(["province", "year"]).reset_index(drop=True)

    df = add_lag_features(df)
    df = add_trend_and_anomaly_features(df)
    df = add_province_dummies(df)

    before = len(df)
    df = df.dropna(subset=["prior_year_yield", "yield_3yr_avg"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with insufficient history for lag/trailing-average features.")

    out_path = Path(FEATURES_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows x {df.shape[1]} columns to {out_path}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()