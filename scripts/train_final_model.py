import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    FEATURE_SCHEMA_PATH,
    FEATURES_PATH,
    MODEL_PATH,
    MODELS_DIR,
    TEST_YEAR_START,
    TEST_YEARS,
    TREND_PARAMS_PATH,
    YIELD_COLUMN,
)

EXCLUDE_COLUMNS = ["province", "year", YIELD_COLUMN]


def fit_province_trends(df):
    trend_params = {}
    for province, group in df.groupby("province"):
        slope, intercept = np.polyfit(group["year"], group[YIELD_COLUMN], deg=1)
        trend_params[province] = {"slope": float(slope), "intercept": float(intercept)}
    return trend_params


def apply_trend(df, trend_params):
    slopes = df["province"].map(lambda p: trend_params[p]["slope"])
    intercepts = df["province"].map(lambda p: trend_params[p]["intercept"])
    return slopes.values * df["year"].values + intercepts.values


def rmse(actual, preds):
    return np.sqrt(mean_squared_error(actual, preds))


def make_model():
    return XGBRegressor(
        max_depth=3,
        n_estimators=150,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=4,
        reg_lambda=1.0,
        random_state=42,
    )


def report_backtest(df, feature_columns):
    cutoff_year = TEST_YEAR_START if TEST_YEAR_START is not None else df["year"].max() - TEST_YEARS + 1
    train_df = df[df["year"] < cutoff_year].copy()
    test_df = df[(df["year"] >= cutoff_year) & (df["year"] < cutoff_year + TEST_YEARS)].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        print("Skipping backtest report: empty train/test split for the configured window.")
        return

    trend_params = fit_province_trends(train_df)
    train_df["trend_pred"] = apply_trend(train_df, trend_params)
    test_df["trend_pred"] = apply_trend(test_df, trend_params)
    train_df["residual"] = train_df[YIELD_COLUMN] - train_df["trend_pred"]

    X_train, y_train_residual = train_df[feature_columns], train_df["residual"]
    X_test, y_test = test_df[feature_columns], test_df[YIELD_COLUMN]

    model = make_model()
    model.fit(X_train, y_train_residual)
    final_pred = test_df["trend_pred"].values + model.predict(X_test)

    mae = mean_absolute_error(y_test, final_pred)
    test_rmse = rmse(y_test, final_pred)
    r2 = r2_score(y_test, final_pred)
    baseline_mae = {
        name: mean_absolute_error(y_test, test_df[name])
        for name in ["prior_year_yield", "yield_3yr_avg"]
    }

    print("BACKTEST (informational only, not the deployed model)")
    print(f"  Train: {len(train_df)} rows (years < {cutoff_year})")
    print(f"  Test: {len(test_df)} rows (years {cutoff_year}-{cutoff_year + TEST_YEARS - 1})")
    print(
        f"  Naive baselines: prior_year_yield MAE={baseline_mae['prior_year_yield']:.3f}"
        f"  yield_3yr_avg MAE={baseline_mae['yield_3yr_avg']:.3f}"
    )
    print(f"  Trend+XGBoost: MAE={mae:.3f}  RMSE={test_rmse:.3f}  R2={r2:.3f}")
    print()


def fit_final_model(df, feature_columns):
    trend_params = fit_province_trends(df)
    trend_pred = apply_trend(df, trend_params)
    residual = df[YIELD_COLUMN] - trend_pred

    model = make_model()
    model.fit(df[feature_columns], residual)

    return model, trend_params


def main():
    df = pd.read_csv(FEATURES_PATH)
    feature_columns = [c for c in df.columns if c not in EXCLUDE_COLUMNS]

    report_backtest(df, feature_columns)

    model, trend_params = fit_final_model(df, feature_columns)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    with open(FEATURE_SCHEMA_PATH, "w") as f:
        json.dump({"feature_columns": feature_columns, "target_column": "residual"}, f, indent=2)
    with open(TREND_PARAMS_PATH, "w") as f:
        json.dump(trend_params, f, indent=2)

    print(f"Deployed model trained on all {len(df)} rows ({df['year'].min()}-{df['year'].max()}).")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved feature schema to {FEATURE_SCHEMA_PATH}")
    print(f"Saved trend params to {TREND_PARAMS_PATH}")


if __name__ == "__main__":
    main()