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


def time_based_split(df):
    cutoff_year = TEST_YEAR_START if TEST_YEAR_START is not None else df["year"].max() - TEST_YEARS + 1
    train_df = df[df["year"] < cutoff_year].copy()
    test_df = df[(df["year"] >= cutoff_year) & (df["year"] < cutoff_year + TEST_YEARS)].copy()
    return train_df, test_df, cutoff_year


def fit_province_trends(train_df):
    trend_params = {}
    for province, group in train_df.groupby("province"):
        slope, intercept = np.polyfit(group["year"], group[YIELD_COLUMN], deg=1)
        trend_params[province] = {"slope": float(slope), "intercept": float(intercept)}
    return trend_params


def apply_trend(df, trend_params):
    slopes = df["province"].map(lambda p: trend_params[p]["slope"])
    intercepts = df["province"].map(lambda p: trend_params[p]["intercept"])
    return slopes.values * df["year"].values + intercepts.values


def rmse(actual, preds):
    return np.sqrt(mean_squared_error(actual, preds))


def naive_baseline_metrics(test_df):
    results = {}
    for name in ["prior_year_yield", "yield_3yr_avg"]:
        preds = test_df[name]
        actual = test_df[YIELD_COLUMN]
        results[name] = {"mae": mean_absolute_error(actual, preds), "rmse": rmse(actual, preds)}
    return results


def main():
    df = pd.read_csv(FEATURES_PATH)
    train_df, test_df, cutoff_year = time_based_split(df)

    print(f"Train: {len(train_df)} rows (years < {cutoff_year})")
    print(f"Test: {len(test_df)} rows (years >= {cutoff_year})")

    if len(test_df) == 0 or len(train_df) == 0:
        raise ValueError("Train or test split is empty. Check TEST_YEARS against your data's year range.")

    trend_params = fit_province_trends(train_df)
    train_df["trend_pred"] = apply_trend(train_df, trend_params)
    test_df["trend_pred"] = apply_trend(test_df, trend_params)
    train_df["residual"] = train_df[YIELD_COLUMN] - train_df["trend_pred"]

    feature_columns = [c for c in df.columns if c not in EXCLUDE_COLUMNS]

    X_train, y_train_residual = train_df[feature_columns], train_df["residual"]
    X_test = test_df[feature_columns]
    y_test = test_df[YIELD_COLUMN]

    baseline = naive_baseline_metrics(test_df)
    print("\nNaive baselines on test set:")
    for name, m in baseline.items():
        print(f"  {name}: MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}")

    model = XGBRegressor(
        max_depth=3,
        n_estimators=150,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=4,
        reg_lambda=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train_residual)

    residual_pred_test = model.predict(X_test)
    final_pred = test_df["trend_pred"].values + residual_pred_test

    mae = mean_absolute_error(y_test, final_pred)
    test_rmse = rmse(y_test, final_pred)
    r2 = r2_score(y_test, final_pred)

    print("\nTrend + XGBoost-residual model on test set:")
    print(f"  MAE={mae:.3f}  RMSE={test_rmse:.3f}  R2={r2:.3f}")

    comparison = test_df[["province", "year", YIELD_COLUMN, "prior_year_yield", "yield_3yr_avg", "trend_pred"]].copy()
    comparison = comparison.rename(columns={YIELD_COLUMN: "actual"})
    comparison["final_pred"] = final_pred
    comparison["error"] = (comparison["actual"] - comparison["final_pred"]).abs()
    print("\nPer-row test set predictions:")
    print(comparison.sort_values("error", ascending=False).to_string(index=False))

    best_baseline_mae = min(m["mae"] for m in baseline.values())
    if mae < best_baseline_mae:
        print(f"\nModel beats the best naive baseline by {best_baseline_mae - mae:.3f} MAE.")
    else:
        print(f"\nModel does NOT beat the best naive baseline (baseline MAE={best_baseline_mae:.3f}).")

    importances = sorted(zip(feature_columns, model.feature_importances_), key=lambda x: x[1], reverse=True)
    print("\nFeature importances (predicting residual from trend):")
    for name, score in importances:
        print(f"  {name}: {score:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    with open(FEATURE_SCHEMA_PATH, "w") as f:
        json.dump({"feature_columns": feature_columns, "target_column": "residual"}, f, indent=2)
    with open(TREND_PARAMS_PATH, "w") as f:
        json.dump(trend_params, f, indent=2)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved feature schema to {FEATURE_SCHEMA_PATH}")
    print(f"Saved trend params to {TREND_PARAMS_PATH}")


if __name__ == "__main__":
    main()