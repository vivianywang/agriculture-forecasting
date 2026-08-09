import json
import sys
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import (
    BU_AC_TO_T_HA,
    CITY_LOOKUP_PATH,
    FEATURE_SCHEMA_PATH,
    MERGED_DATASET_PATH,
    MODEL_PATH,
    TREND_PARAMS_PATH,
    YIELD_COLUMN,
)

RAW_WEATHER_COLUMNS = [
    "mean_temp_c", "total_precip_mm", "frost_days", "heat_days", "growing_degree_days",
    "gs_max_temp", "gs_min_temp", "gs_total_rain", "gs_total_snow",
    "gs_heat_deg_days", "gs_cool_deg_days", "gs_days",
]


class PredictionEngine:
    def __init__(self):
        self.merged_df = pd.read_csv(MERGED_DATASET_PATH)

        self.city_lookup = pd.read_csv(CITY_LOOKUP_PATH)
        self.city_lookup["name_upper"] = self.city_lookup["name"].str.upper()

        with open(FEATURE_SCHEMA_PATH) as f:
            self.feature_columns = json.load(f)["feature_columns"]

        with open(TREND_PARAMS_PATH) as f:
            self.trend_params = json.load(f)

        self.model = XGBRegressor()
        self.model.load_model(str(MODEL_PATH))

        self.province_history_means = (
            self.merged_df.groupby("province")[["mean_temp_c", "total_precip_mm"]].mean()
        )

    def resolve_province(self, city):
        match = self.city_lookup[self.city_lookup["name_upper"] == city.strip().upper()]
        if match.empty:
            return None
        return match.iloc[0]["province"]

    def build_features(self, province, year):
        history = self.merged_df[self.merged_df["province"] == province].sort_values("year")
        latest = history.iloc[-1]
        last_3 = history.tail(3)

        row = {col: 0.0 for col in self.feature_columns}
        for col in RAW_WEATHER_COLUMNS:
            if col in row:
                row[col] = float(latest[col])

        row["prior_year_yield"] = float(latest[YIELD_COLUMN])
        row["yield_3yr_avg"] = float(last_3[YIELD_COLUMN].mean())
        row["year_trend"] = year - int(self.merged_df["year"].min())

        hist_means = self.province_history_means.loc[province]
        row["precip_anomaly"] = float(latest["total_precip_mm"] - hist_means["total_precip_mm"])
        row["temp_anomaly"] = float(latest["mean_temp_c"] - hist_means["mean_temp_c"])

        dummy_col = f"province_{province}"
        if dummy_col in row:
            row[dummy_col] = 1.0

        X = pd.DataFrame([row])[self.feature_columns]
        return X, latest

    def predict(self, city, year=None):
        province = self.resolve_province(city)
        if province is None:
            raise ValueError(f"City '{city}' not found in lookup table.")
        if province not in self.trend_params:
            raise ValueError(
                f"Province '{province}' is not covered by this model. "
                f"Supported provinces: {list(self.trend_params.keys())}"
            )

        if year is None:
            year = int(self.merged_df["year"].max()) + 1

        X, latest = self.build_features(province, year)

        trend = self.trend_params[province]
        trend_pred = trend["slope"] * year + trend["intercept"]
        residual_pred = float(self.model.predict(X)[0])
        pred_bu_ac = trend_pred + residual_pred
        pred_t_ha = pred_bu_ac * BU_AC_TO_T_HA

        precip_anomaly = float(X.iloc[0]["precip_anomaly"])
        frost_days = float(X.iloc[0]["frost_days"])
        heat_days = float(X.iloc[0]["heat_days"])

        drought_risk = "High" if precip_anomaly < -30 else ("Moderate" if precip_anomaly < -10 else "Low")
        frost_risk = "High" if frost_days > 20 else ("Moderate" if frost_days > 10 else "Low")
        heat_risk = "High" if heat_days > 100 else ("Moderate" if heat_days > 70 else "Low")

        risk_levels = {"Low": 0, "Moderate": 1, "High": 2}
        overall_score = max(risk_levels[drought_risk], risk_levels[frost_risk], risk_levels[heat_risk])
        overall_risk = {0: "Low", 1: "Moderate", 2: "High"}[overall_score]

        crop_health_score = max(0, min(100, round(100 - overall_score * 20 - abs(precip_anomaly) * 0.2)))

        return {
            "city": city,
            "province": province,
            "prediction_year": year,
            "predicted_yield_bu_ac": round(pred_bu_ac, 2),
            "predicted_yield_t_ha": round(pred_t_ha, 2),
            "drought_risk": drought_risk,
            "frost_risk": frost_risk,
            "heat_stress_risk": heat_risk,
            "overall_risk": overall_risk,
            "crop_health_score": crop_health_score,
            "weather_basis_year": int(latest["year"]),
            "weather_basis_note": (
                f"Prediction uses {int(latest['year'])} growing-season weather as a stand-in "
                "for current conditions; live weather integration is a Day 10 item."
            ),
        }