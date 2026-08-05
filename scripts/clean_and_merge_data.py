import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    GEO_TO_PROVINCE,
    GDD_BASE_TEMP_C,
    GROWING_SEASON_MONTHS,
    HEAT_DAY_THRESHOLD_C,
    PROCESSED_DATA_DIR,
    PROVINCE_STATION_CANDIDATES,
    STATCAN_PRODUCT_ID,
    YEAR_END,
    YEAR_START,
    resolve_raw_data_dir,
)

WHEAT_CROP = "Wheat, all"
WHEAT_YIELD_METRIC = "Average yield (bushels per acre)"

WEATHER_NUMERIC_COLS = [
    "Max Temp (°C)",
    "Min Temp (°C)",
    "Mean Temp (°C)",
    "Heat Deg Days (°C)",
    "Cool Deg Days (°C)",
    "Total Rain (mm)",
    "Total Snow (cm)",
    "Total Precip (mm)",
]

PROVINCE_FULL_NAME = {
    "SK": "SASKATCHEWAN",
    "AB": "ALBERTA",
    "MB": "MANITOBA",
    "ON": "ONTARIO",
    "QC": "QUEBEC",
}


def load_inventory(raw_dir: Path) -> pd.DataFrame:
    inventory_path = raw_dir / "station_inventory.csv"
    if not inventory_path.exists():
        raise FileNotFoundError(
            f"{inventory_path} not found. Run download_weather_stations.py first."
        )
    df = pd.read_csv(inventory_path, skiprows=3)
    df.columns = [c.strip() for c in df.columns]
    return df


def resolve_station(inventory: pd.DataFrame, province: str, candidates: list):
    full_name = PROVINCE_FULL_NAME[province]
    prov_df = inventory[inventory["Province"].str.upper() == full_name]

    fallback = None
    for name in candidates:
        matches = prov_df[prov_df["Name"].str.upper().str.contains(name, na=False)]
        for _, row in matches.iterrows():
            dly_first, dly_last = row.get("DLY First Year"), row.get("DLY Last Year")
            if pd.isna(dly_first) or pd.isna(dly_last):
                continue
            if fallback is None:
                fallback = row
            if dly_first <= YEAR_START and dly_last >= YEAR_END:
                return row
    return fallback


def build_station_to_province(inventory: pd.DataFrame) -> dict[int, str]:
    mapping = {}
    for province, candidates in PROVINCE_STATION_CANDIDATES.items():
        station = resolve_station(inventory, province, candidates)
        if station is not None:
            mapping[int(station["Station ID"])] = province
    return mapping


def clean_wheat_yield(statcan_path: Path) -> pd.DataFrame:
    df = pd.read_csv(statcan_path, low_memory=False)
    wheat = df[
        (df["Type of crop"] == WHEAT_CROP)
        & (df["Harvest disposition"] == WHEAT_YIELD_METRIC)
        & (df["GEO"].isin(GEO_TO_PROVINCE))
    ].copy()

    wheat["year"] = pd.to_numeric(wheat["REF_DATE"], errors="coerce")
    wheat["wheat_yield_bu_ac"] = pd.to_numeric(wheat["VALUE"], errors="coerce")
    wheat["province"] = wheat["GEO"].map(GEO_TO_PROVINCE)

    cleaned = (
        wheat.loc[
            wheat["year"].between(YEAR_START, YEAR_END) & wheat["wheat_yield_bu_ac"].notna(),
            ["province", "year", "wheat_yield_bu_ac"],
        ]
        .drop_duplicates(subset=["province", "year"])
        .sort_values(["province", "year"])
        .reset_index(drop=True)
    )
    return cleaned


def _read_weather_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for col in WEATHER_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    return df


def clean_weather_features(weather_dir: Path, station_map: dict[int, str]) -> pd.DataFrame:
    pattern = re.compile(r"station_(\d+)_(\d{4})\.csv$")
    frames = []

    for path in sorted(weather_dir.glob("station_*_*.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        station_id = int(match.group(1))
        province = station_map.get(station_id)
        if province is None:
            continue

        df = _read_weather_file(path)
        growing = df[
            df["Month"].isin(GROWING_SEASON_MONTHS)
            & df["Year"].between(YEAR_START, YEAR_END)
        ].copy()
        if growing.empty:
            continue

        growing["province"] = province
        frames.append(growing)

    if not frames:
        raise RuntimeError(f"No weather files found in {weather_dir}")

    daily = pd.concat(frames, ignore_index=True)
    daily["gdd"] = (daily["Mean Temp (°C)"] - GDD_BASE_TEMP_C).clip(lower=0).fillna(0)
    daily["frost"] = (daily["Min Temp (°C)"] < 0).astype(int)
    daily["heat"] = (daily["Max Temp (°C)"] >= HEAT_DAY_THRESHOLD_C).astype(int)

    aggregated = (
        daily.groupby(["province", "Year"], as_index=False)
        .agg(
            mean_temp_c=("Mean Temp (°C)", "mean"),
            total_precip_mm=("Total Precip (mm)", "sum"),
            frost_days=("frost", "sum"),
            heat_days=("heat", "sum"),
            growing_degree_days=("gdd", "sum"),
            gs_max_temp=("Max Temp (°C)", "max"),
            gs_min_temp=("Min Temp (°C)", "min"),
            gs_total_rain=("Total Rain (mm)", "sum"),
            gs_total_snow=("Total Snow (cm)", "sum"),
            gs_heat_deg_days=("Heat Deg Days (°C)", "sum"),
            gs_cool_deg_days=("Cool Deg Days (°C)", "sum"),
            gs_days=("Day", "count"),
        )
        .rename(columns={"Year": "year"})
        .sort_values(["province", "year"])
        .reset_index(drop=True)
    )
    return aggregated


def merge_datasets(
    wheat: pd.DataFrame, weather: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = wheat.merge(weather, on=["province", "year"], how="inner")
    merged = merged.sort_values(["province", "year"]).reset_index(drop=True)

    missing_weather = wheat.merge(
        weather, on=["province", "year"], how="left", indicator=True
    )
    missing_weather = missing_weather[missing_weather["_merge"] == "left_only"]
    if not missing_weather.empty:
        rows = missing_weather[["province", "year"]].drop_duplicates()
        print(
            "WARNING: wheat rows without weather data:",
            rows.to_string(index=False),
        )

    return merged, missing_weather[["province", "year"]].drop_duplicates()


def main():
    raw_dir = resolve_raw_data_dir()
    statcan_path = raw_dir / "statcan" / f"{STATCAN_PRODUCT_ID}.csv"
    weather_dir = raw_dir / "weather"

    if not statcan_path.exists():
        raise FileNotFoundError(
            f"{statcan_path} not found. Run download_statcan_wheat.py first."
        )
    if not weather_dir.exists():
        raise FileNotFoundError(
            f"{weather_dir} not found. Run download_weather_data.py first."
        )

    inventory = load_inventory(raw_dir)
    station_map = build_station_to_province(inventory)
    if not station_map:
        raise RuntimeError("Could not map any weather stations to provinces.")

    print(f"Using raw data from: {raw_dir}")
    print(f"Station mapping: {station_map}")

    wheat = clean_wheat_yield(statcan_path)
    weather = clean_weather_features(weather_dir, station_map)
    merged, _ = merge_datasets(wheat, weather)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    wheat_path = PROCESSED_DATA_DIR / "wheat_yield.csv"
    weather_path = PROCESSED_DATA_DIR / "weather_features.csv"
    merged_path = PROCESSED_DATA_DIR / "merged_dataset.csv"

    wheat.to_csv(wheat_path, index=False)
    weather.to_csv(weather_path, index=False)
    merged.to_csv(merged_path, index=False)

    print(f"\nCleaned wheat yield: {len(wheat)} rows -> {wheat_path}")
    print(f"Cleaned weather features: {len(weather)} rows -> {weather_path}")
    print(f"Merged dataset: {len(merged)} rows -> {merged_path}")
    print("\nMerged preview:")
    print(merged.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
