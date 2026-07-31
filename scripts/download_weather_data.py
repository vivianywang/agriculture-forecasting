import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    PROVINCE_STATION_CANDIDATES,
    RAW_DATA_DIR,
    WEATHER_DIR,
    YEAR_END,
    YEAR_START,
)

INVENTORY_PATH = Path(RAW_DATA_DIR) / "station_inventory.csv"

BULK_URL = (
    "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
    "?format=csv&stationID={station_id}&Year={year}&Month=1&Day=1"
    "&timeframe=2&submit=Download+Data"
)

PROVINCE_FULL_NAME = {
    "SK": "SASKATCHEWAN",
    "AB": "ALBERTA",
    "MB": "MANITOBA",
    "ON": "ONTARIO",
    "QC": "QUEBEC",
}


def load_inventory() -> pd.DataFrame:
    df = pd.read_csv(INVENTORY_PATH, skiprows=3)
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


def download_station_year(station_id: int, year: int, out_dir: Path) -> Path:
    url = BULK_URL.format(station_id=station_id, year=year)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out_path = out_dir / f"station_{station_id}_{year}.csv"
    out_path.write_bytes(resp.content)
    return out_path


def main():
    if not INVENTORY_PATH.exists():
        raise FileNotFoundError(
            f"{INVENTORY_PATH} not found. Run download_weather_stations.py first."
        )

    inventory = load_inventory()
    weather_dir = Path(WEATHER_DIR)
    weather_dir.mkdir(parents=True, exist_ok=True)

    resolved = {}
    for province, candidates in PROVINCE_STATION_CANDIDATES.items():
        station = resolve_station(inventory, province, candidates)
        if station is None:
            print(f"WARNING: no matching station found for {province}, skipping.")
            continue
        resolved[province] = station
        print(
            f"{province}: using '{station['Name']}' "
            f"(Station ID {station['Station ID']}, "
            f"daily coverage {station.get('DLY First Year')}-{station.get('DLY Last Year')})"
        )

    for province, station in resolved.items():
        station_id = int(station["Station ID"])
        for year in range(YEAR_START, YEAR_END + 1):
            try:
                path = download_station_year(station_id, year, weather_dir)
                print(f"  {province} {year}: saved {path.name}")
            except requests.RequestException as e:
                print(f"  {province} {year}: FAILED ({e})")
            time.sleep(0.5)

    print(f"\nDone. Raw daily weather CSVs are in: {weather_dir}")


if __name__ == "__main__":
    main()