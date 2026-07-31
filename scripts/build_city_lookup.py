import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import RAW_DATA_DIR

GEONAMES_URL = "https://download.geonames.org/export/dump/CA.zip"

COLUMNS = [
    "geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude",
    "feature_class", "feature_code", "country_code", "cc2", "admin1_code",
    "admin2_code", "admin3_code", "admin4_code", "population", "elevation",
    "dem", "timezone", "modification_date",
]

ADMIN1_TO_PROVINCE = {
    "01": "AB", "02": "BC", "03": "MB", "04": "NB", "05": "NL",
    "07": "NS", "08": "ON", "09": "PE", "10": "QC", "11": "SK",
    "12": "YT", "13": "NT", "14": "NU",
}

MIN_POPULATION = 1000


def main():
    resp = requests.get(GEONAMES_URL, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open("CA.txt") as f:
            df = pd.read_csv(f, sep="\t", header=None, names=COLUMNS, low_memory=False)

    cities = df[(df["feature_class"] == "P") & (df["population"] >= MIN_POPULATION)].copy()
    cities["province"] = (
        cities["admin1_code"].astype(str).str.zfill(2).map(ADMIN1_TO_PROVINCE)
    )
    cities = cities.dropna(subset=["province"])

    out = cities[["name", "province", "latitude", "longitude", "population"]]
    out = out.sort_values("population", ascending=False).drop_duplicates(subset=["name"])

    out_path = Path(RAW_DATA_DIR) / "city_to_province.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} cities to {out_path}")


if __name__ == "__main__":
    main()