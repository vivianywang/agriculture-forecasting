import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import RAW_DATA_DIR

GEONAMES_CITIES_URL = "https://download.geonames.org/export/dump/CA.zip"
GEONAMES_ADMIN1_URL = "https://download.geonames.org/export/dump/admin1CodesASCII.txt"

CITY_COLUMNS = [
    "geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude",
    "feature_class", "feature_code", "country_code", "cc2", "admin1_code",
    "admin2_code", "admin3_code", "admin4_code", "population", "elevation",
    "dem", "timezone", "modification_date",
]

PROVINCE_NAME_TO_ABBR = {
    "Alberta": "AB", "British Columbia": "BC", "Manitoba": "MB",
    "New Brunswick": "NB", "Newfoundland and Labrador": "NL",
    "Nova Scotia": "NS", "Ontario": "ON", "Prince Edward Island": "PE",
    "Quebec": "QC", "Saskatchewan": "SK", "Yukon": "YT",
    "Northwest Territories": "NT", "Nunavut": "NU",
}

MIN_POPULATION = 1000


def build_admin1_code_map():
    resp = requests.get(GEONAMES_ADMIN1_URL, timeout=60)
    resp.raise_for_status()
    admin1_df = pd.read_csv(
        io.StringIO(resp.text), sep="\t", header=None,
        names=["code", "name", "asciiname", "geonameid"],
    )
    ca_rows = admin1_df[admin1_df["code"].str.startswith("CA.")]

    code_to_abbr = {}
    for _, row in ca_rows.iterrows():
        code_suffix = row["code"].split(".")[1]
        abbr = PROVINCE_NAME_TO_ABBR.get(row["asciiname"])
        if abbr:
            code_to_abbr[code_suffix] = abbr

    missing = set(PROVINCE_NAME_TO_ABBR.values()) - set(code_to_abbr.values())
    if missing:
        print(f"WARNING: could not resolve GeoNames codes for: {missing}")

    return code_to_abbr


def main():
    code_to_abbr = build_admin1_code_map()

    resp = requests.get(GEONAMES_CITIES_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open("CA.txt") as f:
            df = pd.read_csv(
                f, sep="\t", header=None, names=CITY_COLUMNS, low_memory=False,
                dtype={"admin1_code": str},
            )

    cities = df[(df["feature_class"] == "P") & (df["population"] >= MIN_POPULATION)].copy()
    cities["province"] = cities["admin1_code"].astype(str).str.zfill(2).map(code_to_abbr)
    cities = cities.dropna(subset=["province"])

    out = cities[["name", "province", "latitude", "longitude", "population"]]
    out = out.sort_values("population", ascending=False).drop_duplicates(subset=["name"])

    out_path = Path(RAW_DATA_DIR) / "city_to_province.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} cities to {out_path}")
    print(f"Province breakdown:\n{out['province'].value_counts()}")


if __name__ == "__main__":
    main()