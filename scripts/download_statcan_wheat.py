import io
import sys
import zipfile
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import STATCAN_DIR, STATCAN_LANGUAGE, STATCAN_PRODUCT_ID

API_URL = (
    f"https://www150.statcan.gc.ca/t1/wds/rest/getFullTableDownloadCSV/"
    f"{STATCAN_PRODUCT_ID}/{STATCAN_LANGUAGE}"
)


def main():
    out_dir = Path(STATCAN_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    resp = requests.get(API_URL, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("status") != "SUCCESS":
        raise RuntimeError(f"StatCan API returned an error: {payload}")

    zip_url = payload["object"]
    zip_resp = requests.get(zip_url, timeout=120)
    zip_resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
        csv_names = [
            n for n in zf.namelist() if n.endswith(".csv") and "MetaData" not in n
        ]
        if not csv_names:
            raise RuntimeError("No data CSV found inside the downloaded zip.")
        csv_name = csv_names[0]
        zf.extract(csv_name, out_dir)

        for meta_name in [n for n in zf.namelist() if "MetaData" in n]:
            zf.extract(meta_name, out_dir)

    print(f"Saved wheat yield data to: {out_dir / csv_name}")


if __name__ == "__main__":
    main()