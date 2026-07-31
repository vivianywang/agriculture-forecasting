import sys
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import RAW_DATA_DIR

INVENTORY_URL = (
    "https://collaboration.cmc.ec.gc.ca/cmc/climate/"
    "Get_More_Data_Plus_de_donnees/Station%20Inventory%20EN.csv"
)
OUT_PATH = Path(RAW_DATA_DIR) / "station_inventory.csv"


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(INVENTORY_URL, timeout=60)
    resp.raise_for_status()
    OUT_PATH.write_bytes(resp.content)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()