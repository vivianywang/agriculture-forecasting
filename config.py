from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

STATCAN_PRODUCT_ID = "32100359"
STATCAN_LANGUAGE = "en"

PROVINCE_STATION_CANDIDATES = {
    "SK": ["REGINA", "SASKATOON"],
    "AB": ["LETHBRIDGE", "CALGARY", "RED DEER"],
    "MB": ["WINNIPEG", "BRANDON"],
    "ON": ["LONDON", "OTTAWA"],
    "QC": ["MONTREAL", "QUEBEC"],
}

YEAR_START = 1995
YEAR_END = 2024

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
WEATHER_DIR = RAW_DATA_DIR / "weather"
STATCAN_DIR = RAW_DATA_DIR / "statcan"

PROVINCE_NAME_TO_ABBR = {
    "Saskatchewan": "SK",
    "Alberta": "AB",
    "Manitoba": "MB",
    "Ontario": "ON",
    "Quebec": "QC",
}

WHEAT_CROP_TYPE = "Wheat, all"
YIELD_UOM_KEYWORD = "tonnes per hectare"

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
WHEAT_YIELD_CLEAN_PATH = PROCESSED_DATA_DIR / "wheat_yield_clean.csv"
WEATHER_FEATURES_PATH = PROCESSED_DATA_DIR / "weather_features.csv"
MERGED_DATASET_PATH = PROCESSED_DATA_DIR / "merged_dataset.csv"

GROWING_SEASON_MONTHS = [4, 5, 6, 7, 8]
FROST_MONTH = 5
FROST_TEMP_C = 0
HEAT_TEMP_C = 30
GDD_BASE_C = 5

FEATURES_PATH = PROCESSED_DATA_DIR / "features.csv"
TRAILING_WINDOW = 3

YIELD_COLUMN = "wheat_yield_bu_ac"
TEMP_COLUMN = "mean_temp_c"
PRECIP_COLUMN = "total_precip_mm"

TEST_YEARS = 3
TEST_YEAR_START = None
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "wheat_yield_xgb.json"
FEATURE_SCHEMA_PATH = MODELS_DIR / "feature_columns.json"

TREND_PARAMS_PATH = MODELS_DIR / "trend_params.json"

CITY_LOOKUP_PATH = RAW_DATA_DIR / "city_to_province.csv"
BU_AC_TO_T_HA = 0.0673