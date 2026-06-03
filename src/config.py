from pathlib import Path

SEED = 67

N_CLIENTS = 20_000

DATA_START = "2022-01-31"
DATA_END = "2026-04-30"

PRODUCTS = ["deposit", "savings", "card"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROFILES_PATH = DATA_DIR / "profiles.parquet"
MOVEMENTS_PATH = DATA_DIR / "movements.parquet"
CONFORMED_PATH = MOVEMENTS_PATH  # временно TODO
FEATURES_PATH = DATA_DIR / "features.parquet"
SUPERVISED_PATH = DATA_DIR / "supervised.parquet"
SUPERVISED_HORIZON_PATH = DATA_DIR / "supervised_horizon.parquet"

# для TFT
TFT_DS_PATH = DATA_DIR / "tft_dataset.parquet"

TFT_MAX_ENCODER_LENGTH = 12
TFT_MAX_PREDICTION_LENGTH = 12

TFT_GROUP_IDS = ["client_id", "product"]
TFT_TIME_IDX = "time_idx"
TFT_TARGET = "balance"

TFT_STATIC_CATEGORICALS = [
    "product",
    "gender",
    "segment",
    "region",
    "is_salary",
]
TFT_TIME_VARYING_KNOWN_CATEGORICALS = ["month", "quarter", "is_december", "is_summer"]
TFT_ALL_CATEGORICALS = (
    TFT_STATIC_CATEGORICALS + TFT_TIME_VARYING_KNOWN_CATEGORICALS + ["client_id"]
)

TFT_STATIC_REALS = ["age"]
TFT_TIME_VARYING_UNKNOWN_REALS = [
    "balance",
    "tenure_months",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "roll_mean_3",
    "roll_mean_6",
    "roll_mean_12",
    "roll_std_6",
    "roll_min_12",
    "roll_max_12",
    "trend_3",
]
TFT_ALL_REALS = TFT_STATIC_REALS + TFT_TIME_VARYING_UNKNOWN_REALS
