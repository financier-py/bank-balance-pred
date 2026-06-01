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
