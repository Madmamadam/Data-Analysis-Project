from pathlib import Path

# Reuse valid outputs. Rebuilding an upstream step invalidates downstream reuse.
REUSE_MERGED_PARQUET = True  # Set to False to rebuild the merge from the CSV files.
REUSE_STANDARDIZED_PARQUET = True  # Set to False to standardize the merge again.
REUSE_FEATURE_ENGINEERING = False   # Set to False to rebuild the features and interval table.
CHUNK_ROWS = 100_000
ROWS_2024_METADATA = b"tlc_rows_from_2024"

CODE_DIR = Path(__file__).resolve().parent
RAW_DIR = CODE_DIR / "raw_datas"
if not RAW_DIR.is_dir():
    RAW_DIR = CODE_DIR / "raw_data"
TEMP_DIR = CODE_DIR / "temp_file"
# Official NYC TLC taxi-zone lookup bundled for portable OD analysis.
TAXI_ZONE_LOOKUP = CODE_DIR / "requirements" / "taxi_zone_lookup.csv"
MERGED_PARQUET = TEMP_DIR / "merged.parquet"
WORKING_PARQUET = TEMP_DIR / "working.parquet"

INPUT_2024 = "hvfhv_2024_oct-dec_2p5pct.csv"
INPUT_2025 = "hvfhv_2025_2p5pct.csv"
INPUT_2026 = "hvfhv_2026_jan-sep_2p5pct.csv"
INPUT_FILES = (INPUT_2024, INPUT_2025, INPUT_2026)
