from pathlib import Path

# Configuration
REUSE_MERGED_PARQUET = True  # Set to False to rebuild the merge from the CSV files.
CHUNK_ROWS = 100_000
ROWS_2024_METADATA = b"tlc_rows_from_2024"

CODE_DIR = Path(__file__).resolve().parent
RAW_DIR = CODE_DIR / "raw_datas"
TEMP_DIR = CODE_DIR / "temp_file"
MERGED_PARQUET = TEMP_DIR / "merged.parquet"
WORKING_PARQUET = TEMP_DIR / "working.parquet"

INPUT_2024 = "hvfhv_2024_oct-dec_2p5pct.csv"
INPUT_2025 = "hvfhv_2025_2p5pct.csv"
INPUT_2026 = "hvfhv_2026_jan-sep_2p5pct.csv"
INPUT_FILES = (INPUT_2024, INPUT_2025, INPUT_2026)
