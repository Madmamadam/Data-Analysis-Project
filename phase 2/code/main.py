import merge_part
import standardize_part
import config

def main() -> None:
    """Merge the phase 2 CSV files and write a separate standardized Parquet."""
    merged_path = merge_part.merge_csvs(config.RAW_DIR, config.MERGED_PARQUET, config.REUSE_MERGED_PARQUET)
    standardize_part.standardize_parquet(merged_path, config.WORKING_PARQUET)


if __name__ == "__main__":
    main()
