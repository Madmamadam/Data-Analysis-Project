import feature_engineering_step
import merge_part
import standardize_part
import config

def main() -> None:
    """Merge, standardize, and add features to the phase 2 trips."""
    merged_path = merge_part.merge_csvs(config.RAW_DIR, config.MERGED_PARQUET, config.REUSE_MERGED_PARQUET)
    working_path = standardize_part.standardize_parquet(
        merged_path, config.WORKING_PARQUET, reuse_existing=config.REUSE_STANDARDIZED_PARQUET
    )
    feature_engineering_step.add_features(
        working_path, reuse_existing=config.REUSE_FEATURE_ENGINEERING, merged_path=merged_path
    )


if __name__ == "__main__":
    main()
