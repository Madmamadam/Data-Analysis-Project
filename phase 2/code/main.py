import config
import embedding_step
import feature_engineering_step
import merge_part
import standardize_part


def main() -> None:
    """Prepare the full and training trip data for embedding."""
    merged_path = merge_part.merge_csvs(config.RAW_DIR, config.MERGED_PARQUET, config.REUSE_MERGED_PARQUET)
    working_path = standardize_part.standardize_parquet(
        merged_path, config.WORKING_PARQUET, reuse_existing=config.REUSE_STANDARDIZED_PARQUET
    )
    featured_path = feature_engineering_step.add_features(
        working_path, reuse_existing=config.REUSE_FEATURE_ENGINEERING, merged_path=merged_path
    )
    embedding_step.main(featured_path, reuse_existing=config.REUSE_PREEMBEDDING)


if __name__ == "__main__":
    main()
