import argparse

import config
import embedding_step
import feature_engineering_step
import merge_part
import standardize_part


def main(*, cluster: bool = False) -> None:
    """Prepare full and training data and optionally cluster with the pretrained encoder."""
    if cluster:
        import clustring_step

        encoder = clustring_step.load_encoder(config.CLUSTER_ENCODER_PATH)

    merged_path = merge_part.merge_csvs(config.RAW_DIR, config.MERGED_PARQUET, config.REUSE_MERGED_PARQUET)
    working_path = standardize_part.standardize_parquet(
        merged_path, config.WORKING_PARQUET, reuse_existing=config.REUSE_STANDARDIZED_PARQUET
    )
    featured_path = feature_engineering_step.add_features(
        working_path, reuse_existing=config.REUSE_FEATURE_ENGINEERING, merged_path=merged_path
    )
    full, training = embedding_step.main(featured_path, reuse_existing=config.REUSE_PREEMBEDDING)
    if cluster:
        clustring_step.run_clustering(training, full, encoder=encoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--cluster", action="store_true", help="Run clustering after preparing preembedding inputs")
    main(cluster=parser.parse_args().cluster)
