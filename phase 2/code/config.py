from dataclasses import dataclass
from pathlib import Path

# Reuse valid outputs. Rebuilding an upstream step invalidates downstream reuse.
support_reuse = True  # Set to False to rebuild all steps from the raw CSV files.
REUSE_MERGED_PARQUET = True and support_reuse  # Set to False to rebuild the merge from the CSV files.
REUSE_STANDARDIZED_PARQUET = True and support_reuse  # Set to False to standardize the merge again.
REUSE_FEATURE_ENGINEERING = False and support_reuse   # Set to False to rebuild the features and interval table.
REUSE_PREEMBEDDING = True and support_reuse  # Reuse scaled files when the inputs and rules match.
REUSE_EMBEDDING = True and support_reuse  # Reuse the encoder and embeddings for matching inputs and settings.
CHUNK_ROWS = 100_000
ROWS_2024_METADATA = b"tlc_rows_from_2024"
EMBEDDING_ENCODER_HASH = b"tlc_embedding_encoder_sha256"

CODE_DIR = Path(__file__).resolve().parent
RAW_DIR = CODE_DIR / "raw_datas"
if not RAW_DIR.is_dir():
    RAW_DIR = CODE_DIR / "raw_data"
TEMP_DIR = CODE_DIR / "temp_file"
# Official NYC TLC taxi-zone lookup bundled for portable OD analysis.
TAXI_ZONE_LOOKUP = CODE_DIR / "requirements" / "taxi_zone_lookup.csv"
PREEMBEDDING_RULES = CODE_DIR / "requirements" / "preembedding_rules.json"
MERGED_PARQUET = TEMP_DIR / "merged.parquet"
WORKING_PARQUET = TEMP_DIR / "working.parquet"
TRAINING_PARQUET = TEMP_DIR / "training.parquet"
PREEMBEDDING_PARQUET = TEMP_DIR / "preembedding.parquet"
PREEMBEDDING_TRAINING_PARQUET = TEMP_DIR / "preembedding_training.parquet"
EMBEDDING_OUTPUT_DIR = TEMP_DIR / "embedding"
CLUSTER_ENCODER_PATH = CODE_DIR / "requirements" / "encoder.pt"
CLUSTER_OUTPUT_DIR = TEMP_DIR / "clustering"
CLUSTER_K_SELECTION_PLOT = CODE_DIR / "output" / "clustering_k_selection.png"


@dataclass(frozen=True)
class EmbeddingSettings:
    hidden_dims: tuple[int, ...] = (32, 16)
    embedding_dim: int = 6
    negative_slope: float = 0.01
    batch_size: int = 16_384
    learning_rate: float = 1e-3
    max_epochs: int = 8
    patience: int = 2
    min_delta: float = 1e-4
    time_limit_seconds: float = 240.0
    validation_rows: int = 32_768
    seed: int = 42
    device: str = "cpu"
    mixed_precision: bool = True
    cpu_threads: int = 4


EMBEDDING = EmbeddingSettings()


@dataclass(frozen=True)
class ClusteringSettings:
    k_candidates: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10, 15, 20)
    seeds: tuple[int, ...] = (42, 43, 44)
    training_rows: int = 500_000
    evaluation_rows: int = 10_000
    sample_seed: int = 45
    final_seed: int = 46
    batch_size: int = 8_192
    init: str = "k-means++"
    init_size: int | None = None
    n_init: int = 3
    max_iter: int = 100
    max_no_improvement: int | None = 10
    tol: float = 0.0
    reassignment_ratio: float = 0.01
    final_passes: int = 1
    silhouette_metric: str = "euclidean"
    silhouette_memory_mib: int = 64
    threads: int = 4
    encoder_device: str = "cpu"


CLUSTERING = ClusteringSettings()

INPUT_2024 = "hvfhv_2024_oct-dec_2p5pct.csv"
INPUT_2025 = "hvfhv_2025_2p5pct.csv"
INPUT_2026 = "hvfhv_2026_jan-sep_2p5pct.csv"
INPUT_FILES = (INPUT_2024, INPUT_2025, INPUT_2026)
