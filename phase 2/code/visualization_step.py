"""Plot encoder representations of training trips colored by their saved cluster labels."""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

import config
import io_part
from clustring_step import Encoder, _embedding_batches, load_encoder
from console import print_status
from feature_engineering_step import VALIDATION_START_METADATA, _before_validation, _paired_batches
from standardize_part import PICKUP_TIMESTAMP, raw_timestamps

DEFAULT_TRAINING = config.EMBEDDING_OUTPUT_DIR / "embeddings_training.parquet"
DEFAULT_OUTPUT = config.CODE_DIR / "output" / "training_clusters_pca.png"
DEFAULT_MAX_POINTS = 100_000


def _training_labels(
    assignments: pq.ParquetFile, merged: pq.ParquetFile, training_rows: int, cutoff: date,
) -> np.ndarray:
    if assignments.metadata.num_rows != merged.metadata.num_rows or PICKUP_TIMESTAMP not in merged.schema_arrow.names:
        raise ValueError("Use the raw merge that matches the full clustering rows and contains pickup timestamps.")
    for name in ("row_index", "cluster"):
        if name not in assignments.schema_arrow.names or not pa.types.is_integer(assignments.schema_arrow.field(name).type):
            raise ValueError(f"Cluster assignments must have an integer {name!r} column.")

    labels = np.empty(training_rows, dtype=np.int64)
    row_start = training_start = 0
    for batch, raw in _paired_batches(assignments, merged, ["row_index", "cluster"], [PICKUP_TIMESTAMP]):
        row_end = row_start + batch.num_rows
        indices, clusters = batch.column("row_index"), batch.column("cluster")
        if indices.null_count or not np.array_equal(indices.to_numpy(), np.arange(row_start, row_end)):
            raise ValueError("Cluster assignments must follow the full dataset's zero-based row order.")
        if clusters.null_count or np.any(clusters.to_numpy() < 0):
            raise ValueError("Cluster labels must be nonnegative integers without nulls.")
        mask = _before_validation(raw_timestamps(raw, PICKUP_TIMESTAMP), cutoff.year * 12 + cutoff.month)
        selected = clusters.filter(mask).to_numpy()
        training_end = training_start + len(selected)
        if training_end > training_rows:
            raise ValueError("The raw pickup cutoff selects more rows than the training Parquet contains.")
        labels[training_start:training_end] = selected
        row_start, training_start = row_end, training_end
    if training_start != training_rows:
        raise ValueError("The raw pickup cutoff does not select the number of rows in the training Parquet.")
    return labels


def _training_vectors(training: pq.ParquetFile, encoder: Encoder | None) -> np.ndarray:
    row_count = training.metadata.num_rows
    if row_count < 2:
        raise ValueError("Two-dimensional PCA requires at least two training rows.")
    batches = _embedding_batches(training, config.CHUNK_ROWS, encoder)
    first = next(batches)
    if first.shape[1] < 2:
        raise ValueError("Two-dimensional PCA requires at least two embedding dimensions.")
    vectors = np.empty((row_count, first.shape[1]), dtype=np.float32)
    row_start = len(first)
    vectors[:row_start] = first
    for batch in batches:
        row_end = row_start + len(batch)
        vectors[row_start:row_end] = batch
        row_start = row_end
    return vectors


def _save_plot(
    output_path: Path, points: np.ndarray, labels: np.ndarray, variance: np.ndarray, training_rows: int,
) -> None:
    figure = Figure(figsize=(10, 6), layout="constrained")
    axes = figure.subplots()
    groups, color_indices = np.unique(labels, return_inverse=True)
    palette = colormaps.get_cmap("tab10" if len(groups) <= 10 else "tab20")
    if len(groups) > palette.N:
        palette = colormaps.get_cmap("turbo").resampled(len(groups))
    colors = palette(np.arange(len(groups)))
    axes.scatter(points[:, 0], points[:, 1], c=colors[color_indices], s=5, alpha=0.34, linewidths=0)
    axes.set(
        title="Training trip clusters",
        xlabel=f"PC1 ({variance[0]:.1%} of variance)",
        ylabel=f"PC2 ({variance[1]:.1%} of variance)",
    )
    axes.set_axisbelow(True)
    axes.grid(alpha=0.2)
    handles = [
        Line2D([], [], color=color, marker="o", linestyle="none", label=str(group))
        for group, color in zip(groups, colors, strict=True)
    ]
    axes.legend(handles=handles, title="Cluster", loc="upper left", bbox_to_anchor=(1.01, 1))
    figure.supxlabel(f"{len(points):,} of {training_rows:,} training trips shown", fontsize=10, color="#404850")
    with io_part.atomic_output_path(output_path) as temporary_path:
        figure.savefig(temporary_path, format="png", dpi=180)


def _matches_signature(actual: bytes, recorded: bytes | None) -> bool:
    # Device and inode can change when moving saved pipeline files between machines.
    return recorded is not None and actual.split(b":")[-2:] == recorded.split(b":")[-2:]


def plot_training_clusters(
    assignments_path: Path,
    training_path: Path = DEFAULT_TRAINING,
    merged_path: Path = config.MERGED_PARQUET,
    *,
    encoder_path: Path | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
    output_path: Path = DEFAULT_OUTPUT,
) -> Path:
    """Fit PCA on all training embeddings and save a scatter plot of their cluster labels."""
    if max_points < 0:
        raise ValueError("The point limit must be nonnegative. Use zero to plot every row.")
    inputs = [assignments_path, training_path, merged_path]
    if encoder_path is not None:
        inputs.append(encoder_path)
    if output_path.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("Choose an output path distinct from the visualization inputs.")
    signatures = {path: io_part.file_signature(path) for path in inputs}

    with (
        pq.ParquetFile(assignments_path) as assignments,
        pq.ParquetFile(training_path) as training,
        pq.ParquetFile(merged_path) as merged,
        threadpool_limits(limits=config.CLUSTERING.threads),
    ):
        metadata = (assignments.metadata.metadata or {}).get(b"tlc_clustering")
        if metadata is None:
            raise ValueError("Use a cluster_assignments.parquet from the clustering step.")
        run = json.loads(metadata)
        uses_encoder = encoder_path is not None
        if run.get("input_kind") != ("preembedding" if uses_encoder else "embeddings"):
            raise ValueError("Use saved embeddings for an embeddings run, or pass its encoder for a preembedding run.")
        recorded_training = run.get("training_signature", "").encode("ascii")
        if not _matches_signature(signatures[training_path], recorded_training):
            raise ValueError("Use the same training Parquet that was used for the clustering run.")

        training_metadata = training.metadata.metadata or {}
        cutoff = training_metadata.get(VALIDATION_START_METADATA)
        if cutoff is None:
            raise ValueError("The training Parquet must record its pickup-month cutoff.")
        provenance = json.loads(training_metadata.get(b"tlc_embedding_inputs", b"{}"))
        if "merged" in provenance and not _matches_signature(signatures[merged_path], provenance["merged"].encode("ascii")):
            raise ValueError("Use the same raw merge that was used for the embedding run.")

        rows = training.metadata.num_rows
        print_status(f"Matching saved cluster labels to {rows:,} training rows", flush=True)
        labels = _training_labels(assignments, merged, rows, date.fromisoformat(cutoff.decode("ascii") + "-01"))
        encoder = load_encoder(encoder_path) if uses_encoder else None
        vectors = _training_vectors(training, encoder)
        print_status(f"Fitting two-dimensional PCA on {rows:,} training embeddings", flush=True)
        pca = PCA(n_components=2, svd_solver="covariance_eigh").fit(vectors)
        if max_points and rows > max_points:
            sample = np.random.default_rng(config.CLUSTERING.sample_seed).choice(rows, max_points, replace=False)
            vectors, labels = vectors[sample], labels[sample]
        points = pca.transform(vectors)

    if any(io_part.file_signature(path) != signature for path, signature in signatures.items()):
        raise RuntimeError("A visualization input changed during processing. Rerun with stable inputs.")
    _save_plot(output_path, points, labels, pca.explained_variance_ratio_, rows)
    print_status(f"Saved training cluster PCA to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, help="Clustering assignments Parquet (default: latest completed run)")
    parser.add_argument("--training", type=Path, help="Training Parquet used by that clustering run")
    parser.add_argument("--merged", type=Path, default=config.MERGED_PARQUET, help="Raw merge used to select training rows")
    parser.add_argument("--encoder", type=Path, help="TorchScript encoder used by a preembedding clustering run")
    parser.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS, help="Maximum plotted points (0 for all)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination PNG path")
    args = parser.parse_args()
    assignments = args.assignments
    if assignments is None:
        runs = config.CLUSTER_OUTPUT_DIR.glob("run-*/cluster_assignments.parquet")
        assignments = max(runs, key=lambda path: (path.stat().st_mtime_ns, path.parent.name), default=None)
    if assignments is None:
        parser.error("No completed clustering run found. Run clustring_step.py or provide --assignments.")
    default_training = config.PREEMBEDDING_TRAINING_PARQUET if args.encoder else DEFAULT_TRAINING
    try:
        plot_training_clusters(
            assignments, args.training or default_training, args.merged,
            encoder_path=args.encoder, max_points=args.max_points, output_path=args.output,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
