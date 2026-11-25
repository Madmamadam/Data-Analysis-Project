"""Select and fit MiniBatchKMeans on frozen encoder embeddings."""

import argparse
import json
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from uuid import uuid4

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib.figure import Figure
from sklearn import config_context
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

import config
import io_part
from console import print_status

Encoder = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class _Candidate:
    k: int
    scores: tuple[float | None, ...]
    centers: np.ndarray | None

    @property
    def mean(self) -> float | None:
        valid = [score for score in self.scores if score is not None]
        return sum(valid) / len(valid) if len(valid) == len(self.scores) else None


def load_encoder(path: Path, device: str = config.CLUSTERING.encoder_device) -> Encoder:
    """Load a TorchScript encoder whose forward method returns one vector per input row."""
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained TorchScript encoder not found: {path}")

    import torch

    model = torch.jit.load(str(path), map_location=device)
    model.eval()

    def encode(features: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            embeddings = model(torch.from_numpy(features).to(device))
            if not isinstance(embeddings, torch.Tensor):
                raise ValueError("The encoder must return a single embedding tensor.")
            return embeddings.to(device="cpu", dtype=torch.float32).numpy()

    return encode


def _validate_settings(settings: config.ClusteringSettings) -> None:
    candidates = settings.k_candidates
    if not candidates or min(candidates) < 2 or len(candidates) != len(set(candidates)):
        raise ValueError("K candidates must be distinct integers of at least 2.")
    if len(settings.seeds) != 3 or len(set(settings.seeds)) != 3:
        raise ValueError("Set three distinct random seeds for the K search.")
    if min(settings.training_rows, settings.batch_size) < max(candidates):
        raise ValueError("The training sample and batch size must each accommodate the largest K.")
    if settings.evaluation_rows <= max(candidates):
        raise ValueError("The silhouette sample must contain more rows than the largest K.")
    if min(settings.final_passes, settings.silhouette_memory_mib, settings.threads) < 1:
        raise ValueError("Final passes, silhouette memory, and thread count must be positive.")


def _validate_inputs(training: pq.ParquetFile, full: pq.ParquetFile, settings: config.ClusteringSettings) -> None:
    names = training.schema_arrow.names
    if not names or names != full.schema_arrow.names or len(set(names)) != len(names):
        raise ValueError("Training and full Parquets must have the same ordered, unique feature columns.")
    for source in (training, full):
        for field in source.schema_arrow:
            if not (pa.types.is_floating(field.type) or pa.types.is_integer(field.type)):
                raise ValueError(f"Input column {field.name!r} must be numeric.")

    training_encoder = (training.metadata.metadata or {}).get(config.EMBEDDING_ENCODER_HASH)
    full_encoder = (full.metadata.metadata or {}).get(config.EMBEDDING_ENCODER_HASH)
    if training_encoder != full_encoder:
        raise ValueError("Full and training embeddings must come from the same encoder.")

    required = settings.training_rows + settings.evaluation_rows
    if training.metadata.num_rows < required:
        raise ValueError(f"The training Parquet needs {required:,} rows for disjoint samples.")
    if full.metadata.num_rows < training.metadata.num_rows:
        raise ValueError("The full Parquet must cover at least as many rows as the training Parquet.")


def _embed(batch: pa.RecordBatch, encoder: Encoder | None) -> np.ndarray:
    features = io_part.numeric_batch_matrix(batch)
    if encoder is None:
        return features
    embeddings = np.ascontiguousarray(encoder(features), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != batch.num_rows or not embeddings.shape[1]:
        raise ValueError("The encoder must return a nonempty vector for every input row.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Encoder embeddings must be finite float32 values.")
    return embeddings


def _sample_embeddings(
    source: pq.ParquetFile, settings: config.ClusteringSettings, encoder: Encoder | None,
) -> tuple[np.ndarray, np.ndarray]:
    sample_size = settings.training_rows + settings.evaluation_rows
    rng = Random(settings.sample_seed)
    indices = np.asarray(rng.sample(range(source.metadata.num_rows), sample_size), dtype=np.int64)
    order = np.argsort(indices)
    sorted_indices = indices[order]

    samples = None
    row_start = left = right = 0
    for batch in source.iter_batches(batch_size=settings.batch_size):
        row_end = row_start + batch.num_rows
        right = sorted_indices.searchsorted(row_end)
        if right > left:
            selected = batch.take(pa.array(sorted_indices[left:right] - row_start))
            embeddings = _embed(selected, encoder)
            if samples is None:
                samples = np.empty((sample_size, embeddings.shape[1]), dtype=np.float32)
            elif embeddings.shape[1] != samples.shape[1]:
                raise ValueError("The encoder must return the same embedding width for every batch.")
            samples[order[left:right]] = embeddings
        if right == sample_size:
            break
        left, row_start = right, row_end

    if samples is None or right != sample_size:
        raise RuntimeError("The training Parquet ended before all sampled rows were read.")
    return samples[:settings.training_rows], samples[settings.training_rows:]


def _new_model(
    k: int, seed: int, settings: config.ClusteringSettings, centers: np.ndarray | None = None,
) -> MiniBatchKMeans:
    return MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=settings.batch_size,
        init=settings.init if centers is None else centers,
        init_size=settings.init_size,
        n_init=settings.n_init if centers is None else 1,
        max_iter=settings.max_iter,
        max_no_improvement=settings.max_no_improvement,
        tol=settings.tol,
        reassignment_ratio=settings.reassignment_ratio,
        compute_labels=False,
    )


def _evaluate(
    training: np.ndarray, evaluation: np.ndarray, settings: config.ClusteringSettings,
) -> list[_Candidate]:
    results = []
    for k in sorted(settings.k_candidates):
        scores = []
        best_score, centers = -np.inf, None
        for seed in settings.seeds:
            model = _new_model(k, seed, settings).fit(training)
            labels = model.predict(evaluation)
            score = None
            if 1 < np.unique(labels).size < len(evaluation):
                score = float(silhouette_score(evaluation, labels, metric=settings.silhouette_metric))
                if score > best_score:
                    best_score, centers = score, model.cluster_centers_.copy()
            scores.append(score)
            display = f"{score:.6f}" if score is not None else "undefined"
            print_status(f"K={k}, seed={seed}, silhouette={display}", flush=True)
        results.append(_Candidate(k, tuple(scores), centers))
    return results


def _plot_scores(path: Path, candidates: list[_Candidate], winner: _Candidate, seeds: tuple[int, ...]) -> None:
    figure = Figure(figsize=(9, 5), layout="constrained")
    axes = figure.subplots()
    positions = np.arange(len(candidates))
    for index, (seed, offset) in enumerate(zip(seeds, (-0.24, -0.12, 0.14), strict=True)):
        axes.scatter(
            positions + offset, [result.scores[index] for result in candidates], label=f"Seed {seed}", zorder=3,
        )
    axes.plot(positions, [result.mean for result in candidates], "D-", color="black", label="Mean")
    axes.set_xticks(positions, [str(result.k) for result in candidates])
    axes.set(xlabel="Number of clusters (K)", ylabel="Silhouette score", title=f"Silhouette scores (selected K = {winner.k})")
    axes.grid(axis="y", alpha=0.25)
    axes.legend()
    if any(result.mean is None for result in candidates):
        axes.set_xlabel("Number of clusters (K)\nOmitted points indicate undefined scores.")
    figure.savefig(path, dpi=160)


def _embedding_batches(
    source: pq.ParquetFile, batch_size: int, encoder: Encoder | None,
) -> Iterator[np.ndarray]:
    for batch in source.iter_batches(batch_size=batch_size):
        yield _embed(batch, encoder)


def _write_assignments(
    path: Path, model: MiniBatchKMeans, batches: Iterator[np.ndarray], metadata: bytes,
) -> None:
    schema = pa.schema(
        [pa.field("row_index", pa.int64(), nullable=False), pa.field("cluster", pa.int32(), nullable=False)],
        metadata={b"tlc_clustering": metadata},
    )
    row_start = 0
    with io_part.write_parquet(path, schema) as writer:
        for embeddings in batches:
            row_end = row_start + len(embeddings)
            row_indices = np.arange(row_start, row_end, dtype=np.int64)
            labels = model.predict(embeddings)
            writer.write_batch(pa.RecordBatch.from_arrays([row_indices, labels], schema=schema))
            row_start = row_end


def run_clustering(
    training_path: Path,
    full_path: Path,
    *,
    encoder: Encoder | None = None,
    output_dir: Path = config.CLUSTER_OUTPUT_DIR,
    settings: config.ClusteringSettings = config.CLUSTERING,
) -> Path:
    """Cluster saved embeddings or encode both Parquets with the supplied frozen encoder."""
    _validate_settings(settings)
    signatures = {path: io_part.file_signature(path) for path in (training_path, full_path)}
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / f"run-{uuid4().hex}"

    with TemporaryDirectory(dir=output_dir, prefix=".clustering-") as staging_dir:
        stage = Path(staging_dir)
        with (
            pq.ParquetFile(training_path) as training,
            pq.ParquetFile(full_path) as full,
            config_context(working_memory=settings.silhouette_memory_mib),
            threadpool_limits(limits=settings.threads),
        ):
            _validate_inputs(training, full, settings)
            training_sample, evaluation = _sample_embeddings(training, settings, encoder)
            candidates = _evaluate(training_sample, evaluation, settings)
            del training_sample, evaluation

            eligible = [candidate for candidate in candidates if candidate.mean is not None]
            if not eligible:
                raise ValueError("No K has defined silhouette scores for all three seeds.")
            winner = max(eligible, key=lambda candidate: (candidate.mean, -candidate.k))
            print_status(f"Selected K={winner.k} (mean silhouette={winner.mean:.6f})", flush=True)

            model = _new_model(winner.k, settings.final_seed, settings, winner.centers)
            for pass_number in range(settings.final_passes):
                print_status(f"Fitting pass {pass_number + 1} on {full.metadata.num_rows:,} rows", flush=True)
                for embeddings in _embedding_batches(full, settings.batch_size, encoder):
                    model.partial_fit(embeddings)

            joblib.dump(model, stage / "cluster_model.joblib")
            metadata = json.dumps({
                "settings": asdict(settings),
                "selected_k": winner.k,
                "training_input": training_path.name,
                "full_input": full_path.name,
                "training_signature": signatures[training_path].decode("ascii"),
                "full_signature": signatures[full_path].decode("ascii"),
                "input_kind": "preembedding" if encoder is not None else "embeddings",
                "scores": [{"k": result.k, "scores": result.scores, "mean": result.mean} for result in candidates],
            }, allow_nan=False).encode("utf-8")
            _write_assignments(
                stage / "cluster_assignments.parquet", model,
                _embedding_batches(full, settings.batch_size, encoder), metadata,
            )
            _plot_scores(stage / "silhouette_scores.png", candidates, winner, settings.seeds)

        if any(io_part.file_signature(path) != signature for path, signature in signatures.items()):
            raise RuntimeError("A clustering input changed during processing. Rerun clustering with stable inputs.")
        stage.rename(run_dir)

    print_status(f"Saved clustering outputs to {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, help="Training Parquet")
    parser.add_argument("--full", type=Path, help="Full Parquet")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--encoder", type=Path, help="Pretrained TorchScript encoder path")
    mode.add_argument("--embeddings", action="store_true", help="Use precomputed embeddings in both Parquets")
    parser.add_argument("--output-dir", type=Path, default=config.CLUSTER_OUTPUT_DIR, help="Parent for run outputs")
    args = parser.parse_args()
    if args.embeddings and (args.training is None or args.full is None):
        parser.error("--embeddings requires both --training and --full Parquets")

    encoder = None if args.embeddings else load_encoder(args.encoder or config.CLUSTER_ENCODER_PATH)
    run_clustering(
        args.training or config.PREEMBEDDING_TRAINING_PARQUET,
        args.full or config.PREEMBEDDING_PARQUET,
        encoder=encoder,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
