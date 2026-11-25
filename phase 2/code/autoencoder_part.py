"""Train a small autoencoder and stream its encoder outputs to Parquet."""

import json
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from math import ceil, isfinite
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

import config
import io_part
from console import print_status
from feature_engineering_step import VALIDATION_START_METADATA, _paired_batches
from standardize_part import PICKUP_TIMESTAMP, raw_timestamps

EMBEDDING_VERSION = b"1"


def _linear_layers(widths: tuple[int, ...], negative_slope: float) -> nn.Sequential:
    layers = []
    for index, (input_width, output_width) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(input_width, output_width))
        if index < len(widths) - 2:
            layers.append(nn.LeakyReLU(negative_slope))
    return nn.Sequential(*layers)


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, settings: config.EmbeddingSettings = config.EMBEDDING) -> None:
        super().__init__()
        self.encoder = _linear_layers((input_dim, *settings.hidden_dims, settings.embedding_dim), settings.negative_slope)
        self.decoder = _linear_layers(
            (settings.embedding_dim, *reversed(settings.hidden_dims), input_dim), settings.negative_slope,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(features))


def _validate_settings(settings: config.EmbeddingSettings) -> None:
    sizes = (
        *settings.hidden_dims, settings.embedding_dim, settings.batch_size, settings.max_epochs,
        settings.patience, settings.validation_rows, settings.cpu_threads,
    )
    if not settings.hidden_dims or min(sizes) < 1:
        raise ValueError("Embedding dimensions, batch size, epochs, patience, validation size, and threads must be positive.")
    if any(not isfinite(value) or value <= 0 for value in (settings.learning_rate, settings.time_limit_seconds)):
        raise ValueError("The learning rate and training time limit must be finite and positive.")
    if any(not isfinite(value) or value < 0 for value in (settings.negative_slope, settings.min_delta)):
        raise ValueError("The activation slope and minimum improvement must be finite and nonnegative.")


def _input_columns(full: pq.ParquetFile, training: pq.ParquetFile) -> list[str]:
    schema = full.schema_arrow
    if not len(schema) or len(set(schema.names)) != len(schema) or not schema.equals(training.schema_arrow):
        raise ValueError("Full and training Parquets must have the same ordered, unique feature columns.")
    if any(not (pa.types.is_floating(field.type) or pa.types.is_integer(field.type)) for field in schema):
        raise ValueError("Embedding inputs must be numeric columns.")
    if not 0 < training.metadata.num_rows < full.metadata.num_rows:
        raise ValueError("Embedding requires both training trips and a chronological holdout.")
    return schema.names


def _validation_sample(
    full: pq.ParquetFile, training: pq.ParquetFile, merged: pq.ParquetFile, columns: list[str], limit: int,
) -> tuple[np.ndarray, bytes]:
    full_meta = full.metadata.metadata or {}
    train_meta = training.metadata.metadata or {}
    cutoff = full_meta.get(VALIDATION_START_METADATA)
    shared_keys = (
        VALIDATION_START_METADATA, b"tlc_preembedding_rules_sha256", b"tlc_preembedding_scalers",
        b"tlc_preembedding_working_input", b"tlc_preembedding_training_input",
    )
    if cutoff is None or any(full_meta.get(key) != train_meta.get(key) for key in shared_keys):
        raise ValueError("Use paired preembedding Parquets with a shared validation cutoff and scaling rules.")
    if full.metadata.num_rows != merged.metadata.num_rows or PICKUP_TIMESTAMP not in merged.schema_arrow.names:
        raise ValueError("The raw merge must match the full preembedding rows and contain pickup timestamps.")

    cutoff_date = pd.Timestamp(cutoff.decode("ascii"))
    training_rows = training.metadata.num_rows
    stride = ceil((full.metadata.num_rows - training_rows) / limit)
    before_cutoff = holdout_rows = 0
    samples = []
    for features, raw in _paired_batches(full, merged, columns, [PICKUP_TIMESTAMP]):
        pickups = raw_timestamps(raw, PICKUP_TIMESTAMP)
        before_cutoff += pickups.lt(cutoff_date).sum()
        eligible = np.flatnonzero(pickups.ge(cutoff_date).to_numpy())
        # Take every stride-th eligible row across the four held-out months.
        chosen = eligible[(-holdout_rows) % stride::stride]
        holdout_rows += len(eligible)
        if len(chosen):
            samples.append(io_part.numeric_batch_matrix(features.take(pa.array(chosen))))

    if not samples or before_cutoff != training_rows:
        raise ValueError("The raw pickup timestamps do not match the preembedding training and holdout split.")
    return np.concatenate(samples), cutoff


def _training_batches(
    source: pq.ParquetFile, settings: config.EmbeddingSettings, device: torch.device, rng: np.random.Generator,
) -> Iterator[torch.Tensor]:
    chunk_rows = max(1, config.CHUNK_ROWS // settings.batch_size) * settings.batch_size
    for batch in source.iter_batches(batch_size=chunk_rows):
        features = io_part.numeric_batch_matrix(batch)
        shuffled = features[rng.permutation(len(features))]
        yield from torch.from_numpy(shuffled).to(device).split(settings.batch_size)


@torch.inference_mode()
def _evaluate(model: Autoencoder, validation: torch.Tensor, batch_size: int, criterion: nn.HuberLoss) -> float:
    model.eval()
    loss_sum = torch.zeros((), device=validation.device)
    for batch in validation.split(batch_size):
        loss_sum += criterion(model(batch), batch) * len(batch)
    return loss_sum.item() / len(validation)


def _fit(
    training: pq.ParquetFile, validation: np.ndarray, settings: config.EmbeddingSettings, device: torch.device,
) -> tuple[Autoencoder, dict]:
    torch.manual_seed(settings.seed)
    rng = np.random.default_rng(settings.seed)
    model = Autoencoder(validation.shape[1], settings).to(device)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate, fused=device.type == "cuda")
    use_amp = device.type == "cuda" and settings.mixed_precision
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    validation_tensor = torch.from_numpy(validation).to(device)
    best_state = None
    best_loss = improvement_reference = float("inf")
    stale_epochs = steps = completed_checks = best_epoch = rows_seen = 0
    stop_reason = "max epochs"
    started = perf_counter()
    deadline = started + settings.time_limit_seconds
    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        train_loss_sum = torch.zeros((), device=device)
        epoch_rows = 0
        for batch in _training_batches(training, settings, device, rng):
            if steps and perf_counter() >= deadline:
                break
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(batch), batch)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            train_loss_sum += loss.detach() * len(batch)
            epoch_rows += len(batch)
            steps += 1

        if not epoch_rows:
            stop_reason = "time limit"
            break
        rows_seen += epoch_rows
        completed_checks += 1
        train_loss = train_loss_sum.item() / epoch_rows
        validation_loss = _evaluate(model, validation_tensor, settings.batch_size, criterion)
        if not (isfinite(train_loss) and isfinite(validation_loss)):
            raise ValueError("The autoencoder produced a nonfinite reconstruction loss.")
        if validation_loss < best_loss:
            best_state = deepcopy(model.state_dict())
            best_loss, best_epoch = validation_loss, epoch
        if validation_loss < improvement_reference - settings.min_delta:
            improvement_reference, stale_epochs = validation_loss, 0
        else:
            stale_epochs += 1

        print_status(
            f"Epoch {epoch} over {epoch_rows:,} rows, training Huber {train_loss:.6f}, "
            f"validation Huber {validation_loss:.6f}", flush=True,
        )
        if perf_counter() >= deadline:
            stop_reason = "time limit"
            break
        if stale_epochs >= settings.patience:
            stop_reason = "early stopping"
            break

    if best_state is None:
        raise RuntimeError("Training ended before a reconstruction could be evaluated.")
    model.load_state_dict(best_state)
    summary = {
        "validation_huber": best_loss, "best_epoch": best_epoch,
        "validation_checks": completed_checks, "validation_rows": len(validation),
        "training_steps": steps, "training_rows_seen": rows_seen,
        "training_seconds": perf_counter() - started, "stop_reason": stop_reason,
    }
    print_status(
        f"Training finished in {summary['training_seconds']:.1f}s ({stop_reason}), "
        f"best validation Huber {best_loss:.6f}", flush=True,
    )
    return model.eval(), summary


@torch.inference_mode()
def _write_embeddings(
    source: pq.ParquetFile, writer: pq.ParquetWriter, schema: pa.Schema, encoder: nn.Module, device: torch.device,
) -> None:
    for batch in source.iter_batches(batch_size=config.CHUNK_ROWS):
        features = torch.from_numpy(io_part.numeric_batch_matrix(batch)).to(device)
        embeddings = encoder(features).cpu().numpy()
        if not np.isfinite(embeddings).all():
            raise ValueError("The encoder produced nonfinite embeddings.")
        writer.write_batch(pa.RecordBatch.from_arrays([pa.array(column) for column in embeddings.T], schema=schema))


def _reusable(encoder_path: Path, full: Path, training: Path, expected: dict[bytes, bytes]) -> bool:
    if not encoder_path.is_file():
        return False
    digest = sha256(encoder_path.read_bytes()).hexdigest().encode("ascii")
    for path, split in ((full, b"full"), (training, b"training")):
        metadata = io_part.cached_parquet_metadata(path) or {}
        if metadata.get(config.EMBEDDING_ENCODER_HASH) != digest or metadata.get(b"tlc_embedding_split") != split:
            return False
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
    return True


def train_and_embed(
    full_path: Path,
    training_path: Path,
    merged_path: Path,
    *,
    output_dir: Path | None = None,
    settings: config.EmbeddingSettings = config.EMBEDDING,
    reuse_existing: bool = config.REUSE_EMBEDDING,
) -> tuple[Path, Path]:
    """Train from preembedding Parquets and return the full and training embedding paths."""
    _validate_settings(settings)
    output_dir = output_dir or full_path.parent / config.EMBEDDING_OUTPUT_DIR.name
    encoder_path = output_dir / "encoder.pt"
    full_output = output_dir / "embeddings.parquet"
    training_output = output_dir / "embeddings_training.parquet"
    inputs = {"full": full_path, "training": training_path, "merged": merged_path}
    paths = (*inputs.values(), encoder_path, full_output, training_output)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Choose distinct inputs and outputs for embedding.")
    signatures = {name: io_part.file_signature(path).decode("ascii") for name, path in inputs.items()}
    cache_metadata = {
        b"tlc_embedding_version": EMBEDDING_VERSION,
        b"tlc_embedding_inputs": json.dumps(signatures).encode("ascii"),
        b"tlc_embedding_settings": json.dumps(asdict(settings), allow_nan=False).encode("utf-8"),
    }
    if reuse_existing and _reusable(encoder_path, full_output, training_output, cache_metadata):
        print_status(f"Reusing encoder and embeddings in {output_dir}")
        return full_output, training_output

    device = torch.device(settings.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is unavailable with PyTorch {torch.__version__}. Install embedding_cuda_requirements.txt "
            "or choose device='cpu' in EmbeddingSettings for CPU training."
        )
    if device.type not in ("cuda", "cpu"):
        raise ValueError("Choose a CUDA or CPU device for embedding.")
    previous_threads = torch.get_num_threads()
    if device.type == "cpu":
        torch.set_num_threads(settings.cpu_threads)
    try:
        with pq.ParquetFile(full_path) as full, pq.ParquetFile(training_path) as training:
            columns = _input_columns(full, training)
            with pq.ParquetFile(merged_path) as merged:
                validation, cutoff = _validation_sample(full, training, merged, columns, settings.validation_rows)
            device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            print_status(
                f"Training {len(columns)} input features on {device_name} with {len(validation):,} validation rows",
                flush=True,
            )
            model, summary = _fit(training, validation, settings, device)

            run_details = {**summary, "input_columns": columns, "settings": asdict(settings)}
            details = json.dumps(run_details, allow_nan=False).encode("utf-8")
            portable_encoder = torch.jit.script(deepcopy(model.encoder).cpu())
            print_status(f"Encoding {full.metadata.num_rows:,} full and {training.metadata.num_rows:,} training rows", flush=True)
            with io_part.atomic_output_path(encoder_path) as staged_encoder:
                torch.jit.save(portable_encoder, str(staged_encoder), _extra_files={"embedding.json": details})
                metadata = {
                    **cache_metadata, VALIDATION_START_METADATA: cutoff, b"tlc_embedding_run": details,
                    config.EMBEDDING_ENCODER_HASH: sha256(staged_encoder.read_bytes()).hexdigest().encode("ascii"),
                }
                fields = [pa.field(f"embedding_{index:02d}", pa.float32(), nullable=False) for index in range(settings.embedding_dim)]
                full_schema = pa.schema(fields, metadata={**metadata, b"tlc_embedding_split": b"full"})
                training_schema = pa.schema(fields, metadata={**metadata, b"tlc_embedding_split": b"training"})
                with (
                    io_part.write_parquet(full_output, full_schema) as full_writer,
                    io_part.write_parquet(training_output, training_schema) as training_writer,
                ):
                    _write_embeddings(full, full_writer, full_schema, model.encoder, device)
                    _write_embeddings(training, training_writer, training_schema, model.encoder, device)
                    if any(io_part.file_signature(path).decode("ascii") != signatures[name] for name, path in inputs.items()):
                        raise RuntimeError("An embedding input changed during processing. Rerun with stable inputs.")
    finally:
        if device.type == "cpu":
            torch.set_num_threads(previous_threads)

    print_status(f"Saved encoder and embeddings to {output_dir}")
    return full_output, training_output
