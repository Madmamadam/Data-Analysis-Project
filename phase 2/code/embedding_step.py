"""Prepare numerical inputs and train an autoencoder for trip embeddings."""

import argparse
import json
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from math import sqrt
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import config
import io_part
from console import print_status
from feature_engineering_step import VALIDATION_START_METADATA

PASS_THROUGH = -1
EXCLUDE = -2
STANDARD_SCALE = 1
MASKED_SCALE = 2
PREEMBEDDING_VERSION = b"1"
SCALERS_METADATA = b"tlc_preembedding_scalers"


@dataclass(frozen=True)
class _Rule:
    code: int
    mask: str | None


@dataclass(frozen=True)
class _Scaler:
    mean: float
    scale: float
    count: int


@dataclass
class _RunningStats:
    count: int = 0
    mean: float = 0.0
    squared_deviations: float = 0.0

    def update(self, values: np.ndarray) -> None:
        batch_count = len(values)
        if not batch_count:
            return

        # Pool each batch's mean and variance without retaining its rows.
        origin = float(values[0])
        offsets = values - origin
        batch_mean = origin + float(offsets.mean())
        total = self.count + batch_count
        difference = batch_mean - self.mean
        batch_weight = batch_count / total
        between_batches = self.count * batch_weight * difference
        self.squared_deviations += float(offsets.var()) * batch_count + difference * between_batches
        self.mean += difference * batch_weight
        self.count = total

    def scaler(self) -> _Scaler:
        scale = sqrt(self.squared_deviations / self.count) if self.count else 0.0
        return _Scaler(self.mean, scale or 1.0, self.count)


def _load_rules(data: bytes) -> dict[str, _Rule]:
    definitions = json.loads(data)
    if not isinstance(definitions, dict):
        raise ValueError("Preembedding rules must be a JSON object keyed by feature name.")

    rules = {}
    for name, definition in definitions.items():
        code = definition.get("code") if isinstance(definition, dict) else None
        if type(code) is not int or code not in (PASS_THROUGH, EXCLUDE, STANDARD_SCALE, MASKED_SCALE):
            raise ValueError(f"Preembedding rule for {name!r} must have code -2, -1, 1, or 2.")
        mask = definition.get("mask")
        if code == MASKED_SCALE:
            if not isinstance(mask, str) or not mask.strip() or mask == name:
                raise ValueError(f"Preembedding rule for {name!r} must name another column as its mask.")
        elif "mask" in definition:
            raise ValueError(f"Preembedding rule for {name!r} can only set a mask with code 2.")
        if code != EXCLUDE:
            rules[name] = _Rule(code, mask)
    return rules


def _input_columns(rules: dict[str, _Rule]) -> list[str]:
    masks = (rule.mask for rule in rules.values() if rule.mask is not None)
    return list(dict.fromkeys([*rules, *masks]))


def _select_rules(rules: dict[str, _Rule], working: pq.ParquetFile, training: pq.ParquetFile) -> dict[str, _Rule]:
    available = set(working.schema_arrow.names) | set(training.schema_arrow.names)
    selected = {name: rule for name, rule in rules.items() if name in available}
    if not selected:
        raise ValueError("No Parquet columns match the included preembedding rules.")
    columns = _input_columns(selected)
    for role, schema in (("working", working.schema_arrow), ("training", training.schema_arrow)):
        for name in columns:
            index = schema.get_field_index(name)
            if index < 0:
                raise ValueError(f"Expected one preembedding input named {name!r} in the {role} Parquet.")
            data_type = schema.field(index).type
            if not (pa.types.is_integer(data_type) or pa.types.is_floating(data_type) or pa.types.is_boolean(data_type)):
                raise ValueError(f"Preembedding input {name!r} in the {role} Parquet must be numeric or boolean.")
    return selected


def _usable_values(batch: pa.RecordBatch, name: str, mask: str | None) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(batch.column(name).to_numpy(zero_copy_only=False), dtype=np.float64)
    valid = np.isfinite(values)
    if mask is not None:
        valid &= batch.column(mask).to_numpy(zero_copy_only=False) == 1
    return values, valid


def _fit_scalers(training: pq.ParquetFile, rules: dict[str, _Rule]) -> dict[str, _Scaler]:
    scaled = {name: rule for name, rule in rules.items() if rule.code != PASS_THROUGH}
    stats = {name: _RunningStats() for name in scaled}
    if scaled:
        for batch in training.iter_batches(batch_size=config.CHUNK_ROWS, columns=_input_columns(scaled)):
            for name, rule in scaled.items():
                values, valid = _usable_values(batch, name, rule.mask)
                stats[name].update(values[valid])
    return {name: summary.scaler() for name, summary in stats.items()}


def _write_transformed(
    source: pq.ParquetFile, writer: pq.ParquetWriter, schema: pa.Schema,
    rules: dict[str, _Rule], scalers: dict[str, _Scaler],
) -> None:
    for batch in source.iter_batches(batch_size=config.CHUNK_ROWS, columns=_input_columns(rules)):
        columns = []
        for name, rule in rules.items():
            values, valid = _usable_values(batch, name, rule.mask)
            values = values[valid]
            scaler = scalers.get(name)
            result = np.zeros(batch.num_rows, dtype=np.float32)
            with np.errstate(over="ignore", invalid="ignore"):
                if scaler is not None:
                    values = (values - scaler.mean) / scaler.scale
                result[valid] = values
            if not np.isfinite(result).all():
                raise ValueError(f"Preembedding output for {name!r} exceeds the finite float32 range.")
            columns.append(pa.array(result))
        writer.write_batch(pa.RecordBatch.from_arrays(columns, schema=schema))


def _cache_matches(path: Path, expected: dict[bytes, bytes]) -> bool:
    metadata = io_part.cached_parquet_metadata(path) or {}
    return SCALERS_METADATA in metadata and all(metadata.get(key) == value for key, value in expected.items())


def prepare_preembedding(
    working_path: Path = config.WORKING_PARQUET,
    training_path: Path | None = None,
    *,
    rules_path: Path = config.PREEMBEDDING_RULES,
    reuse_existing: bool = config.REUSE_PREEMBEDDING,
) -> tuple[Path, Path]:
    """Fit on the training subset and write full and training preembedding Parquets."""
    if training_path is None:
        training_path = working_path.with_name(config.TRAINING_PARQUET.name)
    full_output = working_path.with_name(config.PREEMBEDDING_PARQUET.name)
    training_output = working_path.with_name(config.PREEMBEDDING_TRAINING_PARQUET.name)
    paths = (working_path, training_path, rules_path, full_output, training_output)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Choose distinct paths for the preembedding inputs and outputs.")

    rules_data = rules_path.read_bytes()
    signatures = {path: io_part.file_signature(path) for path in (working_path, training_path)}
    cache_metadata = {
        b"tlc_preembedding": PREEMBEDDING_VERSION,
        b"tlc_preembedding_rules_sha256": sha256(rules_data).hexdigest().encode("ascii"),
        b"tlc_preembedding_working_input": signatures[working_path],
        b"tlc_preembedding_training_input": signatures[training_path],
    }
    full_metadata = {**cache_metadata, b"tlc_preembedding_split": b"full"}
    training_metadata = {**cache_metadata, b"tlc_preembedding_split": b"training"}
    if (
        reuse_existing
        and _cache_matches(full_output, full_metadata)
        and _cache_matches(training_output, training_metadata)
    ):
        print_status(f"Reusing preembedding data: {full_output} and {training_output}")
        return full_output, training_output

    with pq.ParquetFile(working_path) as working, pq.ParquetFile(training_path) as training:
        rules = _select_rules(_load_rules(rules_data), working, training)
        cutoff = (working.metadata.metadata or {}).get(VALIDATION_START_METADATA)
        if cutoff != (training.metadata.metadata or {}).get(VALIDATION_START_METADATA):
            raise ValueError("Working and training Parquets must have the same validation cutoff.")

        print_status(f"Fitting preembedding scales on {training.metadata.num_rows:,} training rows", flush=True)
        scalers = _fit_scalers(training, rules)
        parameters = {name: asdict(scaler) for name, scaler in scalers.items()}
        fitted_metadata = {SCALERS_METADATA: json.dumps(parameters, allow_nan=False).encode("utf-8")}
        if cutoff is not None:
            fitted_metadata[VALIDATION_START_METADATA] = cutoff
        schema = pa.schema([pa.field(name, pa.float32(), nullable=False) for name in rules])
        full_schema = schema.with_metadata({**full_metadata, **fitted_metadata})
        training_schema = schema.with_metadata({**training_metadata, **fitted_metadata})
        with (
            io_part.write_parquet(full_output, full_schema) as full_writer,
            io_part.write_parquet(training_output, training_schema) as training_writer,
        ):
            _write_transformed(training, training_writer, training_schema, rules, scalers)
            _write_transformed(working, full_writer, full_schema, rules, scalers)
            if any(io_part.file_signature(path) != signature for path, signature in signatures.items()):
                raise RuntimeError("A preembedding input changed during processing. Run the pipeline again.")
            if rules_path.read_bytes() != rules_data:
                raise RuntimeError("The preembedding rules changed during processing. Run the pipeline again.")

        print_status(f"Saved {working.metadata.num_rows:,} rows with {len(rules)} preembedding features to {full_output}")
        print_status(f"Saved {training.metadata.num_rows:,} preembedding training rows to {training_output}")
    return full_output, training_output


def main(
    working_path: Path = config.WORKING_PARQUET,
    training_path: Path | None = None,
    *,
    rules_path: Path = config.PREEMBEDDING_RULES,
    reuse_existing: bool = config.REUSE_PREEMBEDDING,
    merged_path: Path | None = None,
    preembedding_only: bool = False,
    settings: config.EmbeddingSettings = config.EMBEDDING,
) -> tuple[Path, Path]:
    """Prepare both Parquets and optionally return their trained embeddings."""
    full, training = prepare_preembedding(working_path, training_path, rules_path=rules_path, reuse_existing=reuse_existing)
    if preembedding_only:
        return full, training

    from autoencoder_part import train_and_embed

    merged_path = merged_path or working_path.with_name(config.MERGED_PARQUET.name)
    return train_and_embed(full, training, merged_path, settings=settings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preembedding-only", action="store_true", help="Write scaled features only")
    parser.add_argument("--device", default=config.EMBEDDING.device, help="PyTorch device for training")
    parser.add_argument("--time-limit", type=float, default=config.EMBEDDING.time_limit_seconds, help="Training limit in seconds")
    args = parser.parse_args()
    main(
        preembedding_only=args.preembedding_only,
        settings=replace(config.EMBEDDING, device=args.device, time_limit_seconds=args.time_limit),
    )
