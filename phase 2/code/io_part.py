"""Read the source CSV files and save the phase 2 Parquet files."""

# Version 2.0
import csv
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def read_csv_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        columns = next(csv.reader(stream), [])
    if not columns or "" in columns or len(columns) != len(set(columns)):
        raise ValueError(f"Missing, blank, or duplicate CSV headers in {path}")
    return columns


def read_csv_chunks(path: Path, chunk_rows: int):
    return pd.read_csv(path, encoding="utf-8-sig", chunksize=chunk_rows, dtype=str, keep_default_na=False)


def file_signature(path: Path) -> bytes:
    """Identify the current file version using its filesystem metadata."""
    stat = path.stat()
    return f"{stat.st_dev}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}".encode("ascii")


def cached_parquet_metadata(path: Path) -> dict[bytes, bytes] | None:
    """Read a cache footer without loading rows. Return None for a missing or invalid file."""
    try:
        return pq.read_metadata(path).metadata or {}
    except (FileNotFoundError, pa.ArrowInvalid):
        return None


@contextmanager
def read_parquet_batches(path: Path, chunk_rows: int, *, columns: list[str] | None = None):
    with pq.ParquetFile(path) as source:
        footer = source.metadata.metadata or {}
        yield source.schema_arrow, footer, source.iter_batches(batch_size=chunk_rows, columns=columns)


def numeric_batch_matrix(batch: pa.RecordBatch) -> np.ndarray:
    """Copy ordered Arrow columns into a contiguous float32 matrix for a model."""
    matrix = np.empty((batch.num_rows, batch.num_columns), dtype=np.float32)
    for index, column in enumerate(batch.columns):
        if column.null_count:
            raise ValueError(f"Input column {batch.schema.names[index]!r} contains nulls.")
        matrix[:, index] = column.to_numpy(zero_copy_only=False)
    if not np.isfinite(matrix).all():
        raise ValueError("Model inputs must be finite float32 values.")
    return matrix


@contextmanager
def atomic_output_path(path: Path):
    """Stage a file beside its destination and replace it after a successful write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as temp:
        temporary_path = Path(temp.name)

    try:
        yield temporary_path
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


@contextmanager
def write_parquet(path: Path, schema: pa.Schema):
    """Replace the destination only after the Parquet has been written successfully."""
    with atomic_output_path(path) as temporary_path:
        with pq.ParquetWriter(temporary_path, schema, compression="zstd") as writer:
            yield writer
