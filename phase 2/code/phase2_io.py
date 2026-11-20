"""Read the source CSV files and save the phase 2 Parquet files."""

# Version 2.0
import csv
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

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


@contextmanager
def read_parquet_batches(path: Path, chunk_rows: int):
    with pq.ParquetFile(path) as source:
        footer = source.metadata.metadata or {}
        yield source.schema_arrow, footer, source.iter_batches(batch_size=chunk_rows)


@contextmanager
def write_parquet(path: Path, schema: pa.Schema):
    """Replace the destination only after the Parquet has been written successfully."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as temp:
        temporary_path = Path(temp.name)

    try:
        with pq.ParquetWriter(temporary_path, schema, compression="zstd") as writer:
            yield writer
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
