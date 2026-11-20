"""Read and write the CSV and Parquet files used in phase 2."""

import csv
import re
import shutil
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from itertools import count
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


WINDOWS_DEVICE_NAME = re.compile(
    r"CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|(?:COM|LPT)[1-9¹²³]", re.IGNORECASE,
)


@contextmanager
def staged_path(directory: Path) -> Iterator[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as file:
        path = Path(file.name)

    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def replace_file(source: Path, target: Path) -> None:
    source.replace(target)


def _replace_temp_file(source: Path, target: Path) -> Path:
    with staged_path(target.parent) as pending:
        shutil.copyfile(source, pending)
        replace_file(pending, target)
    return target


def _publish_snapshot(pending: Path, target: Path) -> Path:
    timestamp = datetime.now().strftime("%H-%M")
    for number in count(1):
        suffix = "" if number == 1 else f" ({number})"
        candidate = target.with_name(f"{target.stem}_{timestamp}{suffix}{target.suffix}")
        try:
            candidate.touch(exist_ok=False)
        except FileExistsError:
            continue

        try:
            replace_file(pending, candidate)
        except BaseException:
            candidate.unlink(missing_ok=True)
            raise
        return candidate


def _save_output_snapshot(source: Path, target: Path) -> Path:
    with staged_path(target.parent) as pending:
        shutil.copyfile(source, pending)
        return _publish_snapshot(pending, target)


def save_file(source_file: str | Path, destination_folder: str | Path, name: str) -> Path:
    """Copy a file to temp_file or output using the name supplied by the caller."""
    source = Path(source_file)
    folder = Path(destination_folder)
    if name and not name.endswith((" ", ".")) and not Path(name).suffix:
        name += source.suffix
    if (
        not name
        or name.endswith((" ", "."))
        or any(ord(char) < 32 or char in '<>:"/\\|?*' for char in name)
        or WINDOWS_DEVICE_NAME.fullmatch(name.split(".", 1)[0].rstrip())
    ):
        raise ValueError(f"Invalid Windows filename: {name!r}")

    target = folder / name
    if folder.name.casefold() == "temp_file":
        return _replace_temp_file(source, target)
    if folder.name.casefold() == "output":
        return _save_output_snapshot(source, target)
    raise ValueError("Destination folder must be named temp_file or output.")


def read_csv_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        columns = next(csv.reader(stream), [])
    if not columns or "" in columns or len(columns) != len(set(columns)):
        raise ValueError(f"Missing, blank, or duplicate CSV headers in {path}")
    return columns


def read_csv_chunks(path: Path, chunk_rows: int) -> pd.io.parsers.TextFileReader:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        chunksize=chunk_rows,
        dtype=str,
        keep_default_na=False,
    )


@contextmanager
def read_parquet_batches(
    path: Path, chunk_rows: int,
) -> Iterator[tuple[pa.Schema, Iterator[pa.RecordBatch]]]:
    with pq.ParquetFile(path) as source:
        yield source.schema_arrow, source.iter_batches(batch_size=chunk_rows)


def write_parquet_batches(
    path: Path, schema: pa.Schema, batches: Iterable[pa.Table | pa.RecordBatch],
) -> None:
    with pq.ParquetWriter(path, schema, compression="zstd") as writer:
        for batch in batches:
            writer.write(batch)


def export_working_csv(
    working_parquet: Path,
    output_dir: Path,
    *,
    preview: bool,
    preview_rows: int,
    chunk_rows: int,
) -> Path:
    """Export all rows, or a small preview, without loading the whole table."""
    name = "working_preview.csv" if preview else "working.csv"
    remaining = preview_rows if preview else None
    batch_size = chunk_rows if remaining is None else min(chunk_rows, remaining)

    with staged_path(output_dir) as pending:
        with (
            read_parquet_batches(working_parquet, batch_size) as (schema, batches),
            pending.open("w", encoding="utf-8-sig", newline="") as output,
        ):
            csv.writer(output).writerow(schema.names)
            for batch in batches:
                if remaining is not None:
                    batch = batch.slice(0, remaining)
                    remaining -= batch.num_rows
                batch.to_pandas(types_mapper=pd.ArrowDtype).to_csv(
                    output, index=False, header=False, lineterminator="\r\n",
                )
                if remaining == 0:
                    break
        return _publish_snapshot(pending, output_dir / name)
