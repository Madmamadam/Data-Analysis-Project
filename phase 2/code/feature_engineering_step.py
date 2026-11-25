"""Add previous-interval traffic counts and cyclic request-time features.

Run after standardization. The helper Parquet spans the first through last
pickup or drop-off interval in the standardized local clock. Its counts are
reused across all batches of the working Parquet rewrite.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config
import io_part
from standardize_part import TIMESTAMP_PARTS

INTERVAL_SECONDS = 900
INTERVAL_START = "interval_start_datetime"
REQUEST_TIMESTAMP = "request_datetime"
EVENT_COUNTS = {"pickup_datetime": "pickup_count", "dropoff_datetime": "dropoff_count"}
# Each cycle starts at zero. Pandas numbers months starting at one.
REQUEST_CYCLES = {"minute": (60, 0), "hour": (24, 0), "day_of_week": (7, 0), "month": (12, 1)}
FEATURE_FIELDS = [
    *(pa.field(f"{count}_prev_15min", pa.int64()) for count in EVENT_COUNTS.values()),
    *(
        pa.field(f"request_{part}_{component}", pa.float64())
        for part in REQUEST_CYCLES
        for component in ("sin", "cos")
    ),
]


def _timestamp_columns(event: str) -> list[str]:
    return [f"{event}_{part}" for part in TIMESTAMP_PARTS]


def _timestamps(batch: pa.RecordBatch, event: str) -> pd.Series:
    parts = batch.select(_timestamp_columns(event)).to_pandas().set_axis(TIMESTAMP_PARTS, axis=1)
    complete = parts.notna().all(axis=1)
    # Use integers for datetime assembly even when Arrow returns nullable parts as floats.
    timestamps = pd.to_datetime(parts.fillna(0).astype("int64"), errors="coerce")
    return timestamps.where(complete)


def _interval_indexes(timestamps: pd.Series, first_second: int) -> np.ndarray:
    seconds = timestamps.to_numpy(dtype="datetime64[s]").astype(np.int64)
    return (seconds - first_second) // INTERVAL_SECONDS


def _build_interval_table(source: pq.ParquetFile) -> pa.Table:
    columns = [column for event in EVENT_COUNTS for column in _timestamp_columns(event)]
    bounds = []
    for batch in source.iter_batches(batch_size=config.CHUNK_ROWS, columns=columns):
        for event in EVENT_COUNTS:
            timestamps = _timestamps(batch, event)
            first = timestamps.min()
            if pd.notna(first):
                bounds.append((first, timestamps.max()))

    first_second = interval_count = 0
    if bounds:
        first = min(start for start, _ in bounds)
        last = max(end for _, end in bounds)
        first_bin = first.value // (INTERVAL_SECONDS * 10**9)
        last_bin = last.value // (INTERVAL_SECONDS * 10**9)
        first_second = first_bin * INTERVAL_SECONDS
        interval_count = last_bin - first_bin + 1

    counts = {name: np.zeros(interval_count, dtype=np.int64) for name in EVENT_COUNTS.values()}
    if interval_count:
        for batch in source.iter_batches(batch_size=config.CHUNK_ROWS, columns=columns):
            for event, name in EVENT_COUNTS.items():
                indexes = _interval_indexes(_timestamps(batch, event).dropna(), first_second)
                counts[name] += np.bincount(indexes, minlength=interval_count)

    seconds = first_second + np.arange(interval_count, dtype=np.int64) * INTERVAL_SECONDS
    starts = pa.array(seconds.astype("datetime64[s]").astype("datetime64[ms]"), type=pa.timestamp("ms"))
    return pa.table({INTERVAL_START: starts, **counts})


def _feature_arrays(
    request: pd.Series, first_second: int, counts: list[np.ndarray]
) -> list[pa.Array]:
    valid = request.notna().to_numpy()
    indexes = np.full(len(request), -1, dtype=np.int64)
    indexes[valid] = _interval_indexes(request[valid], first_second) - 1
    covered = (indexes >= 0) & (indexes < len(counts[0]))

    features = []
    for interval_counts in counts:
        previous = np.zeros(len(request), dtype=np.int64)
        previous[covered] = interval_counts[indexes[covered]]
        features.append(pa.array(previous, mask=~covered))

    for part, (period, offset) in REQUEST_CYCLES.items():
        values = getattr(request.dt, part).to_numpy(dtype=np.float64, na_value=np.nan)
        angle = 2 * np.pi * (values - offset) / period
        features.append(pa.array(np.sin(angle), from_pandas=True))
        features.append(pa.array(np.cos(angle), from_pandas=True))
    return features


def add_features(
    working_path: Path = config.WORKING_PARQUET, interval_path: Path | None = None
) -> Path:
    """Build the interval table, then atomically update the standardized Parquet."""
    if interval_path is None:
        interval_path = working_path.with_name("trip_counts_15min.parquet")
    if interval_path.resolve() == working_path.resolve():
        raise ValueError("Choose different paths for the interval table and the working Parquet.")

    original_stat = working_path.stat()
    with pq.ParquetFile(working_path) as source:
        source_schema = source.schema_arrow
        required = [
            column
            for event in (REQUEST_TIMESTAMP, *EVENT_COUNTS)
            for column in _timestamp_columns(event)
        ]
        missing = sorted(set(required).difference(source_schema.names))
        if missing:
            raise ValueError(f"Missing standardized timestamp fields in {working_path}: {', '.join(missing)}")

        intervals = _build_interval_table(source)
        first_second = intervals[INTERVAL_START][0].value // 1000 if intervals.num_rows else 0
        counts = [intervals[name].to_numpy() for name in EVENT_COUNTS.values()]
        feature_names = {field.name for field in FEATURE_FIELDS}
        original_fields = [field for field in source_schema if field.name not in feature_names]
        original_names = [field.name for field in original_fields]
        output_schema = pa.schema([*original_fields, *FEATURE_FIELDS], metadata=source_schema.metadata)

        with io_part.write_parquet(working_path, output_schema) as writer:
            for batch in source.iter_batches(batch_size=config.CHUNK_ROWS, columns=original_names):
                original = batch.select(original_names)
                request = _timestamps(original, REQUEST_TIMESTAMP)
                features = _feature_arrays(request, first_second, counts)
                augmented = pa.RecordBatch.from_arrays([*original.columns, *features], schema=output_schema)
                writer.write_batch(augmented)

            # Windows requires the input file to be closed before the atomic replacement.
            source.close()
            current_stat = working_path.stat()
            if (current_stat.st_ino, current_stat.st_size, current_stat.st_mtime_ns) != (
                original_stat.st_ino, original_stat.st_size, original_stat.st_mtime_ns
            ):
                raise RuntimeError(f"{working_path} changed during feature engineering. Run the step again.")
            with io_part.write_parquet(interval_path, intervals.schema) as interval_writer:
                interval_writer.write_table(intervals)

    print(f"Saved {intervals.num_rows:,} trip-count intervals to {interval_path}")
    print(f"Added trip and request-time features to {working_path}")
    return working_path


if __name__ == "__main__":
    add_features()
