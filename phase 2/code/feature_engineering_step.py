"""Add trip features and write the chronological training subset.

Run after standardization. The raw merge supplies the local timestamps for
the training split and the pickup and drop-off interval counts.
"""

from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as arrow_csv
import pyarrow.parquet as pq

import config
import io_part
from standardize_part import (
    DROPOFF_TIMESTAMP,
    MERGED_INPUT_METADATA,
    PICKUP_TIMESTAMP,
    REQUEST_TIMESTAMP,
    STANDARDIZATION_VERSION,
    STANDARDIZED_METADATA,
    raw_timestamps,
    standardize_parquet,
)

INTERVAL_SECONDS = 900
INTERVAL_START = "interval_start_datetime"
EVENT_COUNTS = {PICKUP_TIMESTAMP: "pickup_count", DROPOFF_TIMESTAMP: "dropoff_count"}
VALIDATION_MONTHS = 4
FEATURES_METADATA = b"tlc_feature_engineering"
FEATURES_VERSION = b"6"
INTERVAL_FILE_METADATA = b"tlc_feature_interval_file"
ZONE_LOOKUP_METADATA = b"tlc_feature_zone_lookup"
ZONE_SHARES_FILE_METADATA = b"tlc_feature_zone_shares_file"
TRAINING_FILE_METADATA = b"tlc_feature_training_file"
VALIDATION_START_METADATA = b"tlc_validation_start_month"
LOCATION_IDS = {"PULocationID": "pickup", "DOLocationID": "dropoff"}
ZONE_ATTRIBUTES = {"latitude": pa.float64(), "longitude": pa.float64(), "is_airport": pa.uint8()}
ZONE_SHARE = "trip_endpoint_share"
LOCATION_ATTRIBUTES = {**ZONE_ATTRIBUTES, "location_valid": pa.uint8(), ZONE_SHARE: pa.float64()}
LOCATION_FIELDS = [
    pa.field(f"{endpoint}_{name}", data_type, nullable=False)
    for endpoint in LOCATION_IDS.values()
    for name, data_type in LOCATION_ATTRIBUTES.items()
]
FEATURE_FIELDS = [pa.field(f"{count}_prev_15min", pa.int64()) for count in EVENT_COUNTS.values()]


def _paired_batches(source: pq.ParquetFile, merged: pq.ParquetFile, columns, time_columns):
    """Read matching row slices even when the files yield different batch sizes."""
    batches = source.iter_batches(batch_size=config.CHUNK_ROWS, columns=columns)
    raw_batches = merged.iter_batches(batch_size=config.CHUNK_ROWS, columns=time_columns)
    batch, raw = next(batches, None), next(raw_batches, None)
    while batch is not None and raw is not None:
        rows = min(batch.num_rows, raw.num_rows)
        yield batch.slice(0, rows), raw.slice(0, rows)
        batch = next(batches, None) if rows == batch.num_rows else batch.slice(rows)
        raw = next(raw_batches, None) if rows == raw.num_rows else raw.slice(rows)
    if batch is not None or raw is not None:
        raise ValueError("The working Parquet and raw merge have different row counts.")


def _before_validation(timestamps: pd.Series, validation_start: int) -> pa.Array:
    month = timestamps.dt.year * 12 + timestamps.dt.month
    return pa.array(month.lt(validation_start))


def _load_zone_lookup(data: bytes) -> pa.RecordBatch:
    types = {"LocationID": pa.int32(), **dict.fromkeys(ZONE_ATTRIBUTES, pa.string())}
    table = arrow_csv.read_csv(
        pa.py_buffer(data),
        convert_options=arrow_csv.ConvertOptions(include_columns=list(types), column_types=types),
    )
    ids = table["LocationID"].combine_chunks()
    if not table.num_rows or ids.null_count or pc.count_distinct(ids).as_py() != table.num_rows:
        raise ValueError("The taxi-zone lookup must contain unique, nonempty LocationIDs.")

    values = {name: pd.to_numeric(table[name].to_pandas(), errors="coerce") for name in ZONE_ATTRIBUTES}
    # Use the same NYC and Newark coordinate bounds as the lookup updater.
    valid = (
        values["latitude"].between(40.45, 40.95, inclusive="neither")
        & values["longitude"].between(-74.30, -73.65, inclusive="neither")
        & values["is_airport"].isin([0, 1])
        & ~ids.to_pandas().isin([264, 265])
    )
    columns = [pa.array(values[name].where(valid, 0), type=data_type) for name, data_type in ZONE_ATTRIBUTES.items()]
    columns.append(pa.array(valid.astype("uint8")))
    return pa.RecordBatch.from_arrays([ids, *columns], names=["LocationID", *ZONE_ATTRIBUTES, "location_valid"])


def _build_zone_shares(
    source: pq.ParquetFile, merged: pq.ParquetFile, zones: pa.RecordBatch, validation_start: int | None
) -> pa.Table:
    """Count mapped endpoints observed before the validation period."""
    zone_ids = zones.column("LocationID")
    mapped = zones.column("location_valid").to_numpy().astype(bool)
    counts = {f"{endpoint}_count": np.zeros(len(zone_ids), dtype=np.int64) for endpoint in LOCATION_IDS.values()}
    id_columns = {name: endpoint for name, endpoint in LOCATION_IDS.items() if name in source.schema_arrow.names}
    if validation_start is not None and id_columns:
        for batch, raw in _paired_batches(source, merged, list(id_columns), list(EVENT_COUNTS)):
            pickup = _before_validation(raw_timestamps(raw, PICKUP_TIMESTAMP), validation_start)
            dropoff = _before_validation(raw_timestamps(raw, DROPOFF_TIMESTAMP), validation_start)
            masks = {"pickup": pickup, "dropoff": pc.and_(pickup, dropoff)}
            for id_column, endpoint in id_columns.items():
                ids = batch.column(id_column).filter(masks[endpoint])
                indexes = pc.index_in(ids, value_set=zone_ids, skip_nulls=True)
                counts[f"{endpoint}_count"] += np.bincount(pc.drop_null(indexes).to_numpy(), minlength=len(zone_ids))

    for values in counts.values():
        values[~mapped] = 0
    endpoints = counts["pickup_count"] + counts["dropoff_count"]
    shares = endpoints / max(1, endpoints.sum())
    return pa.table({"LocationID": zone_ids, **counts, ZONE_SHARE: shares})


def _attach_zone_shares(zones: pa.RecordBatch, csv_path: Path) -> pa.RecordBatch:
    types = {"LocationID": pa.int32(), ZONE_SHARE: pa.float64()}
    lookup = arrow_csv.read_csv(
        csv_path, convert_options=arrow_csv.ConvertOptions(include_columns=list(types), column_types=types)
    ).combine_chunks()
    if not lookup["LocationID"].combine_chunks().equals(zones.column("LocationID")):
        raise ValueError("The zone share CSV does not match the taxi-zone lookup IDs.")
    return zones.append_column(ZONE_SHARE, lookup[ZONE_SHARE].combine_chunks())


def _location_arrays(batch: pa.RecordBatch, zones: pa.RecordBatch) -> list[pa.Array]:
    features = []
    for id_column in LOCATION_IDS:
        if id_column not in batch.schema.names:
            features.extend(
                pa.repeat(pa.scalar(0, type=data_type), batch.num_rows) for data_type in LOCATION_ATTRIBUTES.values()
            )
            continue
        indexes = pc.index_in(batch.column(id_column), value_set=zones.column("LocationID"), skip_nulls=True)
        features.extend(pc.fill_null(pc.take(zones.column(name), indexes), 0) for name in LOCATION_ATTRIBUTES)
    return features


def _interval_indexes(timestamps: pd.Series, first_second: int) -> np.ndarray:
    seconds = timestamps.to_numpy(dtype="datetime64[s]").astype(np.int64)
    return (seconds - first_second) // INTERVAL_SECONDS


def _build_interval_table(merged: pq.ParquetFile) -> tuple[pa.Table, int | None]:
    counts = {name: Counter() for name in EVENT_COUNTS.values()}
    for batch in merged.iter_batches(batch_size=config.CHUNK_ROWS, columns=list(EVENT_COUNTS)):
        for event, name in EVENT_COUNTS.items():
            indexes = _interval_indexes(raw_timestamps(batch, event).dropna(), 0)
            bins, frequencies = np.unique(indexes, return_counts=True)
            counts[name].update(dict(zip(bins, frequencies)))

    first_bin = min((min(tally) for tally in counts.values() if tally), default=0)
    last_bin = max((max(tally) for tally in counts.values() if tally), default=-1)
    seconds = np.arange(first_bin, last_bin + 1, dtype=np.int64) * INTERVAL_SECONDS
    starts = pa.array(seconds.astype("datetime64[s]").astype("datetime64[ms]"), type=pa.timestamp("ms"))
    columns = {INTERVAL_START: starts}
    for name, tally in counts.items():
        values = np.zeros(len(starts), dtype=np.int64)
        indexes = np.fromiter(tally, dtype=np.int64) - first_bin
        values[indexes] = np.fromiter(tally.values(), dtype=np.int64)
        columns[name] = values

    # The last pickup interval starts in the same calendar month as that pickup.
    last_pickup = max(counts[EVENT_COUNTS[PICKUP_TIMESTAMP]], default=None)
    validation_start = None
    if last_pickup is not None:
        latest = pd.Timestamp(int(last_pickup) * INTERVAL_SECONDS, unit="s")
        validation_start = latest.year * 12 + latest.month - VALIDATION_MONTHS + 1
    return pa.table(columns), validation_start


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

    return features


def add_features(
    working_path: Path = config.WORKING_PARQUET,
    interval_path: Path | None = None,
    *,
    reuse_existing: bool = False,
    merged_path: Path | None = None,
    zone_lookup_path: Path = config.TAXI_ZONE_LOOKUP,
    training_path: Path | None = None,
    zone_shares_path: Path | None = None,
) -> Path:
    """Reuse a finished feature stage or atomically add it to the working Parquet."""
    if interval_path is None:
        interval_path = working_path.with_name("trip_counts_15min.parquet")
    if training_path is None:
        training_path = working_path.with_name("training.parquet")
    if zone_shares_path is None:
        zone_shares_path = working_path.with_name("zone_trip_shares.csv")
    if merged_path is None:
        merged_path = working_path.with_name(config.MERGED_PARQUET.name)
    destinations = {path.resolve() for path in (working_path, interval_path, training_path, zone_shares_path)}
    inputs = {zone_lookup_path.resolve(), merged_path.resolve()}
    if len(destinations) != 4 or destinations.intersection(inputs):
        raise ValueError("Choose distinct paths for the feature outputs and input files.")

    lookup_data = zone_lookup_path.read_bytes()
    lookup_signature = sha256(lookup_data).hexdigest().encode("ascii")
    merged_signature = io_part.file_signature(merged_path)
    original_signature = io_part.file_signature(working_path)
    shares_signature = io_part.file_signature(zone_shares_path) if zone_shares_path.is_file() else None
    with pq.ParquetFile(working_path) as source:
        source_schema = source.schema_arrow
        cached = source.metadata.metadata or {}
        stale_standardization = (
            cached.get(STANDARDIZED_METADATA) != STANDARDIZATION_VERSION
            or cached.get(MERGED_INPUT_METADATA) != merged_signature
        )
        current_locations = (
            not stale_standardization
            and cached.get(FEATURES_METADATA) == FEATURES_VERSION
            and cached.get(ZONE_LOOKUP_METADATA) == lookup_signature
            and shares_signature is not None
            and cached.get(ZONE_SHARES_FILE_METADATA) == shares_signature
            and all(field in source_schema for field in LOCATION_FIELDS)
            and not set(LOCATION_IDS).intersection(source_schema.names)
        )
        if (
            reuse_existing
            and current_locations
            and interval_path.is_file()
            and cached.get(INTERVAL_FILE_METADATA) == io_part.file_signature(interval_path)
            and training_path.is_file()
            and cached.get(TRAINING_FILE_METADATA) == io_part.file_signature(training_path)
            and all(field in source_schema for field in FEATURE_FIELDS)
        ):
            print(f"Reusing engineered features and training data: {working_path}")
            return working_path

    zones = _load_zone_lookup(lookup_data)
    if stale_standardization or (ZONE_LOOKUP_METADATA in cached and not current_locations):
        standardize_parquet(merged_path, working_path)
        original_signature = io_part.file_signature(working_path)

    with pq.ParquetFile(working_path) as source, pq.ParquetFile(merged_path) as merged:
        source_schema = source.schema_arrow
        source_metadata = source.metadata.metadata or {}
        if (
            source_metadata.get(MERGED_INPUT_METADATA) != merged_signature
            or source.metadata.num_rows != merged.metadata.num_rows
        ):
            raise RuntimeError("The working Parquet and raw merge changed out of sync. Run the pipeline again.")

        intervals, validation_start = _build_interval_table(merged)
        first_second = intervals[INTERVAL_START][0].value // 1000 if intervals.num_rows else 0
        counts = [intervals[name].to_numpy() for name in EVENT_COUNTS.values()]
        location_names = [field.name for field in LOCATION_FIELDS]
        replaced_names = {*LOCATION_IDS, *location_names, *(field.name for field in FEATURE_FIELDS)}
        original_fields = [field for field in source_schema if field.name not in replaced_names]
        original_names = [field.name for field in original_fields]
        location_inputs = (
            location_names if current_locations else [name for name in LOCATION_IDS if name in source_schema.names]
        )
        output_schema = pa.schema([*original_fields, *LOCATION_FIELDS, *FEATURE_FIELDS], metadata=source_schema.metadata)

        split_metadata = {}
        if validation_start is not None:
            year, month = divmod(validation_start - 1, 12)
            split_metadata[VALIDATION_START_METADATA] = f"{year:04d}-{month + 1:02d}".encode("ascii")
        training_schema = output_schema.with_metadata(split_metadata)
        training_count = 0
        with io_part.write_parquet(working_path, output_schema) as writer:
            with ExitStack() as artifacts:
                if not current_locations:
                    shares = _build_zone_shares(source, merged, zones, validation_start)
                    staged_csv = artifacts.enter_context(io_part.atomic_output_path(zone_shares_path))
                    arrow_csv.write_csv(shares, staged_csv)
                    zones = _attach_zone_shares(zones, staged_csv)
                training_writer = artifacts.enter_context(io_part.write_parquet(training_path, training_schema))
                interval_writer = artifacts.enter_context(io_part.write_parquet(interval_path, intervals.schema))
                for batch, raw in _paired_batches(
                    source, merged, original_names + location_inputs, [REQUEST_TIMESTAMP, PICKUP_TIMESTAMP]
                ):
                    request = raw_timestamps(raw, REQUEST_TIMESTAMP)
                    features = _feature_arrays(request, first_second, counts)
                    locations = batch.select(location_names).columns if current_locations else _location_arrays(batch, zones)
                    original = batch.select(original_names)
                    augmented = pa.RecordBatch.from_arrays([*original.columns, *locations, *features], schema=output_schema)
                    writer.write_batch(augmented)
                    if validation_start is not None:
                        training = augmented.filter(_before_validation(raw_timestamps(raw, PICKUP_TIMESTAMP), validation_start))
                        if training.num_rows:
                            training_writer.write_batch(training)
                            training_count += training.num_rows

                # Windows requires the input file to be closed before the atomic replacement.
                source.close()
                if original_signature != io_part.file_signature(working_path):
                    raise RuntimeError(f"{working_path} changed during feature engineering. Run the step again.")
                if merged_signature != io_part.file_signature(merged_path):
                    raise RuntimeError(f"{merged_path} changed during feature engineering. Run the step again.")
                if zone_lookup_path.read_bytes() != lookup_data:
                    raise RuntimeError(f"{zone_lookup_path} changed during feature engineering. Run the step again.")
                if current_locations and shares_signature != io_part.file_signature(zone_shares_path):
                    raise RuntimeError(f"{zone_shares_path} changed during feature engineering. Run the step again.")
                interval_writer.write_table(intervals)
            writer.add_key_value_metadata({
                **split_metadata,
                FEATURES_METADATA: FEATURES_VERSION,
                INTERVAL_FILE_METADATA: io_part.file_signature(interval_path),
                ZONE_LOOKUP_METADATA: lookup_signature,
                ZONE_SHARES_FILE_METADATA: io_part.file_signature(zone_shares_path),
                TRAINING_FILE_METADATA: io_part.file_signature(training_path),
            })

    print(f"Saved {intervals.num_rows:,} trip-count intervals to {interval_path}")
    print(f"Saved {training_count:,} training trips to {training_path}")
    print(f"Zone trip shares: {zone_shares_path}")
    print(f"Added taxi-zone and trip-count features to {working_path}")
    return working_path


if __name__ == "__main__":
    add_features(reuse_existing=config.REUSE_FEATURE_ENGINEERING, merged_path=config.MERGED_PARQUET)
