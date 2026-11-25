"""Add taxi-zone, previous-interval traffic, and cyclic request-time features.

Run after standardization. The helper Parquet spans the first through last
pickup or drop-off interval in the standardized local clock. Its counts are
reused across all batches of the working Parquet rewrite.
"""

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
from standardize_part import TIMESTAMP_PARTS, standardize_parquet

INTERVAL_SECONDS = 900
INTERVAL_START = "interval_start_datetime"
REQUEST_TIMESTAMP = "request_datetime"
EVENT_COUNTS = {"pickup_datetime": "pickup_count", "dropoff_datetime": "dropoff_count"}
FEATURES_METADATA = b"tlc_feature_engineering"
FEATURES_VERSION = b"3"
INTERVAL_FILE_METADATA = b"tlc_feature_interval_file"
ZONE_LOOKUP_METADATA = b"tlc_feature_zone_lookup"
LOCATION_IDS = {"PULocationID": "pickup", "DOLocationID": "dropoff"}
ZONE_ATTRIBUTES = {"latitude": pa.float64(), "longitude": pa.float64(), "is_airport": pa.uint8()}
LOCATION_FIELDS = [
    pa.field(f"{endpoint}_{name}", data_type)
    for endpoint in LOCATION_IDS.values()
    for name, data_type in ZONE_ATTRIBUTES.items()
]
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


def _load_zone_lookup(data: bytes) -> pa.RecordBatch:
    types = {"LocationID": pa.int32(), **ZONE_ATTRIBUTES}
    table = arrow_csv.read_csv(
        pa.py_buffer(data),
        convert_options=arrow_csv.ConvertOptions(include_columns=list(types), column_types=types),
    )
    ids = table["LocationID"]
    if not table.num_rows or ids.null_count or pc.count_distinct(ids).as_py() != table.num_rows:
        raise ValueError("The taxi-zone lookup must contain unique, nonempty LocationIDs.")
    flags = table["is_airport"]
    if flags.null_count or not pc.all(pc.is_in(flags, value_set=pa.array([0, 1], type=flags.type))).as_py():
        raise ValueError("Every taxi-zone airport flag must be 0 or 1.")
    return table.combine_chunks().to_batches()[0]


def _location_arrays(batch: pa.RecordBatch, zones: pa.RecordBatch) -> list[pa.Array]:
    features = []
    for id_column in LOCATION_IDS:
        if id_column not in batch.schema.names:
            features.extend(pa.nulls(batch.num_rows, type=data_type) for data_type in ZONE_ATTRIBUTES.values())
            continue
        indexes = pc.index_in(batch.column(id_column), value_set=zones.column("LocationID"), skip_nulls=True)
        features.extend(pc.take(zones.column(name), indexes) for name in ZONE_ATTRIBUTES)
    return features


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


def _cycle_coordinates(period: int) -> np.ndarray:
    """Return sine and cosine rounded to three decimal places."""
    steps = np.arange(period)
    angles = 2 * np.pi * steps / period
    coordinates = np.column_stack((np.sin(angles), np.cos(angles)))

    # Identify quarter-turns using integer arithmetic instead of approximate angles.
    quarters, remainder = np.divmod(4 * steps, period)
    on_axis = remainder == 0
    axis_coordinates = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]])
    coordinates[on_axis] = axis_coordinates[quarters[on_axis]]
    return np.round(coordinates, decimals=3)


REQUEST_COORDINATES = {part: _cycle_coordinates(period) for part, (period, _) in REQUEST_CYCLES.items()}


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

    missing = ~valid
    for part, (_, offset) in REQUEST_CYCLES.items():
        values = getattr(request.dt, part).to_numpy()
        steps = np.where(valid, values - offset, 0).astype(np.intp)
        sine, cosine = REQUEST_COORDINATES[part][steps].T
        features.append(pa.array(sine, mask=missing))
        features.append(pa.array(cosine, mask=missing))
    return features


def add_features(
    working_path: Path = config.WORKING_PARQUET,
    interval_path: Path | None = None,
    *,
    reuse_existing: bool = False,
    merged_path: Path | None = None,
    zone_lookup_path: Path = config.TAXI_ZONE_LOOKUP,
) -> Path:
    """Reuse a finished feature stage or atomically add it to the working Parquet."""
    if interval_path is None:
        interval_path = working_path.with_name("trip_counts_15min.parquet")
    if interval_path.resolve() == working_path.resolve():
        raise ValueError("Choose different paths for the interval table and the working Parquet.")

    lookup_data = zone_lookup_path.read_bytes()
    lookup_signature = sha256(lookup_data).hexdigest().encode("ascii")
    original_signature = io_part.file_signature(working_path)
    with pq.ParquetFile(working_path) as source:
        source_schema = source.schema_arrow
        cached = source.metadata.metadata or {}
        current_locations = (
            cached.get(FEATURES_METADATA) == FEATURES_VERSION
            and cached.get(ZONE_LOOKUP_METADATA) == lookup_signature
            and all(field in source_schema for field in LOCATION_FIELDS)
            and not set(LOCATION_IDS).intersection(source_schema.names)
        )
        if (
            reuse_existing
            and current_locations
            and interval_path.is_file()
            and cached.get(INTERVAL_FILE_METADATA) == io_part.file_signature(interval_path)
            and all(field in source_schema for field in FEATURE_FIELDS)
        ):
            print(f"Reusing engineered features and interval table: {working_path}")
            return working_path

    zones = _load_zone_lookup(lookup_data)
    if ZONE_LOOKUP_METADATA in cached and not current_locations:
        if merged_path is None:
            raise ValueError("Pass merged_path to rebuild the location features from the original trip IDs.")
        standardize_parquet(merged_path, working_path)
        original_signature = io_part.file_signature(working_path)

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
        location_names = [field.name for field in LOCATION_FIELDS]
        replaced_names = {*LOCATION_IDS, *location_names, *(field.name for field in FEATURE_FIELDS)}
        original_fields = [field for field in source_schema if field.name not in replaced_names]
        original_names = [field.name for field in original_fields]
        location_inputs = (
            location_names if current_locations else [name for name in LOCATION_IDS if name in source_schema.names]
        )
        output_schema = pa.schema([*original_fields, *LOCATION_FIELDS, *FEATURE_FIELDS], metadata=source_schema.metadata)

        with io_part.write_parquet(working_path, output_schema) as writer:
            for batch in source.iter_batches(batch_size=config.CHUNK_ROWS, columns=original_names + location_inputs):
                request = _timestamps(batch, REQUEST_TIMESTAMP)
                features = _feature_arrays(request, first_second, counts)
                locations = batch.select(location_names).columns if current_locations else _location_arrays(batch, zones)
                original = batch.select(original_names)
                augmented = pa.RecordBatch.from_arrays([*original.columns, *locations, *features], schema=output_schema)
                writer.write_batch(augmented)

            # Windows requires the input file to be closed before the atomic replacement.
            source.close()
            if original_signature != io_part.file_signature(working_path):
                raise RuntimeError(f"{working_path} changed during feature engineering. Run the step again.")
            if zone_lookup_path.read_bytes() != lookup_data:
                raise RuntimeError(f"{zone_lookup_path} changed during feature engineering. Run the step again.")
            with io_part.write_parquet(interval_path, intervals.schema) as interval_writer:
                interval_writer.write_table(intervals)
            writer.add_key_value_metadata({
                FEATURES_METADATA: FEATURES_VERSION,
                INTERVAL_FILE_METADATA: io_part.file_signature(interval_path),
                ZONE_LOOKUP_METADATA: lookup_signature,
            })

    print(f"Saved {intervals.num_rows:,} trip-count intervals to {interval_path}")
    print(f"Added taxi-zone, trip-count, and request-time features to {working_path}")
    return working_path


if __name__ == "__main__":
    add_features(reuse_existing=config.REUSE_FEATURE_ENGINEERING, merged_path=config.MERGED_PARQUET)
