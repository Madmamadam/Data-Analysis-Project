# Version 2.0
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

import config
import phase2_io

CONGESTION_FEE = "cbd_congestion_fee"
CONVERSION_OK = "unit_conversion_ok"
STANDARDIZED_METADATA = b"tlc_unit_conversion"
LICENSE_COLUMN = "hvfhs_license_num"
TLC_LICENSES = ("HV0002", "HV0003", "HV0004", "HV0005")
TIMESTAMP_COLUMNS = {
    "request_datetime",
    "on_scene_datetime",
    "pickup_datetime",
    "dropoff_datetime",
}
TIMESTAMP_PARTS = ("year", "month", "day", "hour", "minute", "second")
TLC_TIMESTAMP = (
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]{1,9})?"
)
FLAG_COLUMNS = {
    "shared_request_flag",
    "shared_match_flag",
    "access_a_ride_flag",
    "wav_request_flag",
    "wav_match_flag",
}
INTEGER_COLUMNS = {
    "PULocationID": pa.int32(),
    "DOLocationID": pa.int32(),
    "_sample_source_row": pa.int64(),
}
MEASUREMENT_COLUMNS = {
    "trip_miles": ("trip_km", 1.609344),
    "trip_time": ("trip_time_minutes", 1 / 60),
}
for name in (
    "base_passenger_fare",
    "tolls",
    "bcf",
    "sales_tax",
    "congestion_surcharge",
    "airport_fee",
    "tips",
    "driver_pay",
    CONGESTION_FEE,
    "_sample_weight",
):
    MEASUREMENT_COLUMNS[name] = (name, 1)

CONVERTED_COLUMNS = TIMESTAMP_COLUMNS.union(
    FLAG_COLUMNS,
    INTEGER_COLUMNS.keys(),
    MEASUREMENT_COLUMNS.keys(),
    {LICENSE_COLUMN},
)

def _parse_integer(values: pd.Series, arrow_type: pa.DataType) -> pd.Series:
    # Extract the signed digits from a whole number such as 42 or 42.0.
    integer_text = values.str.extract(r"^([+-]?[0-9]+)(?:\.0+)?\Z", expand=False)
    numbers = pd.to_numeric(integer_text, errors="coerce", dtype_backend="numpy_nullable")
    limits = np.iinfo(arrow_type.to_pandas_dtype())
    if pd.api.types.is_integer_dtype(numbers.dtype):
        return numbers.where(numbers.ge(limits.min) & numbers.le(limits.max))

    # A mix of signed integers and overflows can make pandas use an inexact float dtype.
    def parse_exact(text):
        if pd.isna(text):
            return None
        try:
            number = int(text)
        except ValueError:
            return None
        return number if limits.min <= number <= limits.max else None

    return pd.Series(map(parse_exact, integer_text), index=values.index, dtype="Int64")


def _standardize_batch(batch: pa.RecordBatch, rows_from_2024: int = 0) -> pa.RecordBatch:
    """Convert known fields and flag rows where a nonempty value failed conversion."""
    names = []
    arrays = []
    conversion_ok = np.ones(batch.num_rows, dtype=bool)

    columns = list(zip(batch.schema.names, batch.columns))
    if CONGESTION_FEE not in batch.schema.names:
        empty_fee = pa.nulls(batch.num_rows, type=pa.string())
        columns.append((CONGESTION_FEE, empty_fee))

    for name, original in columns:
        if name not in CONVERTED_COLUMNS:
            names.append(name)
            arrays.append(original)
            continue

        text = pd.Series(original.to_pandas(), dtype="string").str.strip()
        if name == CONGESTION_FEE and rows_from_2024:
            text.iloc[:rows_from_2024] = "-1"
        values = text.mask(text.eq(""))
        missing = values.isna().to_numpy()

        if name == LICENSE_COLUMN:
            conversion_ok &= missing | values.isin(TLC_LICENSES).to_numpy()
            for license_code in TLC_LICENSES:
                indicator = values.eq(license_code).fillna(False).to_numpy(dtype=np.uint8)
                names.append(f"{LICENSE_COLUMN}_{license_code}")
                arrays.append(pa.array(indicator, type=pa.uint8()))
            continue

        output_name = name
        if name in TIMESTAMP_COLUMNS:
            local_time = values.where(values.str.fullmatch(TLC_TIMESTAMP, na=False))
            converted = pd.to_datetime(local_time, format="ISO8601", errors="coerce")
            output_type = pa.timestamp("ns")
        elif name in FLAG_COLUMNS:
            converted = values.map({"Y": 1, "N": 0})
            output_type = pa.uint8()
        elif name in INTEGER_COLUMNS:
            output_type = INTEGER_COLUMNS[name]
            converted = _parse_integer(values, output_type)
        else:
            output_name, multiplier = MEASUREMENT_COLUMNS[name]
            numbers = pd.to_numeric(values, errors="coerce").astype("float64")
            if multiplier != 1:
                numbers *= multiplier
            converted = numbers.where(np.isfinite(numbers))
            output_type = pa.float64()

        conversion_ok &= missing | converted.notna().to_numpy()
        names.append(output_name)
        arrays.append(pa.array(converted, type=output_type, from_pandas=True))

        if name in TIMESTAMP_COLUMNS:
            for part in TIMESTAMP_PARTS:
                part_type = pa.int16() if part == "year" else pa.int8()
                names.append(f"{name}_{part}")
                arrays.append(pa.array(getattr(converted.dt, part), type=part_type, from_pandas=True))

    names.append(CONVERSION_OK)
    arrays.append(pa.array(conversion_ok.astype(np.uint8), type=pa.uint8()))
    return pa.RecordBatch.from_arrays(arrays, names=names)


def standardize_parquet(merged_path: Path, working_path: Path) -> Path:
    """Read the raw merge in batches and write a separate standardized Parquet."""
    if merged_path == working_path:
        raise ValueError("Choose different paths for the merged and standardized Parquet.")

    with phase2_io.read_parquet_batches(merged_path, config.CHUNK_ROWS) as (raw_schema, footer, batches):
        metadata = dict(raw_schema.metadata or {})
        if STANDARDIZED_METADATA in metadata:
            raise ValueError(f"Expected a raw merge, but this file is already standardized: {merged_path}")
        rows_left_from_2024 = int(footer.get(config.ROWS_2024_METADATA, b"0"))

        empty_columns = [pa.array([], type=field.type) for field in raw_schema]
        empty_batch = pa.RecordBatch.from_arrays(empty_columns, schema=raw_schema)
        schema = _standardize_batch(empty_batch).schema
        if len(schema.names) != len(set(schema.names)):
            raise ValueError("A source column conflicts with a standardized column name.")
        metadata.pop(config.ROWS_2024_METADATA, None)
        metadata[STANDARDIZED_METADATA] = b"1"
        schema = schema.with_metadata(metadata)

        with phase2_io.write_parquet(working_path, schema) as writer:
            for batch in batches:
                rows_from_2024 = min(rows_left_from_2024, batch.num_rows)
                writer.write_batch(_standardize_batch(batch, rows_from_2024))
                rows_left_from_2024 -= rows_from_2024
    print(f"Standardized Parquet saved: {working_path}")
    return working_path
