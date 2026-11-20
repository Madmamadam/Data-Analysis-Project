"""Build and standardize the phase 2 working Parquet."""

import argparse
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

import phase2_io

save_file = phase2_io.save_file


# Configuration
SAVE_WORKING_CSV = False  # Set to True to export the full working table to output.
CHUNK_ROWS = 100_000
PREVIEW_ROWS = 1_000

CODE_DIR = Path(__file__).resolve().parent
RAW_DIR = CODE_DIR / "raw_datas"
TEMP_DIR = CODE_DIR / "temp_file"
OUTPUT_DIR = CODE_DIR / "output"
WORKING_PARQUET = TEMP_DIR / "working.parquet"

INPUT_2024 = "hvfhv_2024_oct-dec_2p5pct.csv"
INPUT_2025 = "hvfhv_2025_2p5pct.csv"
INPUT_2026 = "hvfhv_2026_jan-sep_2p5pct.csv"
INPUT_FILES = (INPUT_2024, INPUT_2025, INPUT_2026)
CONGESTION_FEE = "cbd_congestion_fee"
CONVERSION_OK = "unit_conversion_ok"
STANDARDIZED_METADATA = b"tlc_unit_conversion"
LICENSE_COLUMN = "hvfhs_license_num"
TLC_LICENSES = ("HV0002", "HV0003", "HV0004", "HV0005")
TIMESTAMP_COLUMNS = {
    "request_datetime", "on_scene_datetime", "pickup_datetime", "dropoff_datetime",
}
TIMESTAMP_PARTS = ("year", "month", "day", "hour", "minute", "second")
TLC_TIMESTAMP = r"[0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]{1,9})?"
FLAG_COLUMNS = {
    "shared_request_flag", "shared_match_flag", "access_a_ride_flag", "wav_request_flag", "wav_match_flag",
}
INTEGER_COLUMNS = {
    "PULocationID": pa.int32(), "DOLocationID": pa.int32(), "_sample_source_row": pa.int64(),
}
MEASUREMENT_COLUMNS = {
    "trip_miles": ("trip_km", 1.609344),
    "trip_time": ("trip_time_minutes", 1 / 60),
    **{
        name: (name, 1) for name in (
            "base_passenger_fare", "tolls", "bcf", "sales_tax", "congestion_surcharge",
            "airport_fee", "tips", "driver_pay", CONGESTION_FEE, "_sample_weight",
        )
    },
}
CONVERTED_COLUMNS = (
    TIMESTAMP_COLUMNS | FLAG_COLUMNS | INTEGER_COLUMNS.keys() | MEASUREMENT_COLUMNS.keys() | {LICENSE_COLUMN}
)


def build_working_parquet(
    raw_dir: Path = RAW_DIR,
    working_parquet: Path = WORKING_PARQUET,
    *,
    chunk_rows: int = CHUNK_ROWS,
) -> Path:
    """Merge the inputs in source order without accumulating chunks in memory."""
    headers = {name: phase2_io.read_csv_header(raw_dir / name) for name in INPUT_FILES}
    columns = list(dict.fromkeys([
        *headers[INPUT_2025], *headers[INPUT_2024], *headers[INPUT_2026], CONGESTION_FEE,
    ]))

    # Keep the original text values for the later cleaning and type checks.
    schema = pa.schema([(name, pa.string()) for name in columns])

    def merged_tables() -> Iterator[pa.Table]:
        for name in INPUT_FILES:
            print(f"Reading {name}", flush=True)
            with phase2_io.read_csv_chunks(raw_dir / name, chunk_rows) as chunks:
                for chunk in chunks:
                    if name == INPUT_2024:
                        chunk[CONGESTION_FEE] = "-1"
                    aligned = chunk.reindex(columns=columns, fill_value="")
                    yield pa.Table.from_pandas(aligned, schema=schema, preserve_index=False)

    with phase2_io.staged_path(working_parquet.parent) as pending:
        with closing(merged_tables()) as tables:
            phase2_io.write_parquet_batches(pending, schema, tables)
        phase2_io.replace_file(pending, working_parquet)
    return working_parquet


def _parse_integer(values: pd.Series, arrow_type: pa.DataType) -> pd.Series:
    # Accept an integer written as 42.0 without round-tripping it through a float.
    integer_text = values.where(values.str.fullmatch(r"[+-]?[0-9]+(?:\.0+)?", na=False))
    integer_text = integer_text.str.replace(r"\.0+$", "", regex=True)
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


def _standardize_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Convert known fields and flag rows where a nonempty value failed conversion."""
    names = []
    arrays = []
    conversion_ok = np.ones(batch.num_rows, dtype=bool)

    for name, original in zip(batch.schema.names, batch.columns):
        if name not in CONVERTED_COLUMNS:
            names.append(name)
            arrays.append(original)
            continue

        text = pd.Series(original.to_pandas(), dtype="string").str.strip()
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
            numbers = pd.to_numeric(values, errors="coerce").astype("float64") * multiplier
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


def standardize_working_parquet(working_parquet: Path, *, chunk_rows: int = CHUNK_ROWS) -> Path:
    """Apply the unit conversions once and atomically replace the working Parquet."""
    with phase2_io.staged_path(working_parquet.parent) as pending:
        with phase2_io.read_parquet_batches(working_parquet, chunk_rows) as (raw_schema, batches):
            metadata = raw_schema.metadata or {}
            if STANDARDIZED_METADATA in metadata:
                print("Working Parquet is already standardized.")
                return working_parquet

            empty = pa.RecordBatch.from_arrays(
                [pa.array([], type=field.type) for field in raw_schema], schema=raw_schema,
            )
            schema = _standardize_batch(empty).schema
            if len(schema.names) != len(set(schema.names)):
                raise ValueError("A source column conflicts with a standardized column name.")
            schema = schema.with_metadata({**metadata, STANDARDIZED_METADATA: b"1"})
            phase2_io.write_parquet_batches(
                pending, schema, (_standardize_batch(batch) for batch in batches),
            )
        phase2_io.replace_file(pending, working_parquet)
    return working_parquet


def export_working_csv(
    working_parquet: Path = WORKING_PARQUET,
    output_dir: Path = OUTPUT_DIR,
    *,
    preview: bool = False,
) -> Path:
    """Export all rows, or a small preview, without loading the whole table."""
    return phase2_io.export_working_csv(
        working_parquet,
        output_dir,
        preview=preview,
        preview_rows=PREVIEW_ROWS,
        chunk_rows=CHUNK_ROWS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preview", action="store_true",
        help="Export the first 1,000 rows of the existing working Parquet for Excel.",
    )
    mode.add_argument(
        "--standardize-existing", action="store_true",
        help="Standardize the existing working Parquet without reading the original CSV files.",
    )
    args = parser.parse_args()

    if args.preview:
        print(f"Preview saved: {export_working_csv(preview=True)}")
        return

    working_parquet = WORKING_PARQUET if args.standardize_existing else build_working_parquet()
    print("Standardizing units and field types.", flush=True)
    standardize_working_parquet(working_parquet)
    print(f"Working Parquet ready: {working_parquet}")
    if SAVE_WORKING_CSV:
        print(f"Full CSV saved: {export_working_csv(working_parquet)}")


if __name__ == "__main__":
    main()
