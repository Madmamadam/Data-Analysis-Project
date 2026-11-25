# Version 2.0
from pathlib import Path

import pyarrow as pa

import config
import io_part

INPUT_METADATA_PREFIX = b"tlc_merge_input:"
INPUT_ORDER_METADATA = b"tlc_merge_input_order"


def _input_key(name: str) -> bytes:
    return INPUT_METADATA_PREFIX + name.encode("utf-8")


def _inputs_match(
    sources: dict[str, Path], merged_path: Path, metadata: dict[bytes, bytes], input_order: bytes
) -> bool:
    if metadata.get(INPUT_ORDER_METADATA, input_order) != input_order:
        return False
    merged_time = merged_path.stat().st_mtime_ns
    for name, source in sources.items():
        if not source.is_file():
            continue  # A cached merge can be used when the original CSVs are unavailable.
        signature = metadata.get(_input_key(name))
        if signature is None:
            # Older merges have no source signatures. Check the available CSV timestamps.
            if source.stat().st_mtime_ns > merged_time:
                return False
        elif signature != io_part.file_signature(source):
            return False
    return True


def merge_csvs(raw_dir: Path, merged_path: Path, reuse_existing: bool) -> Path:
    """Write the original CSV values to one Parquet, or reuse the existing merge."""
    sources = {name: raw_dir / name for name in config.INPUT_FILES}
    input_order = b"\0".join(name.encode("utf-8") for name in sources)
    if reuse_existing:
        metadata = io_part.cached_parquet_metadata(merged_path)
        if metadata is not None and _inputs_match(sources, merged_path, metadata, input_order):
            print(f"Reusing merged Parquet: {merged_path}")
            return merged_path

    signatures = {_input_key(name): io_part.file_signature(path) for name, path in sources.items()}
    headers = {name: io_part.read_csv_header(path) for name, path in sources.items()}
    # Use the 2025 column order first and append only new column names.
    columns = headers[config.INPUT_2025] + headers[config.INPUT_2024] + headers[config.INPUT_2026]
    columns = list(dict.fromkeys(columns))
    schema = pa.schema([(name, pa.string()) for name in columns])

    rows_from_2024 = 0
    with io_part.write_parquet(merged_path, schema) as writer:
        for name, path in sources.items():
            print(f"Reading {name}", flush=True)
            with io_part.read_csv_chunks(path, config.CHUNK_ROWS) as chunks:
                for chunk in chunks:
                    aligned = chunk.reindex(columns=columns, fill_value="")
                    table = pa.Table.from_pandas(
                        aligned, schema=schema, preserve_index=False
                    )
                    writer.write_table(table)
                    if name == config.INPUT_2024:
                        rows_from_2024 += len(chunk)
        # The 2024 rows come first. Record their count in the footer for standardization.
        if any(signatures[_input_key(name)] != io_part.file_signature(path) for name, path in sources.items()):
            raise RuntimeError("A source CSV changed during the merge. Run the pipeline again.")
        writer.add_key_value_metadata({
            **signatures,
            INPUT_ORDER_METADATA: input_order,
            config.ROWS_2024_METADATA: str(rows_from_2024),
        })
    print(f"Merged Parquet saved: {merged_path}")
    return merged_path
