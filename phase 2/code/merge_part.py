# Version 2.0
from pathlib import Path


import pyarrow as pa

import phase2_io

def merge_csvs(raw_dir: Path, merged_path: Path, reuse_existing: bool) -> Path:
    """Write the original CSV values to one Parquet, or reuse the existing merge."""
    if reuse_existing and merged_path.is_file():
        print(f"Reusing merged Parquet: {merged_path}")
        return merged_path

    headers = {name: phase2_io.read_csv_header(raw_dir / name) for name in INPUT_FILES}
    # Use the 2025 column order first and append only new column names.
    columns = headers[INPUT_2025] + headers[INPUT_2024] + headers[INPUT_2026]
    columns = list(dict.fromkeys(columns))
    schema = pa.schema([(name, pa.string()) for name in columns])

    rows_from_2024 = 0
    with phase2_io.write_parquet(merged_path, schema) as writer:
        for name in INPUT_FILES:
            print(f"Reading {name}", flush=True)
            with phase2_io.read_csv_chunks(raw_dir / name, CHUNK_ROWS) as chunks:
                for chunk in chunks:
                    aligned = chunk.reindex(columns=columns, fill_value="")
                    table = pa.Table.from_pandas(
                        aligned, schema=schema, preserve_index=False
                    )
                    writer.write_table(table)
                    if name == INPUT_2024:
                        rows_from_2024 += len(chunk)
        # The 2024 rows come first. Record their count in the footer for standardization.
        writer.add_key_value_metadata({ROWS_2024_METADATA: str(rows_from_2024)})
    print(f"Merged Parquet saved: {merged_path}")
    return merged_path
