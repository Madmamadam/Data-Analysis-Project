"""Build the phase 2 working CSV and save files on demand."""

import csv
import re
import shutil
from contextlib import contextmanager
from datetime import datetime
from itertools import count
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator

import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
RAW_DIR = CODE_DIR / "raw_datas"
TEMP_DIR = CODE_DIR / "temp_file"
OUTPUT_DIR = CODE_DIR / "output"
WORKING_CSV = TEMP_DIR / "working.csv"

INPUT_2024 = "hvfhv_2024_oct-dec_2p5pct.csv"
INPUT_2025 = "hvfhv_2025_2p5pct.csv"
INPUT_2026 = "hvfhv_2026_jan-sep_2p5pct.csv"
INPUT_FILES = (INPUT_2024, INPUT_2025, INPUT_2026)
CONGESTION_FEE = "cbd_congestion_fee"
CHUNK_ROWS = 100_000
WINDOWS_DEVICE_NAME = re.compile(
    r"CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|(?:COM|LPT)[1-9¹²³]", re.IGNORECASE,
)


@contextmanager
def _staged_path(directory: Path) -> Iterator[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as file:
        path = Path(file.name)

    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def _replace_temp_file(source: Path, target: Path) -> Path:
    with _staged_path(target.parent) as pending:
        shutil.copyfile(source, pending)
        pending.replace(target)
    return target


def _save_output_snapshot(source: Path, target: Path) -> Path:
    with _staged_path(target.parent) as pending:
        shutil.copyfile(source, pending)
        timestamp = datetime.now().strftime("%H-%M")
        for number in count(1):
            suffix = "" if number == 1 else f" ({number})"
            candidate = target.with_name(f"{target.stem}_{timestamp}{suffix}{target.suffix}")
            try:
                candidate.touch(exist_ok=False)
            except FileExistsError:
                continue

            try:
                pending.replace(candidate)
            except BaseException:
                candidate.unlink(missing_ok=True)
                raise
            return candidate


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


def _read_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        columns = next(csv.reader(stream), [])
    if not columns or "" in columns or len(columns) != len(set(columns)):
        raise ValueError(f"Missing, blank, or duplicate CSV headers in {path}")
    return columns


def build_working_csv(
    raw_dir: Path = RAW_DIR,
    working_csv: Path = WORKING_CSV,
    *,
    chunk_rows: int = CHUNK_ROWS,
) -> Path:
    """Merge the inputs in source order without accumulating chunks in memory."""
    headers = {name: _read_header(raw_dir / name) for name in INPUT_FILES}
    columns = list(dict.fromkeys([
        *headers[INPUT_2025], *headers[INPUT_2024], *headers[INPUT_2026], CONGESTION_FEE,
    ]))

    with _staged_path(working_csv.parent) as pending:
        with pending.open("w", encoding="utf-8", newline="") as output:
            csv.writer(output).writerow(columns)
            for name in INPUT_FILES:
                print(f"Reading {name}", flush=True)
                with pd.read_csv(
                    raw_dir / name,
                    encoding="utf-8-sig",
                    chunksize=chunk_rows,
                    dtype=str,
                    keep_default_na=False,
                ) as chunks:
                    for chunk in chunks:
                        if name == INPUT_2024:
                            chunk[CONGESTION_FEE] = "-1"
                        chunk.reindex(columns=columns, fill_value="").to_csv(
                            output, index=False, header=False, lineterminator="\r\n",
                        )
        pending.replace(working_csv)
    return working_csv


if __name__ == "__main__":
    print(f"Working CSV ready: {build_working_csv()}")
