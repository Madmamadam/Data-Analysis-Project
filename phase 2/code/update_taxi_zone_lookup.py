"""Build and verify coordinates and airport flags for the NYC TLC taxi zones."""

import argparse
import csv
import json
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile

import shapefile
from pyproj import CRS, Transformer
from shapely.geometry import Point, shape

from config import TAXI_ZONE_LOOKUP


BOUNDARY_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
BOUNDARY_ARCHIVE = TAXI_ZONE_LOOKUP.with_name("taxi_zones.zip")
AIRPORT_ZONES = TAXI_ZONE_LOOKUP.with_name("taxi_airport_zones.json")
LOOKUP_COLUMNS = ("LocationID", "Borough", "Zone", "service_zone")
COORDINATE_COLUMNS = ("latitude", "longitude")
OUTPUT_COLUMNS = LOOKUP_COLUMNS + COORDINATE_COLUMNS + ("is_airport",)
LOOKUP_IDS = range(1, 266)
MAPPED_IDS = range(1, 264)


def read_lookup(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) not in (LOOKUP_COLUMNS, LOOKUP_COLUMNS + COORDINATE_COLUMNS, OUTPUT_COLUMNS):
            raise ValueError(f"Unexpected taxi-zone columns in {path}")
        rows = list(reader)

    if any(None in row or None in row.values() for row in rows):
        raise ValueError(f"A row in {path} has the wrong number of fields")
    if [int(row["LocationID"]) for row in rows] != list(LOOKUP_IDS):
        raise ValueError(f"Expected one row for each LocationID from 1 to 265 in {path}")
    return rows


def derive_airport_flags(path: Path, rows: list[dict[str, str]]) -> dict[int, int]:
    with path.open(encoding="utf-8") as stream:
        entries = json.load(stream)
    if not isinstance(entries, list):
        raise ValueError(f"Expected a list of airport zones in {path}")

    airports = set()
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or type(entry.get("LocationID")) is not int
            or not isinstance(entry.get("Zone"), str)
        ):
            raise ValueError(f"Expected an integer LocationID and a zone name in {path}")
        airports.add((entry["LocationID"], entry["Zone"]))
    if len(airports) != len(entries):
        raise ValueError(f"Duplicate airport entries in {path}")

    expected = {
        (int(row["LocationID"]), row["Zone"])
        for row in rows
        if row["service_zone"] in {"Airports", "EWR"}
    }
    if airports != expected:
        raise ValueError(
            f"Airport zones in {path} do not match TLC lookup rows. "
            f"Missing {sorted(expected - airports)}. Unexpected {sorted(airports - expected)}."
        )

    flags = {}
    for row in rows:
        location_id = int(row["LocationID"])
        flags[location_id] = int((location_id, row["Zone"]) in airports)
    return flags


def derive_coordinates(archive: bytes, rows: list[dict[str, str]]) -> dict[int, tuple[float, float]]:
    lookup = {int(row["LocationID"]): row for row in rows}
    coordinates = {}
    with ZipFile(BytesIO(archive)) as source:
        crs = CRS.from_wkt(source.read("taxi_zones/taxi_zones.prj").decode("utf-8"))
        to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        to_source = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        with shapefile.Reader(
            shp=BytesIO(source.read("taxi_zones/taxi_zones.shp")),
            dbf=BytesIO(source.read("taxi_zones/taxi_zones.dbf")),
        ) as boundaries:
            for feature in boundaries.iterShapeRecords():
                location_id = feature.record.LocationID
                if location_id not in MAPPED_IDS or location_id in coordinates:
                    raise ValueError(f"Unexpected or duplicate boundary LocationID {location_id}")
                row = lookup[location_id]
                if (row["Zone"], row["Borough"]) != (feature.record.zone, feature.record.borough):
                    raise ValueError(f"Zone name or borough mismatch for LocationID {location_id}")

                geometry = shape(feature.shape.__geo_interface__)
                if geometry.is_empty or not geometry.is_valid or geometry.area <= 0:
                    raise ValueError(f"Invalid boundary for LocationID {location_id}")
                point = geometry.centroid
                if not geometry.contains(point):
                    point = geometry.representative_point()

                longitude, latitude = to_wgs84.transform(point.x, point.y, errcheck=True)
                latitude, longitude = round(latitude, 7), round(longitude, 7)
                if not (40.45 < latitude < 40.95 and -74.30 < longitude < -73.65):
                    raise ValueError(f"Coordinates outside the NYC and Newark area for LocationID {location_id}")
                projected = to_source.transform(longitude, latitude, errcheck=True)
                if not geometry.contains(Point(projected)):
                    raise ValueError(f"Rounded coordinates outside boundary for LocationID {location_id}")
                coordinates[location_id] = latitude, longitude

    missing = sorted(set(MAPPED_IDS) - coordinates.keys())
    if missing:
        raise ValueError(f"Missing TLC boundaries for LocationIDs {missing}")
    return coordinates


def check_lookup(
    rows: list[dict[str, str]], coordinates: dict[int, tuple[float, float]], airport_flags: dict[int, int]
) -> None:
    for row in rows:
        location_id = int(row["LocationID"])
        if row.get("is_airport") != str(airport_flags[location_id]):
            raise ValueError(f"Invalid or missing is_airport for LocationID {location_id}")
        try:
            stored = tuple(row[column] for column in COORDINATE_COLUMNS)
            expected = coordinates.get(location_id)
            if expected is None and stored == ("", ""):
                continue
            if expected is not None and tuple(map(float, stored)) == expected:
                continue
        except (KeyError, ValueError):
            pass
        raise ValueError(f"Invalid or missing coordinates for LocationID {location_id}")


def write_lookup(
    path: Path, rows: list[dict[str, str]], coordinates: dict[int, tuple[float, float]], airport_flags: dict[int, int]
) -> None:
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as stream:
        temporary_path = Path(stream.name)
        try:
            writer = csv.writer(stream, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(OUTPUT_COLUMNS)
            for row in rows:
                location_id = int(row["LocationID"])
                latitude, longitude = coordinates.get(location_id, ("", ""))
                writer.writerow((
                    location_id, row["Borough"], row["Zone"], row["service_zone"],
                    latitude, longitude, airport_flags[location_id],
                ))
            stream.close()
            temporary_path.replace(path)
        finally:
            stream.close()
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Verify the lookup against the bundled TLC boundaries.")
    mode.add_argument("--check-live", action="store_true", help="Verify against the current TLC download endpoint.")
    args = parser.parse_args()

    rows = read_lookup(TAXI_ZONE_LOOKUP)
    airport_flags = derive_airport_flags(AIRPORT_ZONES, rows)
    if args.check_live:
        with urlopen(BOUNDARY_URL, timeout=60) as response:
            archive = response.read()
    else:
        archive = BOUNDARY_ARCHIVE.read_bytes()
    coordinates = derive_coordinates(archive, rows)

    if args.check or args.check_live:
        check_lookup(rows, coordinates, airport_flags)
    else:
        write_lookup(TAXI_ZONE_LOOKUP, rows, coordinates, airport_flags)
        check_lookup(read_lookup(TAXI_ZONE_LOOKUP), coordinates, airport_flags)
    print(f"Verified {len(coordinates)} points inside their TLC boundaries. IDs 264 and 265 have blank coordinates.")
    print(f"Verified is_airport for all {len(airport_flags)} lookup rows. {sum(airport_flags.values())} airport zones.")


if __name__ == "__main__":
    main()
