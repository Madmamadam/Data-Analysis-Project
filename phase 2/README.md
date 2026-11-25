# Phase 2 trip data preparation

This project combines three samples of New York City for-hire vehicle trips into one dataset for analysis. The samples cover October 2024 through September 2026. The pipeline converts distances to kilometers and trip times to minutes.

## Run the pipeline

Use Python 3.12. From this folder, run the following commands in PowerShell.

```powershell
python -m pip install -r .\code\requirements\requirements.txt
python .\code\main.py
```

To move the pipeline, copy the `code` folder, including `requirements` and your raw-data folder (`raw_data` or `raw_datas`). You can run `main.py` from any working directory using its full path. The three CSV filenames are listed in `code/config.py`.

`main.py` merges the CSVs in batches into `code/temp_file/merged.parquet`, then writes `code/temp_file/working.parquet` with typed columns and a row-level conversion flag. The standardized file omits the base numbers and sampling fields and represents each trip timestamp as separate year, month, day, hour, minute, and second columns. The pipeline reuses the raw merge by default. Set `REUSE_MERGED_PARQUET = False` in `code/config.py` after changing the source CSVs. See `AGENT_GUIDE.txt` for pipeline details.

Run `python .\code\utilityFunctions.py` after the pipeline to create the wait-time, monthly license-share, and Uber origin-destination plots in `code/output`.

## Taxi zone coordinates

`code/requirements/taxi_zone_lookup.csv` includes a WGS84 latitude and longitude inside each of the 263 mapped zones. Codes 264 (unknown) and 265 (outside NYC) have blank coordinates. The points come from the bundled taxi zone boundaries. [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page#taxi-zone-maps-and-lookup-tables)

`is_airport` is `1` for the three airport zones and `0` for all other rows. The updater matches `LocationID` and `Zone` against `code/requirements/taxi_airport_zones.json` and checks that the list covers TLC's `Airports` and `EWR` service zones.

Feature engineering replaces the pickup and drop-off `LocationID` columns in `working.parquet` with `latitude`, `longitude`, and `is_airport` fields prefixed by `pickup_` or `dropoff_`. A missing or unmatched ID gives three nulls. IDs 264 and 265 give null coordinates and a `0` airport flag.

```powershell
python -m pip install -r .\code\requirements\taxi_zone_coordinates_requirements.txt
python .\code\update_taxi_zone_lookup.py --check
```

Run the script without `--check` to regenerate from the bundled boundaries. Use `--check-live` to validate against the current TLC download.
