# Phase 2 trip data preparation

This project prepares New York City for-hire vehicle trips from October 2024 through September 2026. It combines trips from all providers so they can be compared, with the most recent four pickup months reserved for validation.

## Run the pipeline

Run from this folder with Python 3.12.

```powershell
python -m pip install -r .\code\requirements\requirements.txt
python .\code\main.py
```

To move the pipeline, copy `code` with its `requirements` and `raw_data` or `raw_datas` subfolders. Run `main.py` by its full path from any directory. `code/config.py` lists the three CSV filenames.

## Outputs in `code/temp_file`

- `merged.parquet` preserves the source fields for chronological splits, traffic counts, and plots.
- `working.parquet` encodes request minute, hour, weekday, and month with sine/cosine pairs. `request_to_on_scene_minutes`, `request_to_pickup_minutes`, and `request_to_dropoff_minutes` each store the signed difference from the request as fractional minutes. A missing or invalid timestamp produces `NULL` in each dependent cell.
- `training.parquet` contains the same fields for pickups before June 1, 2026, the current cutoff.
- `trip_counts_15min.parquet` counts pickups and drop-offs by 15-minute interval. `zone_trip_shares.csv` gives each mapped zone's share of endpoints observed before the cutoff among training trips.

The working and training files use kilometers and minutes and include the `is Uber` and `unit_conversion_ok` fields. Taxi-zone features combine bundled coordinates and airport flags with computed endpoint shares. Missing or invalid zones have a validity flag of `0` and zero-filled location features.

Run `python .\code\utilityFunctions.py` to generate the plots in `code/output`. See `AGENT_GUIDE.txt` for the cache, feature, and taxi-zone lookup details.
