# Phase 2 trip data preparation

This project combines three samples of New York City for-hire vehicle trips into one dataset for analysis. The samples cover October 2024 through September 2026. The pipeline converts distances to kilometers and trip times to minutes.

## Run the pipeline

Use Python 3.12. From this folder, run the following commands in PowerShell.

```powershell
python -m pip install -r .\code\requirements\requirements.txt
python .\code\main.py
```

To move the pipeline, copy the `code` folder, including `raw_datas` and `requirements`. You can run `main.py` from any working directory using its full path. The three CSV filenames are listed in `code/config.py`.

`main.py` merges the CSVs in batches into `code/temp_file/merged.parquet`, then writes `code/temp_file/working.parquet` with typed columns and a row-level conversion flag. The standardized file omits the base numbers and sampling fields and represents each trip timestamp as separate year, month, day, hour, minute, and second columns. The pipeline reuses the raw merge by default. Set `REUSE_MERGED_PARQUET = False` in `code/config.py` after changing the source CSVs. See `AGENT_GUIDE.txt` for pipeline details.

Run `python .\code\utilityFunctions.py` after the pipeline to create the wait-time, monthly license-share, and Uber origin-destination plots in `code/output`. The OD charts use the bundled official TLC taxi-zone lookup.
