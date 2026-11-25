# Phase 2 trip data preparation and clustering

This project prepares New York City for-hire vehicle trips from October 2024 through September 2026. It combines trips from all providers and can group trips with similar booking conditions using a pretrained encoder. The four most recent pickup months are reserved during model selection.

## Run the pipeline

Run from this folder with Python 3.12.

```powershell
python -m pip install -r .\code\requirements\requirements.txt
python .\code\main.py
```

To move the pipeline, copy `code` with its `requirements` and `raw_data` or `raw_datas` subfolders. Run `main.py` by its full path from any directory. `code/config.py` lists the three CSV filenames.

Progress messages start with the system's local time in 24-hour format, such as `[14:05:09]`.

## Cluster trips

Supply a pretrained TorchScript encoder at `code/requirements/encoder.pt`. Its forward method takes the ordered preembedding features and returns one embedding vector per trip.

```powershell
python -m pip install -r .\code\requirements\clustering_encoder_requirements.txt
python .\code\main.py --cluster
```

Run `python .\code\clustring_step.py` to use existing preembedding Parquets. With saved encoder outputs, use the base requirements and pass two Parquets with matching ordered numeric columns.

```powershell
python .\code\clustring_step.py --embeddings --training path\to\training.parquet --full path\to\full.parquet
```

MiniBatchKMeans compares nine K values over three seeds using 500,000 fit embeddings and a fixed, disjoint set of 10,000 for silhouette scoring. It selects K by the mean score and refits in batches on the full dataset. Each run saves the model, row-indexed assignments, and plot under `code/temp_file/clustering`. Edit `ClusteringSettings` in `code/config.py` to change the defaults.

## Prepared data

In `code/temp_file`, `merged.parquet` preserves the source fields. `working.parquet` adds request-time cycles, signed intervals, taxi-zone features, and traffic counts. `training.parquet` covers pickups before June 1, 2026. The two preembedding Parquets contain ordered `float32` inputs defined by `code/requirements/preembedding_rules.json`, with scales fitted on the training split. See `AGENT_GUIDE.txt` for feature and output details.
