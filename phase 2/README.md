# Phase 2 trip embeddings and clustering

This project groups New York City for-hire trips with similar booking conditions. It prepares trips from all providers between October 2024 and September 2026 and learns a compact representation for each trip. The four most recent pickup months are held out during encoder training.

## Run the pipeline

Run from this folder with Python 3.12.

```powershell
python -m pip install -r .\code\requirements\requirements.txt
python .\code\main.py
```

To move the pipeline, copy `code` with its `requirements` and `raw_data` or `raw_datas` subfolders. Run `main.py` by its full path from any directory. `code/config.py` lists the three CSV filenames.

## Train the encoder

```powershell
python -m pip install -r .\code\requirements\clustering_encoder_requirements.txt
python .\code\main.py --embed
```

Run `python .\code\embedding_step.py` to train from prepared trips. `EmbeddingSettings` in `code/config.py` defaults to CPU, a 240-second limit, and `d → 32 → 16 → 6 → 16 → 32 → d`. The encoder and both six-column embedding Parquets are saved in `code/temp_file/embedding`.

## Cluster trips

Pass `--embed --cluster` to train the encoder and cluster its outputs in one run. To cluster saved embeddings, use the base requirements and run the following command.

```powershell
python .\code\clustring_step.py --embeddings --training .\code\temp_file\embedding\embeddings_training.parquet --full .\code\temp_file\embedding\embeddings.parquet
```

MiniBatchKMeans selects K by mean silhouette score and refits on the full dataset in batches. It saves the model, assignments, and plot under `code/temp_file/clustering`. To use an existing TorchScript encoder, install `clustering_encoder_requirements.txt`, place the model at `code/requirements/encoder.pt`, and run `main.py --cluster`.

## Prepared data

In `code/temp_file`, `merged.parquet` preserves the source fields. `working.parquet` adds request-time cycles, signed intervals, taxi-zone features, and traffic counts. `training.parquet` covers pickups before June 1, 2026. The two preembedding Parquets contain ordered `float32` inputs defined by `code/requirements/preembedding_rules.json`, with scales fitted on the training split. See `AGENT_GUIDE.txt` for feature and output details.
