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

MiniBatchKMeans evaluates each K with three seeds using 500,000 fit embeddings and a fixed, disjoint 10,000-row scoring sample. It selects the highest mean silhouette score and refits on the full dataset in batches. `code/output/clustering_k_selection.png` shows all three scores, their mean, and the winning K. Each run also keeps its plot, model, and assignments under `code/temp_file/clustering`. To redraw the plot from saved scores, pass `--plot-from-assignments PATH` to `clustring_step.py` with the run's `cluster_assignments.parquet`.

To use an existing TorchScript encoder, install `clustering_encoder_requirements.txt`, place the model at `code/requirements/encoder.pt`, and run `main.py --cluster`.

## Plot training clusters

```powershell
python .\code\visualization_step.py
```

After clustering saved embeddings, this writes `code/output/training_clusters_pca.png` using the latest completed run. It fits two-component PCA on all training embeddings and colors a reproducible sample of up to 100,000 trips by saved cluster label. Use `--max-points 0` to draw every trip. Select another run with `--assignments PATH`. For clustering run directly on preembedding features, pass `--encoder PATH` with the same TorchScript model. The dependencies are in `code/requirements/requirements.txt`, with PyTorch in `clustering_encoder_requirements.txt` for `--encoder`.

## Prepared data

In `code/temp_file`, `merged.parquet` preserves the source fields. `working.parquet` adds request-time cycles, signed intervals, taxi-zone features, and traffic counts. `training.parquet` covers pickups before June 1, 2026. The two preembedding Parquets contain ordered `float32` inputs defined by `code/requirements/preembedding_rules.json`, with scales fitted on the training split. See `AGENT_GUIDE.txt` for feature and output details.
