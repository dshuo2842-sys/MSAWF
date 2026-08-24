# Dataset Integration Notes

This directory intentionally contains no dataset or checkpoint. Convert externally obtained traces into `msawf.data.TraceRecord` values, generate immutable manifests before training, and keep raw/processed data in ignored local directories.

Use the files under `../configs/` as schema-valid path examples. Dataset-specific adapters must retain stable trace IDs, multi-hot labels, dataset fingerprints, source provenance, and constituent trace IDs where applicable. Run leakage validation before any trainer receives a support batch.
