# Multimodal Fusion (Genomics + Imaging) with Transformers

An end-to-end pipeline that:
- prepares a paired imaging+genomics dataset,
- trains four models (image-only CNN, genomics-only Transformer, early fusion, intermediate fusion),
- evaluates them on a validation split,
- saves interpretable charts and metrics.

## Quickstart
```bash
pip install -r requirements.txt
python src/run_demo.py

## Outputs

Charts: artifacts/loss_curves.png, artifacts/roc_validation.png, artifacts/embedding_pca.png, artifacts/sample_images.png

Metrics: artifacts/metrics.json

Validation predictions: artifacts/val_predictions.csv