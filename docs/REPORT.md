# Project Summary

## Objective
Integrate imaging and genomics to improve clinical prediction using transformer-based embeddings and compare fusion strategies (early, intermediate, late).

## Data
Paired inputs: 32×32 images and a 200-feature gene-expression matrix, split into train/validation.

## Models & Fusion
- Image-only: compact CNN (3 conv blocks + GAP + MLP).
- Genomics-only: lightweight Transformer encoder (genes→tokens→positional embeddings→mean-pool→MLP).
- Early fusion: average-pooled image (8×8) concatenated with gene features → MLP.
- Intermediate fusion: concatenate CNN and Transformer embeddings → MLP head.
- Late fusion (evaluation only): average of unimodal probabilities.

## Evaluation
- ROC–AUC on validation.
- Training loss curves.
- 2D visualization (PCA) of fused embeddings for interpretability.

## Conclusion
On this task, intermediate fusion typically achieves the strongest validation performance, aligning with the hypothesis that shared latent representations capture modality interactions better than unimodal models or naive late averaging.
