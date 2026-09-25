# Aeon Table S3 decoding

Single-latent and all-latent L2 logistic regression; test balanced accuracy.

Uses the existing chronological 80/10/10 split, fixed feature populations, and a training-only StandardScaler. Regularization maximizes validation AUROC over the existing seven-value C grid. Each fitted decoder gets one validation-balanced-accuracy threshold, frozen before test evaluation. No train/validation refit. The wheel single-latent fit and score are preserved exactly.

| Feature | Space | Decode | Test AUROC | C | Threshold |
|---|---|---:|---:|---:|---:|
| wheel | single | 0.889024892 | 0.926085312 | 0.001 | 0.034468715 |
| wheel | all | 0.932787501 | 0.968193795 | 0.001 | 0.121617255 |
| area | single | 0.798948546 | 0.859351245 | 0.001 | 0.223345450 |
| area | all | 0.793679279 | 0.899256268 | 0.001 | 0.171680254 |
| speed | single | 0.711750732 | 0.772143886 | 0.001 | 0.257095014 |
| speed | all | 0.794931287 | 0.920214688 | 0.001 | 0.509689570 |
| direction | single | 0.855049543 | 0.912629046 | 0.001 | 0.093082928 |
| direction | all | 0.900528724 | 0.956620473 | 0.01 | 0.136306932 |

The frozen representations and exploratory feature descriptions use the full recording. These scores measure decoding within that recording, not independent validation of feature discovery.

Reproduction: `uv run --no-sync python -m experiments.aeon.fit_appendix_decoders --output <new-directory>`.
