# Validation-selected balanced-accuracy decoding

These are the saved thresholds and test predictions behind the current Churchland
results and the original Aeon wheel decoder. Only current Churchland selections
and full-space probes remain; retired PCA/NMF and unused main-feature probes are removed.

Logistic L2 regularization is selected by validation AUROC. The probability threshold
maximizes validation balanced accuracy; exact ties select the highest threshold.
Equal probabilities stay together, with positive predictions defined by `p >= t`.
Test balanced accuracy is the mean of sensitivity and specificity. Churchland
intervals use 300 whole-test-trial bootstrap samples with fixed thresholds.

To recompute from the retained fitted models into a fresh experiment directory:

```bash
uv run --no-sync python -m experiments.evaluate_decoder_balanced_accuracy   --output experiments/outputs/recomputed_decoder_scores
```

The dataset report notebooks verify these scores and render the current figures/tables
into their own experiment output directories. No command here installs paper assets.
Aeon's newer eight-decoder results live in `aeon/outputs/appendix_decoders_20260925/`.
