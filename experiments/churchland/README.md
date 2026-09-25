# Churchland results — sections 4.2 and 6.4.2

[analysis.ipynb](analysis.ipynb) regenerates Figure 4, Tables 1/S2, the additional
feature/ROC/traceback figures S12–S14, and the common-H100 timing figure S15.
Run all cells, or use `uv run --no-sync python -m experiments.churchland.paper_results`.
Outputs go to `outputs/paper/`; the manuscript is never overwritten.

## Inputs and numerical checks

- Prepared counts, metadata, and whole-trial partitions: `data/nitschke_20090812/`.
- Frozen selections, all candidate metrics, empirical ROC vertices, and encoder-attribution
  maps: `outputs/s10_endpoint_revision_v4_nice_cutoffs/`.
- Neural representations/checkpoints and shared endpoint indices:
  `/ceph/aeon/aeon/nldisco/churchland/sweeps/20260914_s10_endpoint_v1/`.
  Exact selected representation paths are recorded in `frozen_spec.json`.
- Current decoding thresholds and test predictions:
  `../outputs/decoder_balanced_accuracy_20260923/churchland/`.
  Some fitted probes remain under `outputs/s10_endpoint_revision_v3_192/` because
  those are still the fits used by the current paper, not obsolete duplicates.
- Completed timing experiment: `outputs/training_time_h100_3649999/`.

The notebook recalculates selected latents' coverage, specificity, selectivity and
AUROC from neural activations and behavioral labels, and verifies decoder balanced
accuracy from saved test probabilities with the fixed validation-selected thresholds.
Figure 4 and both tables use **measured scores**. The `paper_inputs/` copies of the
current tables are references for reporting differences only; they never replace
measured metrics in regenerated results. All table maxima are bolded automatically.

Feature order: recent deceleration; fast leftward target reach; early leftward
movement; late, target-specific reach. Main-text features use the illustrative
NLDisco latent; supplementary features use the best-Sel and best-AUROC NLDisco
latents. Baseline selectors are retained. Traceback uses saved gradient-times-input
maps with the corresponding decoder weights, with the same unit/time axes and colormap.

Representation fitting and feature discovery are transductive, on the full recording.
Decoder labels use whole-trial 80/10/10 splits. These are exploratory within-recording
comparisons, not an independent discovery test or the official NLB benchmark.

## Upstream code

`scripts/prepare.py` and `data.py` prepare the Churchland recording. The retained
`sweeps/s10_endpoint_v1/` modules implement window indexing, training and export
(`bootstrap.py`, `nldisco.py`, `cebra.py`, `cebra_192_check.py`, `langevinflow.py`),
dense-coordinate selection, current labels, logistic probes, and encoder attribution.
Run each training script's `--help` and use fresh output directories to avoid altering
frozen fits. `scripts/baselines.py` and `scripts/train_nldisco.py` now contain only the
shared neural-input helpers used by these windowed runners.

To recompute the encoder attributions from the retained checkpoint:

```bash
uv run --no-sync python -m experiments.churchland.sweeps.s10_endpoint_v1.attribution_traceback
```

This writes experiment artifacts only. The default report notebook renders the saved
maps and does not need a GPU. Training-time regeneration likewise reads the fifteen
completed measurements without retraining; see [TRAINING_TIME_BENCHMARK.md](TRAINING_TIME_BENCHMARK.md).

PCA/NMF, candidate-count sensitivity, extra-seed reports, first-pass notebooks,
provisional figures and incomplete timing runs are retired and removed.
