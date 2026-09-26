# Aeon results — sections 4.3 and 6.4.3

[analysis.ipynb](analysis.ipynb) reproduces Figure 5, Table S3 and Figures S16–S20.
Run all cells, or `uv run --no-sync python -m experiments.aeon.paper_results`.
Figures, tables and verification records are written to `outputs/paper/` only.

## Inputs

The frozen recording and fitted representations are read from
`/ceph/aeon/aeon/nldisco/aeon/sweeps/20260921_18h_s20/`:

- `data/behavior.parquet`, source counts, neural-valid masks and cluster metadata;
- `d192/k12_seed0/` and `d256/k12_seed0/` activations, normalization and checkpoints;
- `paper_revision_wheelcms_20260921/` numerical feature/tuning/ROC inputs and
  `attribution/traceback_attribution.npz`.

`outputs/appendix_decoders_20260925/` contains all eight single/full-space logistic
fits, validation-selected thresholds and test predictions. The original wheel fit
and balanced-accuracy reanalysis remain because they establish the exact fit reused
by the current table. No models or data on Ceph are deleted by this cleanup.

## Regeneration and checks

The notebook reconstructs all four condition/validity masks from recorded behavior,
checks them against the frozen masks, and recalculates coverage, specificity,
selectivity and AUROC. It also recalculates decoder balanced accuracy from the saved
probabilities and fixed validation-selected thresholds. The wheel feature uses
D192/k12/latent 52; area, speed and outward direction use D256/k12/latents 107, 109, 10.

The two temporal excerpts (hours 1.2–1.3 and 11.0–11.1) are **regenerated from native
positive latent activations and recorded behavior**, including invalid-data markers.
Their ticks differ from the current manuscript SVG. `wheel_tick_comparison.json`
records the exact counts; the existing paper is left unchanged. Source tracebacks
use gradient-times-input maps and temporal decoder weights, not conditional means
of neural activity. All four features are included in Table S3 and the traceback.

## Upstream computation

The retained `sweep_20260921/` helpers contain streamed training/export, numerical
feature analysis, current feature definitions, temporal-state construction and
encoder attribution. To regenerate numerical feature inputs from the frozen
representations into a **new** directory:

```bash
uv run --no-sync python -m experiments.aeon.sweep_20260921.paper_results \
  --output experiments/aeon/outputs/recomputed_features
```

To fit the single/full-space decoders again (substantially slower than notebook plotting):

```bash
uv run --no-sync python -m experiments.aeon.fit_appendix_decoders \
  --output experiments/aeon/outputs/recomputed_decoders
```

To retrain a selected configuration in a fresh directory, use `shared_training.py`
with the frozen `data` directory, `--d 192` or `--d 256`, `--k 12`, `--steps 5000`,
`--batch-size 1024` and `--seed 0`; inspect its `--help` for device options.
Checkpoints reproduce the saved analysis; new hardware/software need not retrain
bitwise-identical latents. The default notebook neither trains nor changes frozen data.

Exploratory gate/off-wheel features, D128 drafts and hardware-specific one-off launch
scripts are removed. S20's outward-direction condition is defined directly in the
current analysis helper, without depending on the retired gate-analysis workflow.
