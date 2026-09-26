# Evaluation

`nldisco.train.evaluate_model(model, validation_loader, loss_config)` returns:

- `metrics_by_lag`: reconstruction loss, cosine similarity, and R² at each relative lag.
- `weighted_reconstruction`: full-window loss at the largest Matryoshka level.
- `reconstructions` and `targets`: `[window, timebin, output_unit]` tensors.
- `activation_table`: positive activations with latent, replica, and source-time identifiers.
- `evaluation_index`: all eligible source positions, including those with no active latents.

Join activations to behavioral metadata using `source_time_idx`; include inactive positions
from `evaluation_index` when calculating coverage or specificity. Reconstruction quality
alone does not establish interpretability. See [model inference](model_names.md#inference)
for sparsity and temporal-support rules.

## Optional diagnostics

Ablation and spectral diagnostics are disabled by default. They evaluate a frozen model with its existing
inference sparsity settings. They do not fit probes, recalibrate thresholds, or measure
behavioral interpretability. Inputs use the fitted input preprocessing. Reconstruction
errors and power estimates use the fitted **target** representation (for example,
per-output-unit z-scores).

For paired populations, `WindowSample.values` contains encoder inputs and
`WindowSample.targets` contains aligned targets. Every diagnostic encodes only the
inputs and compares decoded outputs against targets, including when the populations
have equal widths but different values. With no separate targets, the dataset uses
the inputs as targets and retains autoencoder behavior. Both populations share their
time grid, window length, split, trial/session boundaries, and valid rows.

## Training and sweeps

Add this block to any [Hydra training or sweep configuration](training_and_sweeps.md):

```yaml
evaluation:
  ablation:
    enabled: true
    level: null              # Largest Matryoshka level
    latents: null            # All latents; or a list such as [0, 12, 52]
    latent_batch_size: 8     # Number of ablations decoded together
  spectral:
    enabled: true
    level: null
    bin_size: 0.02           # Seconds; null uses data.expected_bin_size
    segment_duration: 2.0   # Seconds, independent of the model window duration
    lag: 0                  # Endpoint; -1 selects the preceding reconstruction bin
    bands:
      slow: [0.5, 4.0]      # Hz; choose bands appropriate to the recording
      fast: [4.0, 10.0]
```

The same options work for single runs and sweeps, locally or via Submitit/Slurm. For example:

```bash
uv run python -m nldisco.sweep --config-name train data.path=/absolute/path/counts.npy \
  evaluation.ablation.enabled=true evaluation.spectral.enabled=true \
  evaluation.spectral.bin_size=0.02 evaluation.spectral.segment_duration=2.0
```

Diagnostics run after training and threshold calibration on validation data. Ablations
use the ordinary evaluation windows. Spectral evaluation builds a separate stride-one
view of the same validation rows, preserving trial/session boundaries, timestamp gaps,
and both populations' preprocessing and validity. This does not change the training
stride or validation loss used for sweep selection. The spectral pass reconstructs each
window with all its valid latent occurrences, then exports only the selected lag.

Each run saves `diagnostics/latent_ablation.csv`, a ranked PDF plot, `spectra.npz`
(frequency-by-output-unit arrays), `spectral_metrics.csv`, optional `band_power.csv`, a spectral
comparison PDF, and metadata and finite summary values in JSON. W&B-enabled runs log
summary metrics and diagnostic tables. Undefined relative scores are NaN in tables;
they are omitted from JSON averages.

## Existing models and checkpoints

For a model already loaded from a checkpoint, use the same API without training:

```python
from pathlib import Path
from nldisco.evaluation import (
    AblationConfig, EvaluationConfig, SpectralConfig,
    evaluate_diagnostics, save_diagnostics,
)

options = EvaluationConfig(
    ablation=AblationConfig(enabled=True, latents=[0, 1, 2]),
    spectral=SpectralConfig(enabled=True, bin_size=0.02, segment_duration=2.0),
)
diagnostics = evaluate_diagnostics(
    model, validation_loader, options, list(loss_config.timebin_weights),
    spectral_loader=dense_validation_loader,
)
save_diagnostics(diagnostics, Path("outputs/new_diagnostics"))
```

Both loaders should emit `WindowSample` batches on the same preprocessed evaluation
split. The spectral loader must cover consecutive source bins; it need not be ordered,
because the analysis sorts by session, trial, and source identity. For a
`SpikeWindowDataset`, `nldisco.sweep.training.spectral_validation_loader` constructs
the dense view used by the runner. Alternatively pass `diagnostics=options` and
`spectral_loader=...` to `nldisco.train.evaluate_model`; the result has a `diagnostics`
field. Optional passes add inference cost; ablations stream sufficient statistics
and decode bounded groups, while the spectral pass retains only one time bin per
window on the CPU. `save_diagnostics` requires a new output directory.

Use `SpikeWindowDataset(inputs, seq_len, targets=targets, ...)` for paired diagnostics.
Apply the checkpoint's input normalizer to inputs and its target normalizer to targets;
see the [checkpoint reload example](training_and_sweeps.md). Targets are required to
measure reconstruction quality; ordinary model inference needs only input windows.

## Latent ablation importance

For each requested latent, zero its sparse activations and decode again, without
re-encoding or rerunning sparsification. All other codes, decoder bias, and decoder
output activation remain fixed. For temporal codes, remove every occurrence of the
latent. This supports linear and convolutional decoders, including ReLU/softplus output.

The reported score is `(SSE_without - SSE_full) / SST_target`. Target SST centers each
unit separately at each relative lag across evaluation windows. The full-window score
uses the configured time weights for both SSE and SST; it is not an average of per-lag
ratios. CSV rows include `full_mse`, `ablated_mse`, `delta_mse`, `delta_r2`, and `target_sst`.
This diagnostic always uses squared error, even if training uses another loss.
MSE divides by the number of windows times the **output** unit count. Thus input
population size does not change the score's denominator.

Positive scores indicate helpful contributions, and negative scores indicate that
removing a latent improves reconstruction. Scores **do not sum to total explained
variance**: latents can be correlated, redundant, or interact through the output
activation. Constant targets have undefined `delta_r2` but valid squared-error scores.
Uneven importance is not itself a failure, nor does reconstruction importance establish
behavioral or causal importance. Ablation does not allow the remaining latents to adapt.

## Spectral reconstruction fidelity

The lower-level `spectral_comparison` function accepts aligned `timebin × unit` arrays,
source indices, and optional trial/session codes and timestamps. Duplicate physical
bins are rejected. Never flatten overlapping windows into a time series. If using a
custom dataset, supply timestamps to this function or encode time gaps in source indices;
`spectral_from_model` automatically uses `SpikeWindowDataset` timestamps when available.

Welch PSDs use Hann windows, constant detrending (removing each segment's mean), 50%
overlap, and only complete segments. Contiguous runs are averaged in proportion to
their segment counts. Segment duration must be an integer number of bins, at least
four; frequency-bin spacing is `1 / segment_duration`, and Nyquist is `1 / (2 * bin_size)`.
The code reports the number of segments and used/excluded samples, and rejects requests
with no sufficiently long contiguous stretch. A short model window does not prevent
analysis of slower frequencies across a long sequence of endpoint reconstructions.

Per-unit relative PSD error is `sum(abs(P_reconstruction - P_target)) / sum(P_target)`.
Band powers integrate discrete PSD bins in `[low, high)`, including Nyquist when it is
the upper edge. Zero target power yields an undefined relative error. Plots average
per-unit spectra, rather than taking a spectrum of population-averaged activity.
`unit_idx` in spectral and band-power tables indexes target/output columns. Saved
spectral metadata identifies the output population and its width. Decoder feature
heatmaps also describe output units, while encoder attribution describes input units.
Latent-to-behavior mapping retains the shared source time indices.
Identical power spectra do not imply identical phase or event timing; pointwise
reconstruction metrics remain necessary.

