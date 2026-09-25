# Neural data preprocessing

`nldisco.data.preprocessing` contains data loading, spike binning, and activity
normalization. Every activity matrix uses **[timebin, unit]** order. Windows add
a leading example dimension to give **[batch, timebin, unit]**. The model API's
`n_neurons` argument specifies the input unit count; optional `n_output_neurons`
specifies the target unit count and defaults to the input count. A sorter cluster
need not be an isolated neuron.

## Kilosort / Phy exports

```python
import torch
from nldisco.data import SpikeWindowDataset, fit_normalizer, load_kilosort

binned = load_kilosort(
    "path/to/sorter_output",
    sampling_rate=30_000,
    bin_size=0.02,
    start_time=0,
    stop_time=600,  # Known recording duration, not the time of the last spike.
    # labels=["good", "mua"],  # Optional; the default retains every cluster.
)
assert binned.counts.shape[1] == len(binned.unit_ids)

# Fit only on the training portion; reuse identical statistics for held-out data.
split = int(0.8 * len(binned.counts))
normalizer = fit_normalizer(binned.counts[:split], method="zscore")
activity = normalizer.transform(binned.counts)
train = SpikeWindowDataset(
    torch.as_tensor(activity[:split], dtype=torch.float32),
    seq_len=20,
    timestamps=binned.timestamps[:split],
    expected_bin_size=0.02,
)
validation = SpikeWindowDataset(
    torch.as_tensor(activity[split:], dtype=torch.float32), seq_len=20,
    timestamps=binned.timestamps[split:], expected_bin_size=0.02,
)
```

The loader reads `spike_times.npy` and `spike_clusters.npy` without requiring a
Kilosort installation. Both one-dimensional vectors and `(n_spikes, 1)` arrays
are accepted. Sample indices and cluster IDs must be nonnegative integers.
The sampling rate is explicit; the loader does not execute `params.py` or read
pickled `ops.npy`. These are the standard
[Kilosort export files](https://kilosort.readthedocs.io/en/stable/export_files.html).
The exported assignments are used as-is; no additional duplicate-spike mask is
applied. Phy-edited cluster assignments are supported.

`unit_ids=[...]` selects clusters explicitly. `labels=[...]` filters by
`cluster_group.tsv` when present, otherwise `cluster_KSLabel.tsv`. Curated
metadata takes precedence as a whole; an unlabeled cluster is excluded from a
label selection. Missing label files raise an error if filtering is requested.
Supplying both filters intersects them. Output columns are sorted by original
cluster ID, with the mapping preserved in `binned.unit_ids`.

`bin_spikes` exposes the same binning operation for paired sample-index and
unit-ID arrays from other sources. All intervals are **[start, stop)**, including
the last bin, with exact integer-sample arithmetic. Bounds and bin size must
align to samples. A non-divisible duration produces a shorter final bin, whose
actual edges and center are returned. Drop that partial bin before constructing
fixed-duration windows. Leading/trailing silent bins and selected silent units
are retained. An empty unit selection returns a matrix with zero columns.

The exported sample clock is preserved. Synchronization to behavioral clocks,
clock-gap detection, and cross-recording unit matching belong to the dataset's
analysis. In particular, the Aeon paper's acquisition-clock corrections are not
part of this loader. Retain invalid-row masks and pass timestamps, session IDs,
or `allowed_rows` to `SpikeWindowDataset` to prevent windows crossing gaps.

## Normalization

```python
from nldisco.data import normalize_activity

# Each unit independently across time (default):
zscore = normalize_activity(binned.counts, method="zscore", axis=0)
minmax = normalize_activity(binned.counts, method="minmax", axis=0)

# Each timebin independently across units:
population_zscore = normalize_activity(binned.counts, method="zscore", axis=1)
population_minmax = normalize_activity(binned.counts, method="minmax", axis=1)
```

Z-scoring subtracts the mean and divides by the population standard deviation
(`ddof=0`). Min-max scaling subtracts the minimum and divides by the range.
Constant slices become zero, using a denominator of one. Values must be finite;
empty matrices and NaN/Inf values are rejected. Inputs are never modified, and
normalized output is float64; convert to your model's dtype when constructing
tensors. Z-scored targets can be negative, so use a compatible reconstruction
loss and output activation, such as MSE and the identity activation.

`normalize_activity(axis=0)` fits on every supplied row. Use
`fit_normalizer(training_rows)` and its `transform` method for validation/test
analyses. Exclude invalid rows from the fit and keep unit columns in identical
order across splits. A constant training unit uses scale one, so a held-out
deviation remains visible. Min-max transformation does not clip held-out values:
values outside the training range can produce values outside [0, 1]. With axis 1,
each timebin is normalized independently and there are no fitted statistics to
share between time bins.

## Paired populations

For transcoding, inputs and targets use `[timebin, unit]` order with the same
number of rows, timestamps, and window length. Unit counts and column identities
can differ. Fit normalization separately on each population using the same valid
training rows, then reuse each fitted state for validation and inference:

```python
from nldisco.data import SpikeWindowDataset, fit_normalizer

# input_counts and target_counts are aligned arrays; train_rows is a boolean mask.
# Intersect input/target validity and split boundaries before fitting.
input_stats = fit_normalizer(input_counts[train_rows], method="zscore")
target_stats = fit_normalizer(target_counts[train_rows], method="minmax")
train = SpikeWindowDataset(
    torch.as_tensor(input_stats.transform(input_counts), dtype=torch.float32),
    seq_len=20,
    targets=torch.as_tensor(target_stats.transform(target_counts), dtype=torch.float32),
    allowed_rows=train_rows,
    trial_ids=trial_ids,
    timestamps=timestamps,
    expected_bin_size=0.02,
)
```

`valid_rows` and `target_valid_rows` apply jointly to every window; both sides
must contain finite activity. Optional target timestamps and trial/session IDs
are checked against the input metadata. The same source indices are used for
both tensors. `sample.values` is the input window; `sample.targets` is its target (`sample.target` is an alias).
Omitting `targets` preserves autoencoder behavior. PyTorch's default collation
preserves both tensors and the shared metadata.

Only target preprocessing constrains the reconstruction loss and decoder output
activation. Z-scored inputs may predict nonnegative raw targets using MSLE and
a ReLU decoder. Z-scored targets require MSE and an identity decoder. Min-max
held-out targets can become negative; use MSE and an identity decoder in that
case. Encoder attribution and input unit identifiers refer to the input population;
decoder patterns and output unit identifiers refer to the target population.
Use the target statistics to invert predictions back to the target's original units.
The Hydra runner handles these steps through `data.target_path`,
`data.target_normalization`, and `data.target_valid_rows`; see
[training and sweeps](training_and_sweeps.md).

## Long recordings

Inputs may be NumPy memory maps. The loader processes spikes in chunks, and
normalization fits stable float64 statistics in row chunks. `chunk_size` controls
the working block size. Counts are int64 to avoid narrow-integer overflow.

Loading/binning and normalization accept `output_path="new_file.npy"` to write
memory-mapped output instead of allocating the complete output in RAM. Existing
files are never overwritten. The output directory must already exist; a failed
write or invalid input discovered during streaming may leave a partial file.
Without `output_path`, only intermediate working arrays are chunked; the output
still occupies memory proportional to the complete matrix.

