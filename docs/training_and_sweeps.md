# Training and hyperparameter sweeps

One-off training and sweeps share a **Hydra structured configuration**, the current
`train_model`/`evaluate_model` APIs, preprocessing, and output format. Install with `uv sync`.
The CLI defaults to raw nonnegative counts, MSLE loss, a ReLU decoder, and CPU execution.
For signed targets, set `loss.type=mse model.decoder.output_activation=none`.

```bash
# One training run; activity is a numeric .npy array shaped [timebin, unit].
uv run python -m nldisco.sweep --config-name train data.path=/data/counts.npy training.epochs=10

# Local grid: learning rates [0.001, 0.005] × seeds [0, 1].
uv run python -m nldisco.sweep data.path=/data/counts.npy

# Two concurrent CPU workers, or one worker on each of two visible GPUs.
uv run python -m nldisco.sweep data.path=/data/counts.npy execution.max_parallel=2
uv run python -m nldisco.sweep data.path=/data/counts.npy \
  'execution.devices=[cuda:0,cuda:1]' execution.max_parallel=2

# Hydra's native multirun is also available for one-off configurations.
uv run python -m nldisco.sweep --config-name train -m data.path=/data/counts.npy \
  training.learning_rate=0.001,0.005 seed=0,1

# Inspect all defaults without training.
uv run python -m nldisco.sweep --config-name train --cfg job
```

`nldisco.sweep` defaults to the `sweep` preset; `--config-name train` selects one run.
Both presets accept the same YAML and CLI overrides. Use one search mechanism per invocation:
Hydra `-m` launches one application invocation per combination; `mode=sweep` runs the
configured NLDisco search with its own concurrency and total trial limit.

## YAML and CLI overrides

For a custom `configs/my_experiment.yaml`, extend the registered schema:

```yaml
defaults:
  - nldisco_schema
  - _self_

mode: train
seed: 0
data:
  path: /data/counts.npy
  normalization: minmax
  split: chronological
  train_fraction: 0.8
model:
  seq_len: 8
  dsed_topk_map: {128: 16}
  decoder:
    output_activation: none  # Held-out min-max values can be negative.
loss:
  type: mse
training:
  epochs: 10
  learning_rate: 0.001
execution:
  devices: [cpu]
search:
  method: random
  max_runs: 12
  seed: 42
  parameters:
    training.learning_rate:
      distribution: log_uniform_values
      min: 0.0001
      max: 0.01
    model.dsed_topk_map:
      values: [{128: 8}, {256: 16}]
    seed:
      values: [0, 1, 2]
```

```bash
uv run python -m nldisco.sweep --config-dir "$PWD/configs" --config-name my_experiment \
  training.epochs=20
uv run python -m nldisco.sweep --config-dir "$PWD/configs" --config-name my_experiment \
  mode=sweep search.max_runs=6
```

Hydra checks field names and types; runtime checks validate architecture, losses, and resources.
Its ordinary dictionary overrides merge keys. For a different SED dimension, define the full map
in a custom YAML extending `nldisco_schema`, as above. A search parameter targeting
`model.dsed_topk_map` replaces that entire map for each candidate.

Searchable dotted paths are `model.*`, `training.*`, `loss.*`, and `seed`. A seed is an
ordinary search dimension; it does not silently multiply the run limit. Local grid search
takes the first `search.max_runs` combinations in configuration order. Local random search
samples with replacement using `search.seed`. Supported specifications are `value`, `values`,
and `distribution` with `min`/`max`: `uniform`, `log_uniform_values`, or `int_uniform`
(inclusive integer bounds). Bayesian search is available through W&B.

## Data, splits, and normalization

Use `nldisco.data.preprocessing` to [bin Kilosort/Phy data](preprocessing.md) before training.
The runner consumes a numeric `.npy` `[timebin, unit]` array; optional `.npy` row metadata
are `data.valid_rows` (boolean), `trial_ids`, `session_ids`, and `timestamps`. Metadata must
have shape `[timebin]` and use non-object NumPy dtypes. Timestamps additionally require
`data.expected_bin_size` in matching units.

- `data.split=chronological`: the first `train_fraction` of source rows is training data.
- `data.split=trial`: split whole `(session, trial)` identities, using `data.split_seed`.
- `data.split=session`: select `data.train_sessions`, e.g. `'data.train_sessions=[day1,day2]'`.
  Numeric session IDs can be specified as strings.

Supplied invalid rows and missing metadata are excluded without collapsing time. Windows
cannot cross the split, trial/session boundaries, or timestamp gaps. Normalization (`none`,
`zscore`, or `minmax`) is fitted per unit using **training rows only** and reused on validation
data. Held-out min-max values are not clipped. Signed **targets** require MSE and
`model.decoder.output_activation=none`; z-scoring inputs alone imposes no such restriction.

To predict a separate population, supply `data.target_path` with another numeric `.npy`
`[timebin, output_unit]` array. Input and target rows must share the same time grid and
window length; their unit counts can differ. The runner infers both widths. Optional
`data.target_valid_rows` is intersected with input validity before splitting and fitting
statistics. Shared trial/session/timestamp metadata apply to both arrays. Optional
`target_trial_ids`, `target_session_ids`, and `target_timestamps` must match their input
counterparts wherever both are present; when only target metadata are supplied, they
become the shared metadata. Missing metadata rows remain excluded on both sides.
`data.input_population` and `data.target_population` may identify the populations in saved
configuration; encoder attribution describes inputs and decoder patterns describe targets.

```bash
# Z-scored input activity predicting raw, nonnegative activity in another population.
uv run python -m nldisco.sweep --config-name train data.path=/data/input_counts.npy \
  data.target_path=/data/output_counts.npy data.normalization=zscore \
  data.target_normalization=none loss.type=msle model.decoder.output_activation=relu

# Independently z-score both populations, fitting each on the same training rows.
uv run python -m nldisco.sweep --config-name train data.path=/data/input_counts.npy \
  data.target_path=/data/output_counts.npy data.normalization=zscore \
  data.target_normalization=zscore loss.type=mse model.decoder.output_activation=none
```

`data.target_normalization` accepts `none`, `zscore`, or `minmax`. Its default `null` means
`none` when a separate target is supplied; without a separate target it inherits
`data.normalization`, preserving autoencoder behavior. Separate target normalization
without `target_path` is rejected. Each population's training statistics are reused for
validation and saved for inference. A held-out target transformed below zero is incompatible
with MSLE or a nonnegative decoder even when training targets were nonnegative.

An empty `loss.timebin_weights` uses uniform weights for the selected sequence length.
`data.stride=null` uses nonoverlapping windows for ordinary models. For shift-equivariant
TransformerWindow models it uses the number of supported occurrence positions, with boundary
and overlap masks supplied by `SpikeWindowDataset`.

The runner materializes activity on CPU and transfers minibatches to the selected GPU.
Evaluation uses the existing full evaluation API, which collects reconstructions and feature
activations in memory. Size data/windows and worker concurrency accordingly for long recordings.
The two partitions are training and validation; keep a separate test set for final reporting.

## Weights & Biases and .env

Copy `.env.example` to `.env` and supply `WANDB_API_KEY`, or authenticate using `wandb login`.
Existing environment variables take precedence. `env_file=/path/to/file` selects another file;
`env_file=null` disables loading. The file is optional. Workers load it again on their host,
so use a shared path on Slurm. Credentials are read from the environment and are not inserted
into saved application configs. Keep credentials out of YAML and CLI overrides, which Hydra saves.
Data paths in YAML may use `${oc.env:NLDISCO_DATA}` after setting that value in `.env`.

```bash
# Log a one-off run (offline logging is also available).
uv run python -m nldisco.sweep --config-name train data.path=/data/counts.npy wandb.enabled=true \
  wandb.project=my-project wandb.mode=offline

# W&B Bayesian search with two local agents.
uv run python -m nldisco.sweep --config-name wandb data.path=/data/counts.npy \
  wandb.project=my-project execution.max_parallel=2 search.max_runs=20

# Join an existing sweep, with at most five additional assignments for this invocation.
uv run python -m nldisco.sweep --config-name wandb data.path=/data/counts.npy \
  wandb.project=my-project wandb.sweep_id=SWEEP_ID search.max_runs=5
```

W&B sweeps require online access. W&B chooses parameter assignments and minimizes
`validation/loss`, the full-window weighted reconstruction loss at the largest SED level.
Each agent receives an explicit quota whose sum is `search.max_runs`; new sweeps additionally
set the server's `run_cap`. Joining an existing sweep leaves its search space and global cap
unchanged. Supply matching local model/data settings. Exhausted grids or stopped sweeps can
complete fewer runs than the requested cap. Local searches can also log trials to W&B using
`wandb.enabled=true`; logging is separate from the search backend.

## Slurm through Submitit

Search and execution backends are independent: local grid/random search and W&B search
both support local processes or Slurm jobs through Submitit. Slurm is selected explicitly;
the runner does not silently fall back to local execution.

```bash
# Local grid search, with trials scheduled as a GPU Slurm array.
uv run python -m nldisco.sweep --config-name slurm data.path=/shared/counts.npy \
  execution.slurm.partition=gpu execution.slurm.account=my-account

# W&B search; Slurm schedules agents, which obtain assignments from W&B.
uv run python -m nldisco.sweep --config-name slurm data.path=/shared/counts.npy \
  search.backend=wandb search.method=bayes wandb.project=my-project \
  execution.slurm.partition=gpu

# A one-off CPU training job on Slurm.
uv run python -m nldisco.sweep --config-name train data.path=/shared/counts.npy \
  execution.backend=slurm execution.wait=false
```

`execution.max_parallel` bounds local processes or Slurm array concurrency. Slurm resources
are per trial or per W&B agent: `timeout_min`, `cpus_per_task`, `mem_gb`, `gpus_per_node`
(zero or one), `partition`, `account`, and `constraint`. GPU jobs use `[cuda]`; Slurm assigns
the physical GPU and the worker uses its logical device 0. Local workers use the selected
visible device indices, with no concurrent sharing of a GPU. `execution.threads_per_worker`
sets PyTorch's CPU thread count.

`execution.wait=false` returns after Slurm submission and records IDs in `jobs.json`.
The packaged `slurm` preset uses this setting; otherwise the default is to wait and report
worker failures. The same environment, package installation, code, data, and output paths must
be accessible on compute nodes. W&B agents additionally need network access there.

## Outputs and Python use

Each invocation reserves a new directory beneath `output_dir` with a timestamp and random
suffix. Each successful trial saves `config.yaml`, `model.pt`, `metrics.json`, `metrics_by_lag.csv`,
`history.json`, `status.json`, and (when used) `normalization.npz` for inputs and
`target_normalization.npz` for outputs. Checkpoints contain calibrated model weights, the
resolved input/target configuration, both unit counts, and both normalization states (or
explicit `None` for unnormalized populations); they are inference checkpoints, not
optimizer-resume checkpoints. Failed trials retain their error status. Waited executions
write `summary.json` and return a nonzero CLI exit status on failure. Asynchronous Slurm jobs
write individual status files and Submitit logs; the submitter cannot report future failures.

```python
import numpy as np
import torch
from nldisco.sweep.training import load_checkpoint

saved = load_checkpoint("path/to/model.pt", device="cpu")
model = saved.model  # Already in eval mode; saved sparsity thresholds are restored.
inputs = np.load("path/to/new_input_activity.npy")  # [timebin, input_unit]
if saved.input_normalization is not None:
    inputs = saved.input_normalization.transform(inputs)
x = torch.as_tensor(inputs[:model.cfg.seq_len], dtype=model.cfg.dtype).unsqueeze(0)
with torch.no_grad():
    predictions = model(x).reconstructions[max(model.cfg.dsed_topk_map)]
# predictions: [batch, timebin, output_unit]; targets are unnecessary for inference.
if saved.target_normalization is not None:
    stats = saved.target_normalization
    predictions = predictions.cpu().numpy() * stats.scale + stats.offset
```

`load_checkpoint` reads new checkpoints without access to the original data paths. It also
loads existing autoencoder checkpoints using `n_units` and the adjacent
`normalization.npz` when normalization was enabled. Keep that sidecar with legacy weights.
For manual reconstruction, pass both saved widths to
`model_and_loss(cfg, checkpoint["n_input_units"], checkpoint["n_output_units"])`.
`prepare_data(cfg, model_cfg)` retains its original `(train_loader, validation_loader,
input_stats)` return; `return_target_stats=True` appends `target_stats`.

For Python orchestration, compose the packaged `train`/`sweep` configuration using Hydra's
`initialize_config_module(config_module="nldisco.sweep.conf", version_base="1.3")` and
`compose`, then call `nldisco.sweep.main.launch(cfg)`. Import that module first to register
the schema. Public trial helpers also support direct use without a Hydra process.

The configuration and dispatch conventions follow the official
[Hydra structured schema guide](https://hydra.cc/docs/tutorials/structured_config/schema/),
[W&B sweep configuration reference](https://docs.wandb.ai/models/sweeps/sweep-config-keys),
and [Submitit job-array examples](https://github.com/facebookincubator/submitit/blob/main/docs/examples.md).
