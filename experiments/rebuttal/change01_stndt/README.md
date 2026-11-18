# ST-NDT baseline (rebuttal change 1)

`environments/STNDT/` is the unmodified MIT-licensed published source from
<https://github.com/trungle93/STNDT>, pinned at
`92ffe21c2042eb886389b7ebc97d300c2fd23cac`. Keep its license and source intact. The
vendored copy has no git metadata; `environments/setup_stndt.py` verifies the pin by the
SHA-256 of `src/model.py` and `src/mask.py`, and every run re-checks it.

## Environment

The model code needs only PyTorch and `yacs`. `environments/setup_stndt.py` builds a small
overlay environment that adds `yacs==0.1.8` (and PyMuPDF, used only to overlay vector
markers on the printed Figure 4; `environments/stndt-requirements.txt`) and inherits the
project environment through `nldisco_project.pth`:

```bash
python experiments/rebuttal/change01_stndt/environments/setup_stndt.py \
    --project-python <project>/.venv/bin/python --environment <new overlay directory>
STNDT_PY=<new overlay directory>/bin/python
$STNDT_PY -m pytest -q experiments/rebuttal/change01_stndt/test_stndt.py
```

The overlay used for the paper is
`/ceph/aeon/aeon/nldisco/rebuttal_20261118/change01_stndt/stndt-venv/bin/python`
(Python 3.9, PyTorch 2.5.1; the published environment is Python 3.8, PyTorch 1.11).

## Runner for any binned recording (`stndt.py`)

```bash
$STNDT_PY -m experiments.rebuttal.change01_stndt.stndt \
    --counts counts.npy --index index.npz --output NEW_DIRECTORY \
    [--train-mask train.npy] [--validation-mask heldout.npy] \
    [--layers 4 --heads 1 --hidden 128 --dropout 0.1 --contrast-weight 0.1 \
     --learning-rate 0.001 --epochs 10 --batch-size 256 --seed 0 --device cuda]
```

- `counts.npy`: integer spike counts, `(time bins, units)`.
- `index.npz`: `source_indices`, `(windows, bins per window)` rows of `counts` in time
  order; the last entry of each row is the bin the exported vector is aligned to.
  Optional `anchor_index` is copied to the output.
- `--train-mask` / `--validation-mask`: boolean `.npy` over windows. Training uses only
  the training windows (default all); held-out masked prediction is reported each epoch
  on the validation windows; representations are exported for every window.
- Outputs: `activations.npy` `(windows, units)` float32, `index.npz`, `model_final.pt`,
  `config.json`, `progress.json` (per-epoch masked Poisson NLL, contrastive loss and,
  with a validation mask, `masked_nll`, `mean_rate_nll` and `bits_per_spike`) and
  `summary.json` (`train_seconds` excludes validation and export).
- `train_and_export(...)` is the same entry point from Python.

Model facts that follow from the upstream code: the transformer width equals the number
of units (191 for the macaque recording), and `heads` must divide both the unit count
and the bins per window (191 is prime, so one head). The exported representation is the
upstream "features" (`layer_outputs` of the final transformer layer) at the last bin of
each window; because a window ends at its scored bin, no later activity is seen.

Adapter choices (the model and masking code are unchanged): settings start from the
upstream `configs/mc_maze.yaml`; attention spans the whole window (`FULL_CONTEXT`);
one optimiser update per minibatch with AdamW and the upstream warm-up/cosine schedule
(10% warm-up) stepped per update; the two contrastive views and the masked input are
drawn by the upstream `Masker` exactly as in the upstream training loop.

## Macaque sweep (`macaque.py`)

```bash
$STNDT_PY -m experiments.rebuttal.change01_stndt.macaque --mode sweep        # all windows
$STNDT_PY -m experiments.rebuttal.change01_stndt.macaque --mode convergence  # 80% trials
```

Twelve settings: contrastive weight {0.01, 0.1, 1} x learning rate {0.001, 0.003} x
dropout {0.1, 0.3}, ten epochs, batch 256, seed 0, on the shared 130,507 endpoint windows.
Outputs: `/ceph/aeon/aeon/nldisco/rebuttal_20261118/change01_stndt/{sweep,convergence}/`.

## Scoring, timing and paper assets

```bash
python -m experiments.rebuttal.change01_stndt.score            # project interpreter
sbatch experiments/rebuttal/change01_stndt/timing_h100.sh       # five H100 fits, seeds 0-4
$STNDT_PY -m experiments.rebuttal.change01_stndt.figures --timing <timing output directory>
```

- `score.py` reuses the Section 4.2 code unchanged (`comparison_features`,
  `nice_cutoffs_revision`, `feature_decoding.tune_probe`, `decoder_metrics`): it ranks the
  twelve runs by mean best-eligible-latent AUROC on the original development labels,
  keeps the best-Sel and best-AUROC coordinates of the selected run, rescores those fixed
  coordinates, signs and thresholds on the current labels, and fits the single-latent and
  full-space logistic probes (one BLAS thread, which reproduces the frozen baseline
  probes exactly). Outputs: `outputs/selection.json`, `outputs/roc_plot_data.json`,
  `outputs/results.json`; probes and test predictions under `decoding/` on Ceph.
- Applied to the frozen LangevinFlow export, this path returns the paper's selected
  latents and association scores for all four descriptions (to 1e-12) and, for the
  probes checked (recent deceleration single-latent and full-space, early leftward
  full-space), the same regularisation, threshold and test score.
- `timing.py` follows `experiments/churchland/TRAINING_TIME_BENCHMARK.md`: one H100
  pinned by UUID, fresh process per fit, four CPU threads, occupancy sampled every two
  seconds, and a run is rejected if any other compute process appears on the GPU.
- `figures.py` never regenerates an existing result: `figure4.pdf` is the printed figure
  with ST-NDT markers overlaid as vectors, the ROC and timing figures redraw the saved
  curves and the fifteen saved fits and add ST-NDT, and the two tables receive ST-NDT rows
  with column maxima re-bolded; every existing number is kept as printed
  (`test_scoring.py`). Assets go to `paper/iclr_paper/figures/rebuttal/change01_stndt/`.
