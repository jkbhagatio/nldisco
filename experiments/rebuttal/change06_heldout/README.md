# Change 6: held-out validation of latent interpretations

Retrains the paper's NLDisco models on a training part only, selects each feature's best-matching
latent on the training part, and scores it on the held-out part (Cov, Spec, Sel, AUROC; three seeds).
Features and cutoffs are the paper's (macaque: `score_macaque.revised_labels` and
`feature_decoding.label_windows`; Aeon: `paper_results.definitions`).

- Macaque split: the stored whole-trial split (`split` 0 = training; 1 and 2 = held-out).
- Aeon split: `aeon_split.py` (72 blocks of 15 min, blocks with index mod 5 == 2 held out).

Run from the repository root with the main checkout's interpreter (`PY=/nfs/nhome/live/jbhagat/nldisco/.venv/bin/python`,
`WANDB_MODE=disabled`, `PYTHONPATH=.`) and a free GPU:

```bash
$PY -m experiments.rebuttal.change06_heldout.macaque_run --seed 0 --output RUNS/macaque_seed0
$PY -m experiments.rebuttal.change06_heldout.aeon_run --d 192 --k 12 --seed 0 --output RUNS/aeon_d192_seed0
$PY -m experiments.rebuttal.change06_heldout.aeon_run --d 256 --k 12 --seed 0 --output RUNS/aeon_d256_seed0
$PY -m experiments.rebuttal.change06_heldout.score_macaque --runs RUNS --output macaque_scores.json
$PY -m experiments.rebuttal.change06_heldout.score_aeon --runs RUNS --output aeon_scores.json
$PY -m experiments.rebuttal.change06_heldout.make_table --macaque macaque_scores.json --aeon aeon_scores.json --output-dir OUT
```

Tests: `$PY -m pytest experiments/rebuttal/change06_heldout`.
