# Change 3: end-to-end feature-discovery comparison on Aeon session B

Protocol (fixed before session B existed): `PROTOCOL.md`. Part A feature
definitions: `PART_A_FEATURES.md` / `features.py`. Rater instructions:
`RATER_INSTRUCTIONS.md`. Small results: `outputs/`. Large outputs:
`/ceph/aeon/aeon/nldisco/rebuttal_20261118/change03_discovery/session_b/`.

Run from the repository root with the project interpreter (`PY`); `DATA` is the
prepared session, `RUNS` a new output folder.

```bash
# Representations (one GPU each; W&B off). Fit on the fit portion only.
$PY -m experiments.rebuttal.change03_discovery.run_nldisco --data $DATA --output $RUNS/nldisco/d192_k12_seed0
$PY -m experiments.rebuttal.change03_discovery.run_pca --data $DATA --output $RUNS/pca/d192
$PY -m experiments.rebuttal.change03_discovery.run_cebra --data $DATA --output $RUNS/cebra/lr1e-3 --config lr1e-3   # also lr1e-4, lr3e-4
$LANGEVIN_PY -m experiments.rebuttal.change03_discovery.run_langevinflow --data $DATA --output $RUNS/langevinflow/lr0p001_ramp500 --config lr0p001_ramp500   # 4 configs
$STNDT_PY -m experiments.rebuttal.change03_discovery.run_stndt --data $DATA --output $RUNS/stndt/cw0p1_lr0p003 --config cw0p1_lr0p003   # 4 configs; needs experiments/rebuttal/change01_stndt

# Part A: selects each baseline's run by its own validation loss, then scores.
$PY -m experiments.rebuttal.change03_discovery.part_a --data $DATA --runs $RUNS --output $RUNS/part_a
# Part B: cards and rater folders, then (after the raters) held-out validation.
$PY -m experiments.rebuttal.change03_discovery.part_b build --data $DATA --runs $RUNS --output $RUNS/part_b
$PY -m experiments.rebuttal.change03_discovery.part_b analyze --data $DATA --output $RUNS/part_b
# Tables, example figure and headline numbers.
$PY -m experiments.rebuttal.change03_discovery.paper --data $DATA --results $RUNS
```

`LANGEVIN_PY` is the LangevinFlow overlay interpreter
(`experiments/churchland/environments/README.md`); `STNDT_PY` is change 1's overlay
(`experiments/rebuttal/change01_stndt/README.md`). `launch.sh` starts one runner
detached on one GPU. Each rater is one agent that reads `INSTRUCTIONS.md` in its
folder and uses only `rater_cli.py` (`next` / `submit`), which logs serve and
answer times.

Tests: `$PY -m pytest experiments/rebuttal/change03_discovery`; the LangevinFlow
adapter check is skipped there and runs with
`$LANGEVIN_PY -m pytest experiments/rebuttal/change03_discovery/test_c03_langevinflow.py`.
