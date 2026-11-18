# Multi-session NLDisco on Aeon: transfer of interpretations across sessions

Rebuttal change (2). The paper's frozen session A model (Section 4.3) is extended
to further sessions of the same mouse and probe insertion through new input
mappings and decoders (`nldisco.model.multisession`), trained on neural activity
only. Behavior is used for scoring alone.

Large outputs are on Ceph under
`/ceph/aeon/aeon/nldisco/rebuttal_20261118/change02_multisession/` (`$C` below).
Run everything from the repository root with the change's interpreter,
`$C/venv/bin/python` (`WANDB_MODE=disabled`).

## Sessions

| Paper name | Interval (half-open) | Data directory | Units | Valid windows | Food deliveries |
|---|---|---|---:|---:|---:|
| A | 2024-06-06 23:00 to 2024-06-07 17:00 | `/ceph/aeon/aeon/nldisco/aeon/sweeps/20260921_18h_s20/data` | 90 | 3,238,600 | 181 |
| B | 2024-06-05 23:00 to 2024-06-06 17:00 | `$C/session_b` (frozen behind `READY.json`) | 87 | 3,238,618 | 176 |
| C | 2024-06-07 23:00 to 2024-06-08 17:00 | `$C/more_sessions/session_c` | 93 | 3,238,621 | 152 |
| D | 2024-06-08 23:00 to 2024-06-09 17:00 | `$C/more_sessions/session_e` | 86 | 3,189,848 | 162 |

The paper's session D is stored in the folder `session_e`. The block starting
2024-06-04 fails the preparation script's raw-file length check and is not used.

## Steps

1. `prepare_session.py` (ingestion interpreter `$C/ingest_venv/bin/python`):
   the session A script with block, interval and output as arguments; stages
   `neural`, `behavior`, `behavior_qc`, `ready`. `unit_positions.py` writes each
   cluster's probe position.
2. Replication: `python -m experiments.aeon.sweep_20260921.shared_training --data $C/session_b
   --output $C/models/b_only/d192_k12_seed0 --d 192 --k 12 --steps 5000 --batch-size 1024 --seed 0`
   (and `--d 256`), then `evaluate_single.py` and `screen_features.py`.
3. Transfer: `adapt.py`. One run from flags, or a grid in one process, e.g.
   `python -m experiments.rebuttal.change02_multisession.adapt --output $C/explore/runs2 --grid $C/explore/grids/g2a.json`.
   `--init position|random|shuffled`, `--steps 0` for position matching alone,
   `--data`/`--positions` for another session. Each run writes `checkpoint.pt`,
   `summary.json`, `latent_scores.csv`, `tuning_profile.npy` and
   `identity_transfer.json`.
4. Session B feature: `new_feature.py --run $C/models/b_only/d192_k12_seed0 --latents 77 53 --window 0.3 1.0 --output ...`.
5. Joint training (exploratory): `python -m experiments.rebuttal.change02_multisession.joint --output $C/explore/joint/d192_position_seed0 --tie position`
   (`--tie none` for independent initialisation).
6. Figures and tables: `schematic.py`, then
   `python -m experiments.rebuttal.change02_multisession.figures --runs $C/eval/figure_runs.json
   --controls $C/eval/controls.json --pellet $C/eval/b_only_d192_k12_seed0/pellet_w0.3_1.0
   --pellet-latent 77 --pellet-window 0.3 1.0`, which writes
   `paper/iclr_paper/figures/rebuttal/change02_multisession/`.

## Tests

`python -m pytest tests/test_multisession.py experiments/rebuttal/change02_multisession -q`
