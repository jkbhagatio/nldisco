# Change 4: visual stimulus features in Allen session 732592105

Run from the repository root (outputs go to `/ceph/aeon/aeon/nldisco/rebuttal_20261118/change04_allen/`):

1. `python -m experiments.rebuttal.change04_allen.data`: align the processed counts with the NWB stimulus tables, label each 50-ms bin with the stimulus shown 50 ms earlier, and split presentations 80/20 (`cache/bins.parquet`).
2. `python -m experiments.rebuttal.change04_allen.train --name s1_z --normalization zscore`: train the SED on training bins only (also run: `s1_raw`, `s4_raw --seq-len 4`).
3. `python -m experiments.rebuttal.change04_allen.search --name s1_z`: score every latent against 473 candidate conditions on the training split.
4. `python -m experiments.rebuttal.change04_allen.results`: score the four selected features on both splits and render the appendix figure, table and metrics JSON into `paper/iclr_paper/figures/rebuttal/change04_allen/`.

Tests: `python -m pytest experiments/rebuttal/change04_allen/tests`.
