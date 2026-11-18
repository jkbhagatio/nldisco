#!/usr/bin/env bash
# Nine macaque retrains: batch sizes 512/1024/2048 x seeds 11,12,13 (D=192, k=16, 2 levels).
# Run from the worktree root. One stream per GPU (three runs each).
set -u
PY=/nfs/nhome/live/jbhagat/nldisco/.venv/bin/python
OUT=/ceph/aeon/aeon/nldisco/rebuttal_20261118/change05_batchsize/churchland
GPUS=(GPU-5a924816-0a8e-d1b5-086e-e98ffd4382bb GPU-4c0ffb35-cb66-c83a-3df3-833fd58876cd GPU-cbefc9af-e673-789a-7605-8209a24294fb)
export WANDB_MODE=disabled
mkdir -p "$OUT/logs"
i=0
for seed in 11 12 13; do
  gpu=${GPUS[$((i % 3))]}
  (
    for bs in 512 1024 2048; do
      CUDA_VISIBLE_DEVICES=$gpu $PY -m experiments.rebuttal.change05_batchsize.train_churchland \
        --batch-size $bs --seed $seed --output $OUT/bs${bs}_seed${seed} > $OUT/logs/bs${bs}_seed${seed}.log 2>&1
    done
  ) &
  i=$((i + 1))
done
wait
