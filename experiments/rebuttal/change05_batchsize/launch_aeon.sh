#!/usr/bin/env bash
# Launch the nine Aeon D=192, k=12 retrains (3 batch sizes x 3 seeds) with a fixed
# total number of training examples (steps * batch size = 5,120,000).
# Usage: bash launch_aeon.sh  (run from the worktree root)
set -u
PY=/nfs/nhome/live/jbhagat/nldisco/.venv/bin/python
OUT=/ceph/aeon/aeon/nldisco/rebuttal_20261118/change05_batchsize/aeon
DATA=/ceph/aeon/aeon/nldisco/aeon/sweeps/20260921_18h_s20/data
GPUS=(GPU-5a924816-0a8e-d1b5-086e-e98ffd4382bb GPU-4c0ffb35-cb66-c83a-3df3-833fd58876cd GPU-cbefc9af-e673-789a-7605-8209a24294fb)
export WANDB_MODE=disabled
mkdir -p "$OUT/logs"
i=0
for seed in 11 12 13; do
  gpu=${GPUS[$((i % 3))]}
  for bs in 512 1024 2048; do
    steps=$((5120000 / bs))
    name=bs${bs}_seed${seed}
    CUDA_VISIBLE_DEVICES=$gpu nohup $PY -m experiments.aeon.sweep_20260921.shared_training \
      --data $DATA --output $OUT/$name --d 192 --k 12 --steps $steps --batch-size $bs \
      --seed $seed --gpu-memory-gb 12 > $OUT/logs/$name.log 2>&1 &
  done
  i=$((i + 1))
done
wait
