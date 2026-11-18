#!/usr/bin/env bash
# Usage: launch_aeon.sh GPU_INDEX SEED D [D ...]  -- trains Aeon (k=12, 5000 steps, batch 1024) runs sequentially.
set -u
gpu=$1; seed=$2; shift 2
R=/ceph/aeon/aeon/nldisco/rebuttal_20261118/change08_dictsize/runs
DATA=/ceph/aeon/aeon/nldisco/aeon/sweeps/20260921_18h_s20/data
PY=/nfs/nhome/live/jbhagat/nldisco/.venv/bin/python
cd /nfs/nhome/live/jbhagat/nldisco-worktrees/change08_dictsize
export WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=$gpu
for d in "$@"; do
  out=$R/aeon/d${d}_k12_seed${seed}
  [ -e "$out" ] && continue
  $PY -m experiments.aeon.sweep_20260921.shared_training --data $DATA --output $out \
    --d $d --k 12 --steps 5000 --batch-size 1024 --seed $seed --gpu-memory-gb 28 \
    > $R/logs/aeon_d${d}_s${seed}.log 2>&1
done
