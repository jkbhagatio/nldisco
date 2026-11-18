#!/usr/bin/env bash
# Usage: launch_timing.sh GPU DATASET REPEATS  -- controlled timing pass, sizes run back to back on one GPU.
set -u
gpu=$1; dataset=$2; reps=$3
R=/ceph/aeon/aeon/nldisco/rebuttal_20261118/change08_dictsize
cd /nfs/nhome/live/jbhagat/nldisco-worktrees/change08_dictsize
export WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=$gpu
/nfs/nhome/live/jbhagat/nldisco/.venv/bin/python -m experiments.rebuttal.change08_dictsize.time_training \
  --dataset $dataset --dims 128 192 256 512 1024 --repeats $reps --output $R/timing/$dataset.json \
  > $R/runs/logs/timing_$dataset.log 2>&1
