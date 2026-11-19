#!/usr/bin/env bash
# Launch one runner on one GPU of the current host, detached, with W&B off.
# Usage: launch.sh <gpu-index> <log-file> <interpreter> <module> [runner args...]
set -euo pipefail
gpu=$1; log=$2; interpreter=$3; module=$4; shift 4
cd "$(dirname "$0")/../../.."
export WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1
nohup "$interpreter" -m "$module" "$@" > "$log" 2>&1 &
echo "started $module on GPU $gpu (pid $!) -> $log"
