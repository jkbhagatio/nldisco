#!/usr/bin/env bash
#SBATCH --job-name=nldisco-training-times
#SBATCH --partition=gpu_branco
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --output=/nfs/nhome/live/jbhagat/nldisco/experiments/churchland/outputs/training-time-%j.log
set -euo pipefail

cd /nfs/nhome/live/jbhagat/nldisco
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"

# CUDA sees the allocated device after Slurm's cgroup index remapping.
NLDISCO_TIMING_GPU_UUID="$(uv run --no-sync python -c '
import torch
assert torch.cuda.device_count() == 1, "Expected exactly one allocated GPU"
print("GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-"))
')"
export NLDISCO_TIMING_GPU_UUID
export CUDA_VISIBLE_DEVICES="$NLDISCO_TIMING_GPU_UUID"
benchmark_output="$PWD/experiments/churchland/outputs/training_time_h100_${SLURM_JOB_ID}"

uv run --no-sync python -u -m experiments.churchland.training_time_benchmark run \
    --output "$benchmark_output"

