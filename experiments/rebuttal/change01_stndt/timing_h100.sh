#!/usr/bin/env bash
#SBATCH --job-name=stndt-training-times
#SBATCH --partition=gpu_lowp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH --output=/ceph/aeon/aeon/nldisco/rebuttal_20261118/change01_stndt/logs/training-time-%j.log
# Same protocol as experiments/churchland/training_time_h100.sh, for the selected ST-NDT
# model. Submit from the worktree root: sbatch experiments/rebuttal/change01_stndt/timing_h100.sh
set -euo pipefail

CEPH=/ceph/aeon/aeon/nldisco/rebuttal_20261118/change01_stndt
STNDT_PY="$CEPH/stndt-venv/bin/python"
MAIN=/nfs/nhome/live/jbhagat/nldisco
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled

# CUDA sees the allocated device after Slurm's cgroup index remapping.
NLDISCO_TIMING_GPU_UUID="$("$STNDT_PY" -c '
import torch
assert torch.cuda.device_count() == 1, "Expected exactly one allocated GPU"
print("GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-"))
')"
export NLDISCO_TIMING_GPU_UUID
export CUDA_VISIBLE_DEVICES="$NLDISCO_TIMING_GPU_UUID"
output="$CEPH/training_time_h100_${SLURM_JOB_ID}"

"$STNDT_PY" -u -m experiments.rebuttal.change01_stndt.timing run --output "$output"

# Reference: one fit of the paper's NLDisco timing worker on this same GPU, to relate
# this allocation to the existing fifteen-fit benchmark (not reported as a new result).
( cd "$MAIN" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$MAIN/src:$MAIN" "$MAIN/.venv/bin/python" -u -m \
    experiments.churchland.training_time_benchmark worker \
    --output "$output/reference_nldisco_seed0" --method nldisco --seed 0 ) \
    > "$output/reference_nldisco_seed0.log" 2>&1 || echo "reference fit failed" >&2
