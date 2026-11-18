"""Fresh-process H100 timing of the selected ST-NDT model (five seeds).

Follows `experiments/churchland/training_time_benchmark.py`: one allocated H100 pinned by
UUID, five sequential fits in separate processes (seeds 0-4), four CPU threads, no
experiment tracking, GPU occupancy sampled every two seconds and any competing compute
process invalidating the run. The measured interval is the CUDA-synchronised
optimisation time of the ten training epochs; loading, model construction, checkpoint
writing and export are excluded.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from beartype import beartype

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GPU = os.environ.get("NLDISCO_TIMING_GPU_UUID", "")
SEEDS = tuple(range(5))


@beartype
def device_pids() -> list:
    """Compute processes on the benchmark GPU only."""
    output = subprocess.check_output([
        "nvidia-smi", "-i", GPU, "--query-compute-apps=pid", "--format=csv,noheader,nounits"
    ], text=True)
    return [int(line.strip()) for line in output.splitlines() if line.strip()]


@beartype
def selected_config(config_id: Optional[str] = None) -> dict:
    """Read the frozen sweep choice (or a named setting); timing never selects a model."""
    from experiments.rebuttal.change01_stndt.macaque import configurations

    chosen = config_id or json.loads((HERE / "outputs/selection.json").read_text())["selected"]
    return next(c for c in configurations() if c["config_id"] == chosen)


@beartype
def worker(seed: int, folder: Path, config_id: Optional[str] = None) -> None:
    """One complete fit of the selected setting on every window."""
    import torch

    from experiments.rebuttal.change01_stndt.macaque import load_inputs
    from experiments.rebuttal.change01_stndt.stndt import Hyperparameters, train_and_export

    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU or torch.cuda.device_count() != 1:
        raise RuntimeError("Pin the benchmark H100 by UUID")
    if "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This benchmark requires an H100")
    torch.set_num_threads(4)
    config = selected_config(config_id)
    started = datetime.now(timezone.utc).isoformat()
    counts, anchors, indices, _ = load_inputs()
    hp = Hyperparameters(seed=seed, **{k: v for k, v in config.items() if k != "config_id"})
    summary = train_and_export(counts, indices, folder, hp, anchors=anchors,
                               provenance=dict(config_id=config["config_id"], mode="timing"))
    recorded = json.loads((folder / "config.json").read_text())
    result = dict(
        status="complete", method="stndt", seed=seed, pid=os.getpid(),
        config_id=config["config_id"], train_seconds=summary["train_seconds"],
        total_seconds=summary["seconds"],
        export_seconds=summary["extraction_and_reload_seconds"],
        calibration_seconds=0., updates=recorded["total_updates"],
        training_examples=hp.epochs * len(indices), dimension=summary["dimensions"],
        epochs=hp.epochs, windows=len(indices), batch_size=hp.batch_size,
        epoch_definition="exhaustive shuffled passes", started_utc=started,
        finished_utc=datetime.now(timezone.utc).isoformat(), gpu_uuid=GPU,
        gpu_name=torch.cuda.get_device_name(0), host=socket.gethostname(),
        peak_allocated_bytes=summary["allocated_peak_bytes"],
        parameter_count=recorded["parameters"],
        trainable_parameter_count=recorded["parameters"], torch_version=torch.__version__,
        cuda_version=torch.version.cuda, python_environment=sys.executable, cpu_threads=4)
    (folder / "timing.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


@beartype
def run(output: Path, config_id: Optional[str] = None) -> None:
    """Launch five nonoverlapping child processes on the requested H100."""
    if not GPU:
        raise RuntimeError("Set NLDISCO_TIMING_GPU_UUID to the allocated H100")
    if device_pids():
        raise RuntimeError("Benchmark GPU is occupied; no process is stopped automatically")
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, WANDB_MODE="disabled",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    rows = []
    for order, seed in enumerate(SEEDS, 1):
        if device_pids():
            raise RuntimeError("Benchmark GPU acquired another job; stop before the next fit")
        folder = output / f"{order:02d}_stndt_seed{seed}"
        command = [sys.executable, "-u", "-m", "experiments.rebuttal.change01_stndt.timing",
                   "worker", "--output", str(folder), "--seed", str(seed),
                   *(("--config-id", config_id) if config_id else ())]
        samples = []
        with (output / f"{folder.name}.log").open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT)
            while child.poll() is None:
                samples.append(dict(time=time.time(), pids=device_pids()))
                time.sleep(2)
        (output / f"{order:02d}_occupancy.json").write_text(json.dumps(dict(samples=samples)))
        if child.returncode:
            raise RuntimeError(f"Fit failed ({child.returncode}); see {folder.name}.log")
        row = json.loads((folder / "timing.json").read_text())
        row["foreign_gpu_pids"] = sorted({p for s in samples for p in s["pids"]} - {row["pid"]})
        row["order"] = order
        rows.append(row)
        (output / "results.json").write_text(json.dumps(dict(
            runs=rows, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            protocol="experiments/churchland/TRAINING_TIME_BENCHMARK.md"), indent=2) + "\n")
        if row["foreign_gpu_pids"]:
            raise RuntimeError("GPU contention detected; timings must not be reported")
        print(f"Completed {order}/5: {row['train_seconds']:.3f} s", flush=True)
    print(f"Complete: {output}", flush=True)


@beartype
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "worker"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=0)
    parser.add_argument("--config-id", help="default: the frozen sweep selection")
    args = parser.parse_args()
    if args.stage == "worker":
        worker(args.seed, args.output, args.config_id)
    else:
        run(args.output, args.config_id)


if __name__ == "__main__":
    main()
