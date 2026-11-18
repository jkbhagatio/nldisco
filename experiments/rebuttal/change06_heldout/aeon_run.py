"""Retrain the paper's Aeon models (D=192 or 256, k=12) on the training blocks only.

Uses the architecture, loss, optimizer, 5,000 updates, B=1024 and warmup/cosine schedule of
``experiments/aeon/sweep_20260921/shared_training.py``. Normalization statistics, the
training windows, the dead-feature bookkeeping and the activation-threshold calibration
pass see only windows lying wholly inside training blocks. Native sparse activations of
all non-crossing windows (training and held-out) are then exported.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
from beartype import beartype

from experiments.aeon.sweep_20260921.shared_training import (
    WindowBatches, compute_statistics, make_config)
from experiments.rebuttal.change06_heldout import aeon_split
from nldisco.config import LossConfig, TrainConfig
from nldisco.model.sed import build_sed
from nldisco.train import train_model

DATA = Path("/ceph/aeon/aeon/nldisco/aeon/sweeps/20260921_18h_s20/data")


@beartype
def export_sparse(model: torch.nn.Module, loader: WindowBatches, path: Path, d: int) -> dict:
    """Stream native calibrated sparse endpoint codes of every loader window."""
    model.eval()
    device = next(model.parameters()).device
    n = len(loader.starts)
    sparse = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, d))
    active = np.zeros(d, dtype=np.int64)
    with torch.inference_mode():
        for offset in range(0, n, loader.batch_size):
            indices = np.arange(offset, min(n, offset + loader.batch_size))
            batch = loader.batch(indices)
            native = model(batch.values.to(device),
                           occurrence_mask=batch.occurrence_mask.to(device)).sparse_acts[d][:, -1].cpu().numpy()
            if not np.isfinite(native).all():
                raise ValueError("Nonfinite activation export")
            sparse[indices] = native
            active += (native > 0).sum(0)
    sparse.flush()
    return dict(n_windows=n, mean_l0=float(active.sum() / n), dead_latents=np.flatnonzero(active == 0).tolist())


@beartype
def run(d: int, k: int, seed: int, output: Path, steps: int, batch_size: int, memory_gb: float,
        subsample: int = 1) -> None:
    """Train on training blocks, calibrate on them, export all windows, verify reload."""
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(memory_gb * 2**30 / total, 1.), 0)
    counts = np.load(DATA / "counts.npy", mmap_mode="r")
    starts = np.load(DATA / "valid_starts.npy")[::subsample]  # subsample > 1 only for smoke tests
    valid = np.load(DATA / "neural_valid.npy")
    held = aeon_split.heldout_bins(len(counts))
    label = aeon_split.partition_windows(starts, held)
    train_starts, eval_starts = starts[label == 0], starts[label >= 0]
    summary = aeon_split.write_split(output / "split.json", starts)
    mean, std, n_valid = compute_statistics(counts, valid & ~held)
    np.savez(output / "normalization.npz", mean=mean, std=std, n_valid=n_valid)
    np.save(output / "eval_endpoint_bins.npy", eval_starts + 19)
    np.save(output / "eval_part.npy", label[label >= 0])
    config = make_config(d, k, counts.shape[1])
    loss_config = LossConfig(timebin_weights=[1.] * 20, type="mse",
                             dsed_loss_weight_map={d // 2: 1., d: 1.}, dead_feature_loss_weight=1.)
    train_config = TrainConfig(batch_size=batch_size, epochs=1, learning_rate=.005,
                               use_lr_schedule=True, log_frequency=50)
    architecture = dataclasses.asdict(config)
    architecture["dtype"] = str(architecture["dtype"])
    (output / "config.json").write_text(json.dumps(dict(
        architecture=architecture, loss=dataclasses.asdict(loss_config),
        training=dataclasses.asdict(train_config), d=d, k=k, seed=seed, steps=steps,
        batch_size=batch_size, split=summary,
        protocol="fit, normalization and threshold calibration on training blocks only"), indent=2))
    loader = WindowBatches(counts, train_starts, mean, std, batch_size, steps, seed)
    model = build_sed(config).to("cuda")
    result = train_model(model, loader, loss_config, train_config)
    (output / "training_history.json").write_text(json.dumps(dataclasses.asdict(result)))
    torch.save({"state_dict": model.state_dict(), "architecture": architecture, "seed": seed,
                "steps": steps}, output / "checkpoint.pt")
    seen = loader.seen
    exposure = dict(unique_windows=int((seen > 0).sum()), train_windows=len(train_starts),
                    min_visits=int(seen.min()), max_visits=int(seen.max()))
    evaluation = WindowBatches(counts, eval_starts, mean, std, batch_size, steps, seed)
    metrics = export_sparse(model, evaluation, output / "activations.npy", d)
    restored = build_sed(config)
    restored.load_state_dict(torch.load(output / "checkpoint.pt", map_location="cpu",
                                        weights_only=True)["state_dict"])
    restored = restored.to("cuda").eval()
    probe = evaluation.batch(np.arange(32))
    with torch.inference_mode():
        a = model(probe.values.cuda()).sparse_acts[d].cpu()
        b = restored(probe.values.cuda()).sparse_acts[d].cpu()
    if not torch.equal(a, b):
        raise AssertionError("Checkpoint reload did not reproduce native sparse activations")
    thresholds = {str(level): float(model.sparsifier.inference_threshold(level)) for level in (d // 2, d)}
    metrics.update(exposure=exposure, thresholds=thresholds, seconds=time.monotonic() - started,
                   gpu_peak_gb=torch.cuda.max_memory_allocated() / 2**30)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "COMPLETE.json").write_text(json.dumps(dict(d=d, k=k, seed=seed)))
    print(json.dumps(metrics), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--gpu-memory-gb", type=float, default=8.)
    parser.add_argument("--subsample", type=int, default=1, help="smoke tests only")
    args = parser.parse_args()
    run(args.d, args.k, args.seed, args.output, args.steps, args.batch_size, args.gpu_memory_gb,
        args.subsample)


if __name__ == "__main__":
    main()
