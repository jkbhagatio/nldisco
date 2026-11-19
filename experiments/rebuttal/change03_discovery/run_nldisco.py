"""NLDisco with the Section 4.3 settings, fitted on the fit portion only."""
from __future__ import annotations

import argparse
import dataclasses
import os
import time
from pathlib import Path

import numpy as np
import torch
from beartype import beartype

from experiments.aeon.sweep_20260921 import shared_training as shared
from experiments.rebuttal.change03_discovery import common, data
from nldisco.config import LossConfig, TrainConfig
from nldisco.model.sed import build_sed
from nldisco.train import train_model


@beartype
def run(args: argparse.Namespace) -> None:
    """Train on fit windows, calibrate thresholds on them, export every window."""
    output = common.new_output(Path(args.output))
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        common.cap_gpu_memory(args.gpu_memory_gb)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    session = data.load_neural(Path(args.data))
    fit_starts = session.starts[session.part == data.FIT]
    mean, std, n_valid = shared.compute_statistics(session.counts, data.fit_bins(session))
    np.savez(output / "normalization.npz", mean=mean, std=std, n_valid=n_valid)
    config = shared.make_config(args.d, args.k, session.counts.shape[1])
    loss_config = LossConfig(timebin_weights=[1.] * 20, type="mse",
                             dsed_loss_weight_map={args.d // 2: 1., args.d: 1.},
                             dead_feature_loss_weight=1.)
    train_config = TrainConfig(batch_size=args.batch_size, epochs=1, learning_rate=.005,
                               use_lr_schedule=True, log_frequency=50)
    loader = shared.WindowBatches(session.counts, fit_starts, mean, std, args.batch_size,
                                  args.steps, args.seed)
    architecture = dataclasses.asdict(config)
    architecture["dtype"] = str(architecture["dtype"])
    common.write_json(output / "config.json", {
        "architecture": architecture, "loss": dataclasses.asdict(loss_config),
        "training": dataclasses.asdict(train_config), "args": vars(args),
        "fit_windows": len(fit_starts), "all_windows": len(session.starts),
        "scope": "fit portion of the discovery part only; no behavior labels",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")})
    model = build_sed(config).to(device)
    result = train_model(model, loader, loss_config, train_config)
    common.write_json(output / "training_history.json", dataclasses.asdict(result))
    torch.save({"state_dict": model.state_dict(), "architecture": architecture,
                "seed": args.seed, "steps": args.steps}, output / "checkpoint.pt")
    everything = shared.WindowBatches(session.counts, session.starts, mean, std,
                                      args.batch_size, 1, args.seed)
    metrics = shared.export(model, everything, output)
    link = output / "representation.npy"
    if not link.exists():
        link.symlink_to("activations.npy")
    metrics.update(elapsed_seconds=time.monotonic() - started, method="nldisco",
                   train_seconds=None)
    common.write_json(output / "summary.json", metrics)
    common.write_json(output / "COMPLETE.json", {"elapsed_seconds": metrics["elapsed_seconds"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--d", type=int, default=common.LATENTS)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--steps", type=int, default=common.STEPS)
    parser.add_argument("--batch-size", type=int, default=common.BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-gb", type=float, default=8.)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
