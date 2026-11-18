"""ST-NDT on Aeon windows through change 1's dataset-agnostic runner.

Run with change 1's overlay interpreter (see its RUNNER_READY.json); needs
`experiments/rebuttal/change01_stndt` from branch rebuttal/c01-stndt.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from beartype import beartype

from experiments.rebuttal.change03_discovery import common, data

CONFIGURATIONS = {  # contrastive weight and learning rate; other settings as in change 1
    "cw0p01_lr0p001": dict(contrast_weight=.01, learning_rate=.001),
    "cw0p1_lr0p001": dict(contrast_weight=.1, learning_rate=.001),
    "cw1_lr0p001": dict(contrast_weight=1., learning_rate=.001),
    "cw0p1_lr0p003": dict(contrast_weight=.1, learning_rate=.003)}
EPOCHS = 3  # whole epochs of the fit windows: 6,405 updates of 1,024 windows


@beartype
def run(args: argparse.Namespace) -> None:
    """Train one configuration on fit windows and export every window."""
    from experiments.rebuttal.change01_stndt.stndt import Hyperparameters, train_and_export
    output = Path(args.output)
    started = time.monotonic()
    common.cap_gpu_memory(args.gpu_memory_gb)
    session = data.load_neural(Path(args.data))
    indices = session.starts[:, None] + np.arange(data.SEQ_LEN)
    hp = Hyperparameters(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
                         **CONFIGURATIONS[args.config])
    summary = train_and_export(
        np.asarray(session.counts, dtype=np.int64), indices, output, hp,
        train_mask=session.part == data.FIT, validation_mask=session.part == data.VALIDATION,
        device=args.device, provenance=dict(config_id=args.config, change="change03_discovery"))
    (output / "representation.npy").symlink_to("activations.npy")
    common.write_json(output / "COMPLETE.json", dict(
        elapsed_seconds=time.monotonic() - started,
        validation_loss=summary["validation"]["masked_nll"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", choices=tuple(CONFIGURATIONS), required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=common.BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-gb", type=float, default=16.)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
