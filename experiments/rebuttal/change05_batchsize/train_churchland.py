"""Retrain the Section 4.2 NLDisco model (D=192, k=16, two levels) at another batch size.

Everything but the batch size and seed follows ``s10_endpoint_v1/nldisco.py``: same
architecture, data, ten shuffled epochs (so every batch size sees the same number of
training examples), Adam at 0.005 with the warmup/cosine schedule over all updates,
equal level weights. The dead-latent window and logging interval keep the runner's
fixed fractions of an epoch (one third and one quarter). Writes only to a new folder.

Run from the worktree root:
    CUDA_VISIBLE_DEVICES=<gpu> python -m experiments.rebuttal.change05_batchsize.train_churchland \
        --batch-size 512 --seed 11 --output <new folder>
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from beartype import beartype
from torch.utils.data import DataLoader

from experiments.churchland.sweeps.s10_endpoint_v1 import nldisco as runner
from nldisco.config import LossConfig, TrainConfig
from nldisco.model import build_sed
from nldisco.train import train_model
from nldisco.util import set_seed

DIMENSION, BUDGET, LEVELS, EPOCHS = 192, 16, 2, 10
# Git-ignored recording inputs live in the main checkout, which is read-only here.
MAIN_CHECKOUT = Path('/nfs/nhome/live/jbhagat/nldisco')


@beartype
def training_config(batch_size: int, n_batches: int) -> TrainConfig:
    """Training settings of the paper model; epoch-fraction logging and dead window."""
    return TrainConfig(epochs=EPOCHS, batch_size=batch_size, learning_rate=.005,
                       log_frequency=max(1, n_batches // 4),
                       dead_feature_window=max(1, n_batches // 3))


@beartype
def total_variance_by_bin(dataset, batch_size: int = 4096) -> np.ndarray:
    """Per-bin variance of the standardized inputs about their per-unit means.

    Returns the mean over units, for the ten bins of a window; the reconstruction
    variance explained is ``1 - mse / variance`` bin by bin.
    """
    total, squared, n = 0., 0., 0
    for batch in DataLoader(dataset, batch_size=batch_size):
        values = batch.values.double()
        total = total + values.sum(0)
        squared = squared + values.square().sum(0)
        n += len(values)
    variance = (squared - total.square() / n) / n
    return variance.mean(1).numpy()


@beartype
def run(batch_size: int, seed: int, output: Path) -> dict:
    """Train, export both levels and write summary.json."""
    output.mkdir(parents=True, exist_ok=False)
    runner.ROOT = MAIN_CHECKOUT
    dataset, anchors, scale, mean = runner.load_dataset()
    set_seed(seed)
    cfg = runner.configuration(dataset.spike_counts.shape[1], DIMENSION, BUDGET, LEVELS)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    loss_cfg = LossConfig(timebin_weights=[1.] * 10, type='mse')
    training = training_config(batch_size, len(loader))
    record = dict(batch_size=batch_size, seed=seed, updates=EPOCHS * len(loader),
                  examples=EPOCHS * len(dataset), sed=asdict(cfg), training=asdict(training))
    record['sed']['dtype'] = str(cfg.dtype)
    (output / 'config.json').write_text(json.dumps(record, indent=2))
    started = time.monotonic()
    model = build_sed(cfg).cuda()
    history = train_model(model, loader, loss_cfg, training, log_wandb=False)
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'config': record}, output / 'final_model.pt')
    (output / 'history.json').write_text(json.dumps(asdict(history), indent=2))
    levels = runner.export(model, dataset, 1024, anchors, output)
    variance = total_variance_by_bin(dataset)
    for level, metrics in levels.items():
        mse = np.asarray(metrics['mse_by_bin'])
        metrics['variance_by_bin'] = variance.tolist()
        metrics['variance_explained'] = float(1. - mse.mean() / variance.mean())
        metrics['dead_latents'] = int(round(metrics['dead_feature_fraction'] * int(level)))
    summary = dict(status='complete', batch_size=batch_size, seed=seed,
                   updates=record['updates'], examples=record['examples'],
                   training_seconds=seconds, levels=levels)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Requires a GPU; set CUDA_VISIBLE_DEVICES')
    torch.set_num_threads(4)
    run(args.batch_size, args.seed, args.output)


if __name__ == '__main__':
    main()
