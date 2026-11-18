"""Train the Section 4.2 NLDisco model (k=16, two Matryoshka levels) at another dictionary size.

Reuses the frozen runner of ``s10_endpoint_v1`` (architecture, data, optimiser, export)
and changes only the largest dictionary size ``D`` (levels D/2 and D) and the seed
(initialisation and shuffle), exactly as the paper's five-seed D=192 replication did.
"""
import argparse
import hashlib
import json
import socket
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from beartype import beartype
from torch.utils.data import DataLoader

from experiments.churchland.sweeps.s10_endpoint_v1 import nldisco as training

K = 16
MAIN_CHECKOUT = Path('/nfs/nhome/live/jbhagat/nldisco')  # git-ignored data lives here
training.ROOT = MAIN_CHECKOUT


@beartype
def run(dimension: int, seed: int, output: Path, k: int = K) -> dict:
    """Train, export both levels, verify an exact checkpoint reload; return the summary."""
    config_id = f'd{dimension}_k{k}_l2_seed{seed}'
    folder = output / config_id
    folder.mkdir(parents=True, exist_ok=False)
    dataset, anchors, scale, mean = training.load_dataset()
    training.set_seed(seed)
    cfg = training.configuration(dataset.spike_counts.shape[1], dimension, k, 2)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    loss = training.LossConfig(timebin_weights=[1.] * 10, type='mse')
    train = training.TrainConfig(epochs=10, batch_size=1024, learning_rate=.005,
                                 log_frequency=max(1, len(loader) // 4),
                                 dead_feature_window=max(1, len(loader) // 3))
    config = dict(config_id=config_id, seed=seed, dimension=dimension, k=k, windows=len(dataset),
                  loss=asdict(loss), training=asdict(train), host=socket.gethostname(),
                  gpu_name=torch.cuda.get_device_name(0),
                  changed='Largest dictionary size and seed only; all else as s10_endpoint_v1/nldisco.py')
    (folder / 'config.json').write_text(json.dumps(config, indent=2))
    np.savez(folder / 'preprocessing.npz', scale=scale, mean=mean)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = training.build_sed(cfg).cuda()
    history = training.train_model(model, loader, loss, train, log_wandb=False)
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    torch.save(dict(state_dict={n: v.detach().cpu() for n, v in model.state_dict().items()},
                    config=config), folder / 'final_model.pt')
    (folder / 'history.json').write_text(json.dumps(asdict(history), indent=2))
    levels = training.export(model, dataset, 1024, anchors, folder)
    restored = training.build_sed(cfg).cuda()
    restored.load_state_dict(torch.load(folder / 'final_model.pt', weights_only=True)['state_dict'])
    restored.eval()
    probe = next(iter(DataLoader(dataset, batch_size=32)))
    with torch.inference_mode():
        a = model(probe.values.cuda(), occurrence_mask=probe.occurrence_mask.cuda())
        b = restored(probe.values.cuda(), occurrence_mask=probe.occurrence_mask.cuda())
        for level in (dimension // 2, dimension):
            torch.testing.assert_close(a.sparse_acts[level], b.sparse_acts[level], rtol=0, atol=0)
    summary = dict(config_id=config_id, seed=seed, dimension=dimension, status='complete',
                   levels=levels, training_seconds_including_calibration=seconds,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   gpu_name=torch.cuda.get_device_name(0), host=socket.gethostname(),
                   updates=len(loader) * train.epochs)
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dims', type=int, nargs='+', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    for dimension in args.dims:
        for seed in args.seeds:
            if not (args.output / f'd{dimension}_k{K}_l2_seed{seed}').exists():
                run(dimension, seed, args.output)


if __name__ == '__main__':
    main()
