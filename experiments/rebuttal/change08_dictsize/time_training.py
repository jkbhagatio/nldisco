"""Controlled training-time measurement: one process, one GPU, sizes run back to back.

Times ``train_model`` (all updates plus the threshold-calibration pass) with CUDA
synchronisation, for the paper's exact schedule at each dictionary size; excludes data
loading, model construction, checkpointing and export (as in the paper's timing benchmark).

    CUDA_VISIBLE_DEVICES=<gpu> python -m experiments.rebuttal.change08_dictsize.time_training \
        --dataset macaque --dims 128 192 256 512 1024 --repeats 3 --output <json>
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

K = dict(macaque=16, aeon=12)


def gpu_busy() -> dict:
    """Snapshot of other processes' load on the visible GPU (contention is not controlled)."""
    out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout.splitlines()
    return dict(nvidia_smi=out)


def macaque_setup():
    from experiments.rebuttal.change08_dictsize import train_macaque as tm
    training = tm.training
    dataset, *_ = training.load_dataset()

    def run(d: int, seed: int) -> float:
        training.set_seed(seed)
        cfg = training.configuration(dataset.spike_counts.shape[1], d, K['macaque'], 2)
        loader = DataLoader(dataset, batch_size=1024, shuffle=True, generator=torch.Generator().manual_seed(seed))
        loss = training.LossConfig(timebin_weights=[1.] * 10, type='mse')
        train = training.TrainConfig(epochs=10, batch_size=1024, learning_rate=.005,
                                     log_frequency=max(1, len(loader) // 4),
                                     dead_feature_window=max(1, len(loader) // 3))
        model = training.build_sed(cfg).cuda()
        torch.cuda.synchronize()
        started = time.monotonic()
        training.train_model(model, loader, loss, train, log_wandb=False)
        torch.cuda.synchronize()
        return time.monotonic() - started
    return run


def aeon_setup():
    from experiments.aeon.sweep_20260921 import shared_training as st
    data = Path('/ceph/aeon/aeon/nldisco/aeon/sweeps/20260921_18h_s20/data')
    counts = np.load(data / 'counts.npy', mmap_mode='r')
    starts = np.load(data / 'valid_starts.npy', mmap_mode='r')
    valid = np.load(data / 'neural_valid.npy', mmap_mode='r')
    mean, std, _ = st.compute_statistics(counts, valid)

    def run(d: int, seed: int) -> float:
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg = st.make_config(d, K['aeon'], counts.shape[1])
        loss = st.LossConfig(timebin_weights=[1.] * 20, type='mse', dsed_loss_weight_map={d // 2: 1., d: 1.},
                             dead_feature_loss_weight=1.)
        train = st.TrainConfig(batch_size=1024, epochs=1, learning_rate=.005, use_lr_schedule=True, log_frequency=50)
        loader = st.WindowBatches(counts, starts, mean, std, 1024, 5000, seed)
        model = st.build_sed(cfg).cuda()
        torch.cuda.synchronize()
        started = time.monotonic()
        st.train_model(model, loader, loss, train)
        torch.cuda.synchronize()
        return time.monotonic() - started
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('macaque', 'aeon'), required=True)
    parser.add_argument('--dims', type=int, nargs='+', required=True)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run = macaque_setup() if args.dataset == 'macaque' else aeon_setup()
    records = []
    for rep in range(args.repeats):
        for d in args.dims:
            before = gpu_busy()
            seconds = run(d, seed=100 + rep)
            records.append(dict(D=d, repeat=rep, seconds=seconds, before=before, gpu=torch.cuda.get_device_name(0)))
            print(json.dumps(records[-1]), flush=True)
            args.output.write_text(json.dumps(records, indent=1))


if __name__ == '__main__':
    main()
