"""Training-time micro-benchmark that is robust to other users' load on a shared GPU.

Wall-clock times of full runs on shared GPUs vary several-fold with other jobs' load. Here, for
every dictionary size, ``train_model`` (the paper's training loop, including the dead-feature
auxiliary loss and optimiser) is timed on a short schedule of ``n`` updates, and the threshold
calibration pass (a no-grad forward over the loader) is timed separately. Sizes are interleaved
over many repeats and the **minimum** per size is kept, which approximates uncontended speed:

    update_seconds = (T_train(n) - T_forward(n)) / n      calibration_seconds = T_forward(n) / n
    schedule_seconds = n_updates * update_seconds + n_calibration_batches * calibration_seconds

with the paper's schedule (macaque: 1,280 updates, 128 calibration batches; Aeon: 5,000 updates,
3,163 calibration batches). Data loading, model construction, checkpointing and export are excluded.

    CUDA_VISIBLE_DEVICES=<gpu> python -m experiments.rebuttal.change08_dictsize.time_training \
        --dataset macaque --dims 128 192 256 512 1024 --repeats 10 --output <json>
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

K = dict(macaque=16, aeon=12)
SCHEDULE = dict(macaque=dict(updates=1280, calibration=128), aeon=dict(updates=5000, calibration=3163))


def gpu_state() -> list:
    """Utilisation/memory of all visible GPUs at the start of a measurement."""
    return subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True).stdout.splitlines()


def sync_time(fn) -> float:
    """Seconds taken by ``fn`` with CUDA synchronisation on both sides."""
    torch.cuda.synchronize()
    started = time.monotonic()
    fn()
    torch.cuda.synchronize()
    return time.monotonic() - started


def forward_pass(model, loader, device) -> None:
    """What the threshold calibration does: eval-mode no-grad forwards over every batch."""
    model.eval()
    with torch.no_grad():
        for batch in loader:
            model(batch.values.to(device), occurrence_mask=batch.occurrence_mask.to(device))
    model.train()


def macaque_setup(n: int):
    from experiments.rebuttal.change08_dictsize import train_macaque as tm
    training = tm.training
    dataset, *_ = training.load_dataset()

    def make(d: int, seed: int):
        training.set_seed(seed)
        cfg = training.configuration(dataset.spike_counts.shape[1], d, K['macaque'], 2)
        subset = Subset(dataset, range(n * 1024))
        subset.spike_counts = dataset.spike_counts
        loader = DataLoader(dataset if False else subset, batch_size=1024, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        loss = training.LossConfig(timebin_weights=[1.] * 10, type='mse')
        train = training.TrainConfig(epochs=1, batch_size=1024, learning_rate=.005,
                                     log_frequency=max(1, n // 4), dead_feature_window=max(1, n // 3))
        return training.build_sed(cfg).cuda(), loader, loss, train, training.train_model
    return make


def aeon_setup(n: int):
    from experiments.aeon.sweep_20260921 import shared_training as st
    data = Path('/ceph/aeon/aeon/nldisco/aeon/sweeps/20260921_18h_s20/data')
    counts = np.load(data / 'counts.npy', mmap_mode='r')
    starts = np.load(data / 'valid_starts.npy', mmap_mode='r')
    valid = np.load(data / 'neural_valid.npy', mmap_mode='r')
    mean, std, _ = st.compute_statistics(counts, valid)

    class Short(st.WindowBatches):
        """Paper loader whose calibration pass covers only the first ``n`` batches."""

        def __iter__(self):
            if self.iterations >= 1:
                self.iterations += 1
                order = np.random.default_rng(self.seed + 1).permutation(len(self.starts))[:n * self.batch_size]
                for offset in range(0, len(order), self.batch_size):
                    yield self.batch(order[offset:offset + self.batch_size])
                return
            yield from super().__iter__()

    def make(d: int, seed: int):
        torch.manual_seed(seed)
        cfg = st.make_config(d, K['aeon'], counts.shape[1])
        loss = st.LossConfig(timebin_weights=[1.] * 20, type='mse', dsed_loss_weight_map={d // 2: 1., d: 1.},
                             dead_feature_loss_weight=1.)
        train = st.TrainConfig(batch_size=1024, epochs=1, learning_rate=.005, use_lr_schedule=True, log_frequency=50)
        loader = Short(counts, starts, mean, std, 1024, n, seed)
        return st.build_sed(cfg).cuda(), loader, loss, train, st.train_model
    return make


def model_only(dataset: str, dims: list, repeats: int, steps: int = 30) -> list:
    """GPU compute per optimisation step on one preloaded batch (no data loading, no aux loss).

    Mirrors the core of the training step: forward, summed per-level MSE, backward, dictionary
    constraint, Adam step, dictionary normalisation. Minimum over repeats, sizes interleaved.
    """
    if dataset == 'macaque':
        from experiments.rebuttal.change08_dictsize import train_macaque as tm
        training = tm.training
        data, *_ = training.load_dataset()
        batch = next(iter(DataLoader(data, batch_size=1024)))
        build = lambda d: training.build_sed(training.configuration(data.spike_counts.shape[1], d, K['macaque'], 2))
    else:
        from experiments.aeon.sweep_20260921 import shared_training as st
        base = Path('/ceph/aeon/aeon/nldisco/aeon/sweeps/20260921_18h_s20/data')
        counts = np.load(base / 'counts.npy', mmap_mode='r')
        starts = np.load(base / 'valid_starts.npy', mmap_mode='r')
        mean, std, _ = st.compute_statistics(counts, np.load(base / 'neural_valid.npy', mmap_mode='r'))
        batch = st.WindowBatches(counts, starts, mean, std, 1024, 1, 0).batch(np.arange(1024))
        build = lambda d: st.build_sed(st.make_config(d, K['aeon'], counts.shape[1]))
    values, mask = batch.values.cuda(), batch.occurrence_mask.cuda()
    out = []
    for rep in range(repeats + 1):
        for d in dims:
            torch.manual_seed(rep)
            model = build(d).cuda()
            opt = torch.optim.Adam(model.parameters(), lr=.005)

            def run():
                for _ in range(steps):
                    opt.zero_grad()
                    result = model(values, occurrence_mask=mask)
                    loss = sum((r - values).square().mean() for r in result.reconstructions.values())
                    loss.backward()
                    model.constrain_dictionary()
                    opt.step()
                    model.normalize_dictionary()
            run()  # warm-up
            seconds = sync_time(run)
            out.append(dict(D=d, repeat=rep, warmup=rep == 0, step_seconds=seconds / steps))
            del model, opt
            torch.cuda.empty_cache()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('macaque', 'aeon'), required=True)
    parser.add_argument('--dims', type=int, nargs='+', required=True)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--updates', type=int, default=40, help='updates (= calibration batches) per measurement')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-only', action='store_true', help='GPU compute per step on a preloaded batch')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.model_only:
        args.output.write_text(json.dumps(model_only(args.dataset, args.dims, args.repeats), indent=1))
        return
    n = args.updates
    make = (macaque_setup if args.dataset == 'macaque' else aeon_setup)(n)
    device = torch.device('cuda')
    records = []
    for rep in range(args.repeats + 1):  # repetition 0 is a discarded warm-up
        for d in args.dims:
            model, loader, loss, train, train_model = make(d, seed=100 + rep)
            state = gpu_state()
            t_train = sync_time(lambda: train_model(model, loader, loss, train))
            t_forward = sync_time(lambda: forward_pass(model, loader, device))
            records.append(dict(D=d, repeat=rep, warmup=rep == 0, train_with_calibration=t_train,
                                forward=t_forward, n=n, gpu_state=state, gpu=torch.cuda.get_device_name(0)))
            print(json.dumps({k: records[-1][k] for k in ('D', 'repeat', 'train_with_calibration', 'forward')}), flush=True)
            del model, loader
            torch.cuda.empty_cache()
        args.output.write_text(json.dumps(records, indent=1))


def estimate(records: list, dataset: str) -> dict:
    """Per-size schedule estimates (minimum over non-warm-up repeats) from the saved records."""
    out = {}
    plan = SCHEDULE[dataset]
    for d in sorted({r['D'] for r in records}):
        rs = [r for r in records if r['D'] == d and not r['warmup']]
        n = rs[0]['n']
        t_forward = min(r['forward'] for r in rs)
        update = max(min(r['train_with_calibration'] for r in rs) - t_forward, 0.) / n
        out[d] = dict(update_seconds=update, calibration_batch_seconds=t_forward / n,
                      schedule_seconds=plan['updates'] * update + plan['calibration'] * t_forward / n,
                      median_update_seconds=float(np.median([(r['train_with_calibration'] - r['forward']) / n for r in rs])))
    return out


if __name__ == '__main__':
    main()
