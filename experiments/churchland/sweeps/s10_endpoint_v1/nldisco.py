"""Prespecified endpoint NLDisco sweep; reads neural counts and trial IDs only."""

import argparse
import gc
import json
import os
import socket
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from beartype import beartype
from torch.utils.data import DataLoader

from experiments.churchland.scripts.train_nldisco import file_sha256, prepare_inputs
from nldisco.config import DecoderConfig, EncoderConfig, LossConfig, SedConfig, TrainConfig
from nldisco.data import SpikeWindowDataset
from nldisco.model import build_sed
from nldisco.train import train_model
from nldisco.util import set_seed

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path('/ceph/aeon/aeon/nldisco/churchland/sweeps/20260914_s10_endpoint_v1')
GPU = 'GPU-e04bd614-057a-08f7-450d-b1bfa27f297d'


@beartype
def configuration(neurons: int, dimension: int, budget: int, levels: int) -> SedConfig:
    """Build the frozen single-block, ten-bin endpoint model."""
    divisors = (2, 1) if levels == 2 else (4, 2, 1)
    return SedConfig(
        n_neurons=neurons, seq_len=10,
        dsed_topk_map={dimension // d: budget // d for d in divisors},
        encoder=EncoderConfig(type='TransformerWindow', shift_equivariant=True,
                              d_model=128, n_heads=4, n_layers=1,
                              d_feedforward=512, dropout=0.0, causal=True,
                              attention_radius=9),
        decoder=DecoderConfig(output_activation='none', temporal_kernel_len=10,
                              temporal_alignment='causal'),
    )


@beartype
def load_dataset() -> tuple:
    """Standardize all neural rows and retain every complete within-trial window."""
    folder = ROOT / 'experiments/churchland/data/nitschke_20090812'
    with np.load(folder / 'metadata.npz') as archive:
        trials = archive['trial_id']
    counts = np.load(folder / 'counts.npy').astype(np.float32)
    inputs, scale, mean = prepare_inputs(counts, trials, 0.0, 'unit_zscore')
    dataset = SpikeWindowDataset(torch.from_numpy(inputs), seq_len=10, stride=1,
                                 trial_ids=trials, occurrence_support=(9, 0))
    anchors = np.asarray(dataset.valid_starts, dtype=np.int64) + 9
    if len(anchors) != 130507 or not np.all(np.diff(anchors) > 0):
        raise AssertionError(f'Unexpected endpoint count/order: {len(anchors)}')
    return dataset, anchors, scale, mean


class MemoryGuard:
    """Track this process's full GPU allocation and enforce a decimal 40 GB ceiling."""

    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(
                    ['nvidia-smi', '-i', GPU, '--query-compute-apps=pid,used_gpu_memory',
                     '--format=csv,noheader,nounits'], capture_output=True, text=True,
                    check=True, timeout=4)
                used = sum(float(row.split(',')[1]) * 1024**2
                           for row in result.stdout.splitlines()
                           if int(row.split(',')[0]) == os.getpid())
                self.samples.append({'time': time.time(), 'gpu_process_bytes': used})
                if used > 40_000_000_000:
                    print(json.dumps({'event': 'memory_ceiling_exceeded', 'bytes': used}), flush=True)
                    os._exit(42)
            except (subprocess.SubprocessError, ValueError):
                pass
            self.stop.wait(1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)


@beartype
def benchmark(dataset: SpikeWindowDataset, folder: Path) -> int:
    """Compare the three approved batch sizes using the largest architecture."""
    records = []
    for batch_size in (256, 512, 1024):
        set_seed(0)
        model = build_sed(configuration(dataset.spike_counts.shape[1], 256, 48, 3)).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        batch = next(iter(DataLoader(dataset, batch_size=batch_size)))
        values, mask = batch.values.cuda(), batch.occurrence_mask.cuda()
        torch.cuda.reset_peak_memory_stats()
        with MemoryGuard() as memory:
            for step in range(25):
                if step == 5:
                    torch.cuda.synchronize()
                    started = time.monotonic()
                optimizer.zero_grad()
                output = model(values, occurrence_mask=mask)
                loss = sum((reconstruction - values).square().mean()
                           for reconstruction in output.reconstructions.values())
                loss.backward()
                model.constrain_dictionary()
                optimizer.step()
                model.normalize_dictionary()
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
        records.append(dict(batch_size=batch_size, examples_per_second=20 * batch_size / elapsed,
                            seconds=elapsed, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                            peak_process_bytes=max((s['gpu_process_bytes'] for s in memory.samples), default=0)))
        print(json.dumps({'event': 'benchmark', **records[-1]}), flush=True)
        del model, optimizer, batch, values, mask, output, loss
        gc.collect()
        torch.cuda.empty_cache()
    best = max(records, key=lambda row: row['examples_per_second'])['batch_size']
    (folder / 'benchmark.json').write_text(json.dumps({'records': records, 'frozen_batch_size': best}, indent=2))
    return best


@beartype
def export(model: torch.nn.Module, dataset: SpikeWindowDataset, batch_size: int,
           anchors: np.ndarray, folder: Path) -> dict:
    """Export each endpoint once and calculate neural-only reconstruction metrics."""
    arrays = {level: np.lib.format.open_memmap(folder / f'activations_level_{level}.npy',
              mode='w+', dtype=np.float32, shape=(len(dataset), level))
              for level in model.cfg.dsed_topk_map}
    squared = {level: torch.zeros(10, device='cuda', dtype=torch.float64) for level in arrays}
    position = 0
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size):
            n = len(batch.values)
            np.testing.assert_array_equal(batch.anchor_index.numpy(), anchors[position:position+n])
            values = batch.values.cuda()
            output = model(values, occurrence_mask=batch.occurrence_mask.cuda())
            for level in arrays:
                if torch.count_nonzero(output.sparse_acts[level][:, :9]):
                    raise AssertionError('Nonendpoint occurrence active')
                arrays[level][position:position+n] = output.sparse_acts[level][:, 9].cpu().numpy()
                squared[level] += (output.reconstructions[level] - values).double().square().sum((0, 2))
            position += n
    if position != len(anchors):
        raise AssertionError('Incomplete export')
    metrics = {}
    for level, array in arrays.items():
        array.flush()
        if not np.isfinite(array).all() or not torch.isfinite(squared[level]).all():
            raise FloatingPointError(f'Nonfinite export/reconstruction at level {level}')
        metrics[str(level)] = dict(mean_l0=float(np.count_nonzero(array, axis=1).mean()),
            dead_feature_fraction=float((~np.any(array > 0, axis=0)).mean()),
            mse_by_bin=(squared[level] / (len(dataset) * dataset.spike_counts.shape[1])).cpu().tolist(),
            inference_threshold=float(model.sparsifier.inference_threshold(level)))
    np.savez(folder / 'activation_index.npz', source_index=anchors,
             window_start=anchors - 9, feature_time_position=np.full(len(anchors), 9))
    return metrics


@beartype
def run_one(dataset: SpikeWindowDataset, anchors: np.ndarray, scale: np.ndarray,
            mean: np.ndarray, batch_size: int, dimension: int, budget: int, levels: int) -> None:
    """Train and persist one approved configuration, with final-only model saving."""
    import wandb

    config_id = f'd{dimension}_k{budget}_l{levels}_seed0'
    folder = OUTPUT / 'nldisco' / config_id
    folder.mkdir(parents=True, exist_ok=False)
    set_seed(0)
    cfg = configuration(dataset.spike_counts.shape[1], dimension, budget, levels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(0))
    loss_cfg = LossConfig(timebin_weights=[1.] * 10, type='mse')
    training = TrainConfig(epochs=10, batch_size=batch_size, learning_rate=.005,
                           log_frequency=max(1, len(loader) // 4),
                           dead_feature_window=max(1, len(loader) // 3))
    config = dict(config_id=config_id, seed=0, method='nldisco', sed=asdict(cfg),
                  loss=asdict(loss_cfg), training=asdict(training), windows=len(dataset),
                  preprocessing='all neural rows, unit_zscore, unsmoothed',
                  support=[9, 0], endpoint_position=9, bin_ms=50,
                  gpu_uuid=GPU, gpu_name=torch.cuda.get_device_name(0), host=socket.gethostname(),
                  pid=os.getpid(), checkpoint_policy='final only',
                  sources={str(p.relative_to(ROOT)): file_sha256(p) for p in
                           [Path(__file__), ROOT / 'experiments/churchland/scripts/train_nldisco.py',
                            *sorted((ROOT / 'src/nldisco').rglob('*.py'))]},
                  manifest_sha256=file_sha256(OUTPUT / 'manifest.json'))
    config['sed']['dtype'] = str(cfg.dtype)
    (folder / 'config.json').write_text(json.dumps(config, indent=2))
    (folder / 'runner_source.py').write_text(Path(__file__).read_text())
    np.savez(folder / 'preprocessing.npz', scale=scale, mean=mean)
    run = wandb.init(entity='jkbhagatio', project='NLDisco-paper', name=config_id,
                     tags=['nldisco', 's10_endpoint_v1'], config=config, dir=str(folder))
    (folder / 'wandb.json').write_text(json.dumps({'id': run.id, 'url': run.url}, indent=2))
    print(json.dumps({'event': 'training_start', 'config_id': config_id, 'pid': os.getpid(), 'wandb_id': run.id}), flush=True)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with MemoryGuard() as memory:
        model = build_sed(cfg).cuda()
        history = train_model(model, loader, loss_cfg, training, log_wandb=True)
        torch.cuda.synchronize()
        training_seconds = time.monotonic() - started
        torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'config': config}, folder / 'final_model.pt')
        (folder / 'history.json').write_text(json.dumps(asdict(history), indent=2))
        metrics = export(model, dataset, batch_size, anchors, folder)
        restored = build_sed(cfg).cuda()
        restored.load_state_dict(torch.load(folder / 'final_model.pt', weights_only=True)['state_dict'])
        restored.eval()
        probe = next(iter(DataLoader(dataset, batch_size=32)))
        with torch.inference_mode():
            original = model(probe.values.cuda(), occurrence_mask=probe.occurrence_mask.cuda())
            reloaded = restored(probe.values.cuda(), occurrence_mask=probe.occurrence_mask.cuda())
            for level in metrics:
                torch.testing.assert_close(original.sparse_acts[int(level)], reloaded.sparse_acts[int(level)])
                torch.testing.assert_close(original.reconstructions[int(level)], reloaded.reconstructions[int(level)])
        del restored, original, reloaded
    summary = dict(config_id=config_id, status='complete', levels=metrics, checkpoint_reload=True,
                   host=socket.gethostname(), pid=os.getpid(), gpu_uuid=GPU,
                   gpu_name=torch.cuda.get_device_name(0), wandb_id=run.id, wandb_url=run.url,
                   training_seconds_including_threshold_calibration=training_seconds,
                   total_seconds=time.monotonic()-started,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   peak_process_bytes=max((s['gpu_process_bytes'] for s in memory.samples), default=0))
    (folder / 'memory.json').write_text(json.dumps(memory.samples))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(list(history.loss), list(history.loss.values()), label='total')
    ax.plot(list(history.weighted_reconstruction), list(history.weighted_reconstruction.values()), label='largest-level MSE')
    ax.set(xlabel='Training step', ylabel='Loss')
    ax.legend()
    fig.savefig(folder / 'loss.png', dpi=160)
    plt.close(fig)
    run.summary.update(summary)
    artifact = wandb.Artifact(config_id, type='model')
    artifact.add_file(str(folder / 'final_model.pt'))
    artifact.add_file(str(folder / 'runner_source.py'))
    artifact.add_file(str(folder / 'config.json'))
    artifact.add_file(str(folder / 'preprocessing.npz'))
    run.log_artifact(artifact)
    run.finish()
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'event': 'complete', **summary}), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


@beartype
def main() -> None:
    """Benchmark or execute all twelve configurations after shared preflight."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark-only', action='store_true')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU or not torch.cuda.is_available():
        raise RuntimeError('Requires approved local GPU1 UUID exclusively')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(34_000_000_000 / torch.cuda.get_device_properties(0).total_memory)
    folder = OUTPUT / 'nldisco'
    folder.mkdir(parents=True, exist_ok=True)
    dataset, anchors, scale, mean = load_dataset()
    benchmark_path = folder / 'benchmark.json'
    batch_size = (json.loads(benchmark_path.read_text())['frozen_batch_size']
                  if benchmark_path.exists() else benchmark(dataset, folder))
    if args.benchmark_only:
        return
    # Root writes this marker only after verifying remote W&B project privacy.
    gate = json.loads((OUTPUT / 'manifest.json').read_text())
    if gate.get('wandb_privacy_verified') is not True:
        raise RuntimeError('Private W&B preflight not confirmed')
    with np.load(OUTPUT / 'shared/index.npz') as shared:
        np.testing.assert_array_equal(anchors, shared['anchor_index'])
    for dimension in (128, 256):
        for budget in (16, 32, 48):
            for levels in (2, 3):
                run_one(dataset, anchors, scale, mean, batch_size, dimension, budget, levels)


if __name__ == '__main__':
    main()
