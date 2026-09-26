"""Endpoint-aligned CEBRA-Time with explicit ten-bin neural windows."""
from __future__ import annotations

import argparse
import itertools
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, Int, jaxtyped


@beartype
def configurations() -> list[dict]:
    """Return the approved twelve configurations in stable array order."""
    return [dict(config_id=f"d{d}_t{str(t).replace('.', 'p')}_lag{lag}",
                 dimension=d, temperature=t, lag=lag)
            for d, t, lag in itertools.product((32, 64, 128), (0.3, 1.0), (1, 5))]


@jaxtyped(typechecker=beartype)
def pair_endpoints(endpoints: Int[np.ndarray, "windows"],
                   trials: Int[np.ndarray, "time"], lag: int) -> Int[np.ndarray, "pairs"]:
    """Find reference endpoints whose forward positive retains full trial context."""
    candidate = endpoints[endpoints + lag < len(trials)]
    valid = np.zeros(len(trials), dtype=bool)
    valid[endpoints] = True
    return candidate[(trials[candidate] == trials[candidate + lag]) & valid[candidate + lag]]


@jaxtyped(typechecker=beartype)
def windows(data: Float[torch.Tensor, "time neurons"],
            endpoints: Int[torch.Tensor, "batch"]) -> Float[torch.Tensor, "batch neurons 10"]:
    """Materialize exactly [t-9, ..., t], in encoder channel-first order."""
    indices = endpoints[:, None] + torch.arange(-9, 1, device=data.device)
    return rearrange(data[indices], "b t c -> b c t").contiguous()


@beartype
def encoder(neurons: int, dimension: int, device: str) -> torch.nn.Module:
    """Build the official CEBRA 0.4.0 offset10 encoder without centered indexing."""
    from cebra.models import init
    return init("offset10-model", num_neurons=neurons, num_units=128,
                num_output=dimension).to(device)


@beartype
def transform(model: torch.nn.Module, data: torch.Tensor,
              endpoints: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Export one normalized embedding for every source endpoint, in order."""
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(endpoints), batch_size):
            ix = torch.as_tensor(endpoints[start:start + batch_size], device=data.device)
            outputs.append(model(windows(data, ix)).cpu().numpy())
    result = np.concatenate(outputs)
    if not np.isfinite(result).all() or not np.allclose(np.linalg.norm(result, axis=1), 1, atol=1e-5):
        raise ValueError("CEBRA outputs must be finite and unit normalized.")
    return result


@beartype
def main() -> None:
    """Run or benchmark one approved configuration on a single allocated GPU."""
    import joblib
    from sklearn.preprocessing import StandardScaler
    from cebra.models import FixedCosineInfoNCE
    from experiments.churchland.scripts.baselines import load_data, trial_slices

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--root', type=Path, default=Path('/ceph/aeon/aeon/nldisco/churchland/sweeps/20260914_s10_endpoint_v1'))
    parser.add_argument('--data', type=Path, default=Path('experiments/churchland/data/nitschke_20090812'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError('An allocated CUDA GPU is required.')
    properties = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(34e9, properties.total_memory * 0.85) / properties.total_memory)
    sampled_process_peak = 0

    def sample_process_memory() -> int:
        nonlocal sampled_process_peak
        report = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'
        ], text=True)
        memory = sum(int(line.split(',')[1].strip()) * 1024 ** 2
                     for line in report.splitlines()
                     if line.split(',')[0].strip() == str(os.getpid()))
        sampled_process_peak = max(sampled_process_peak, memory)
        if memory >= 40e9:
            raise RuntimeError(f'Process GPU memory exceeded 40 GB: {memory}')
        return memory
    counts, trials = load_data(args.data)
    anchors = np.concatenate([np.arange(s.start + 9, s.stop) for s in trial_slices(trials)]).astype(np.int64)
    if len(anchors) != 130507:
        raise ValueError(f'Unexpected anchor count {len(anchors)}')
    shared_path = args.root / 'shared' / 'index.npz'
    if shared_path.exists():
        with np.load(shared_path) as shared:
            np.testing.assert_array_equal(anchors, shared['anchor_index'])
    elif not args.benchmark:
        raise RuntimeError('Shared endpoint index is not yet available.')
    scaler = StandardScaler()
    processed = scaler.fit_transform(np.sqrt(counts)).astype(np.float32)
    data = torch.from_numpy(processed).cuda()
    cfg = configurations()[args.index]
    refs_np = pair_endpoints(anchors, trials, cfg['lag'])
    refs = torch.as_tensor(refs_np, device='cuda')
    all_endpoints = torch.as_tensor(anchors, device='cuda')
    model = encoder(data.shape[1], cfg['dimension'], 'cuda')
    objective = FixedCosineInfoNCE(temperature=cfg['temperature']).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    def step(batch: int) -> tuple:
        reference = refs[torch.randint(len(refs), (batch,), device='cuda')]
        negative = all_endpoints[torch.randint(len(anchors), (batch,), device='cuda')]
        optimizer.zero_grad(set_to_none=True)
        embedded = model(windows(data, torch.cat((reference, reference + cfg['lag'], negative))))
        loss, alignment, uniformity = objective(*embedded.chunk(3))
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite objective')
        loss.backward()
        optimizer.step()
        return loss, alignment, uniformity

    if args.benchmark:
        results = []
        for batch in (512, 1024):
            for _ in range(5):
                step(batch)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(30):
                step(batch)
            torch.cuda.synchronize()
            sample_process_memory()
            seconds = time.perf_counter() - start
            results.append(dict(batch_size=batch, steps=30, seconds=seconds,
                                reference_draws_per_second=30 * batch / seconds,
                                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved()))
            results[-1]['sampled_process_gpu_peak_bytes'] = sampled_process_peak
            results[-1]['gpu_name'] = properties.name
        dest = args.root / 'cebra' / 'benchmark.json'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(results, indent=2))
        print(json.dumps(results), flush=True)
        return

    gate = json.loads((args.root / 'manifest.json').read_text())
    if gate.get('wandb_privacy_verified') is not True:
        raise RuntimeError('Private W&B project verification is required before production.')
    output = args.root / 'cebra' / cfg['config_id']
    output.mkdir(parents=True, exist_ok=False)
    steps = math.ceil(10 * len(anchors) / args.batch_size)
    cfg.update(seed=0, architecture='official cebra offset10-model', hidden_units=128,
               learning_rate=3e-4, optimizer='Adam', temperature_mode='constant',
               objective='official FixedCosineInfoNCE', batch_size=args.batch_size,
               steps=steps, epoch_equivalents=10, reference_draws=steps * args.batch_size,
               positive_draws=steps * args.batch_size, negative_draws=steps * args.batch_size,
               negative_similarity_comparisons=steps * args.batch_size ** 2,
               anchors=len(anchors), eligible_reference_anchors=len(refs_np),
               sampling='uniform eligible references with replacement; forward within-trial positives; uniform global valid-window negatives with replacement',
               endpoint_semantics='exact input [t-9,...,t], one valid convolution output at t',
               preprocessing='sqrt counts; neuron-wise StandardScaler fit full neural data',
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               index_file_sha256=hashlib.sha256(shared_path.read_bytes()).hexdigest(),
               gpu_name=properties.name, gpu_total_memory=properties.total_memory,
               slurm_job_id=os.environ.get('SLURM_JOB_ID'),
               slurm_array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
               slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
               counts_sha256=hashlib.sha256((args.data / 'counts.npy').read_bytes()).hexdigest(),
               trial_id_sha256=hashlib.sha256(trials.tobytes()).hexdigest(),
               endpoint_sha256=hashlib.sha256(anchors.tobytes()).hexdigest(),
               versions={p: importlib.metadata.version(p) for p in ('cebra', 'torch', 'numpy', 'scikit-learn')})
    (output / 'config.json').write_text(json.dumps(cfg, indent=2))
    (output / 'cebra_source.py').write_bytes(Path(__file__).read_bytes())
    import wandb
    run = wandb.init(entity='jkbhagatio', project='NLDisco-paper', group='20260914_s10_endpoint_v1',
                     name='cebra_' + cfg['config_id'], config=cfg, dir=str(output),
                     job_type='cebra-time', mode='online')
    (output / 'wandb_run.json').write_text(json.dumps(dict(id=run.id, url=run.url), indent=2))
    history = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    model.train()
    for i in range(steps):
        values = [float(value.detach()) for value in step(args.batch_size)]
        history.append(values)
        if i % 25 == 0 or i == steps - 1:
            process_gpu_bytes = sample_process_memory()
            run.log(dict(step=i + 1, reference_draws=(i + 1) * args.batch_size,
                         loss=values[0], alignment=values[1], uniformity=values[2],
                         process_gpu_bytes=process_gpu_bytes))
        if i % 100 == 0 or i == steps - 1:
            print(json.dumps(dict(config=cfg['config_id'], step=i + 1, steps=steps, loss=values[0])), flush=True)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    torch.save(dict(state_dict=model.state_dict(), config=cfg), output / 'model.pt')
    joblib.dump(scaler, output / 'preprocessing.joblib')
    np.save(output / 'objective_history.npy', np.asarray(history))
    activations = transform(model, data, anchors)
    np.savez(output / 'endpoint_activations.npz', activations=activations, source_index=anchors,
             trial_id=trials[anchors])
    reloaded_scaler = joblib.load(output / 'preprocessing.joblib')
    reloaded_data = torch.as_tensor(reloaded_scaler.transform(np.sqrt(counts)).astype(np.float32), device='cuda')
    restored = encoder(data.shape[1], cfg['dimension'], 'cuda')
    restored.load_state_dict(torch.load(output / 'model.pt', map_location='cuda', weights_only=False)['state_dict'])
    reloaded = transform(restored, reloaded_data, anchors)
    np.testing.assert_allclose(activations, reloaded, rtol=1e-6, atol=1e-6)
    sample_process_memory()
    summary = dict(status='complete', train_seconds=train_seconds,
                   total_seconds=time.perf_counter() - started,
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                   sampled_process_gpu_peak_bytes=sampled_process_peak,
                   wandb_url=run.url, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                   reload_max_abs_error=float(np.max(np.abs(activations - reloaded))),
                   loss_initial_100_mean=float(np.mean(np.asarray(history)[:100, 0])),
                   loss_final_100_mean=float(np.mean(np.asarray(history)[-100:, 0])),
                   endpoint_count=len(anchors), reference_draws=steps * args.batch_size)
    run.summary.update(summary)
    artifact = wandb.Artifact('cebra_' + cfg['config_id'], type='model')
    for name in ('model.pt', 'config.json'):
        artifact.add_file(str(output / name))
    run.log_artifact(artifact)
    run.finish()
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
