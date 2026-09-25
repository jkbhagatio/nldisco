"""One width-only CEBRA-Time check, preserving the original sweep artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from beartype import beartype
from sklearn.preprocessing import StandardScaler

from experiments.churchland.sweeps.s10_endpoint_v1 import cebra as original
from experiments.churchland.sweeps.s10_endpoint_v1 import comparison_features as shared

OUTPUT = shared.SWEEP / 'width192_checks/cebra/d192_t1p0_lag1'
GPU_UUID = 'GPU-e04bd614-057a-08f7-450d-b1bfa27f297d'


@beartype
def digest(path: Path) -> str:
    """Hash an input or saved artifact."""
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


@beartype
def choose(rows: list, metric: str) -> dict:
    """Select an eligible dimension; exact score ties favor the lowest ID."""
    if metric not in ('sel', 'auroc'):
        raise ValueError(metric)
    return max((row for row in rows if row['eligible']),
               key=lambda row: (row[metric], -row['latent_id']))


@beartype
def train(output: Path) -> None:
    """Train the selected CEBRA recipe with only output width changed to 192."""
    from cebra.models import FixedCosineInfoNCE

    from experiments.churchland.scripts.baselines import load_data, trial_slices

    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
        raise RuntimeError('This check is restricted to the authorized physical GPU1 UUID.')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly the authorized H100 must be visible.')
    properties = torch.cuda.get_device_properties(0)
    if 'H100' not in properties.name:
        raise RuntimeError(f'Expected H100, got {properties.name}')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.set_per_process_memory_fraction(34e9 / properties.total_memory)
    sampled_peak = 0

    def sample_memory() -> int:
        nonlocal sampled_peak
        report = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
            '--format=csv,noheader,nounits'], text=True)
        used = sum(int(line.split(',')[1].strip()) * 1024 ** 2
                   for line in report.splitlines()
                   if line.split(',')[0].strip() == str(os.getpid()))
        sampled_peak = max(sampled_peak, used)
        if used >= 40e9:
            raise RuntimeError('Training process exceeded the 40GB ceiling.')
        return used

    data_path = shared.ROOT / 'experiments/churchland/data/nitschke_20090812'
    counts, trials = load_data(data_path)
    anchors = np.concatenate([np.arange(s.start + 9, s.stop)
                              for s in trial_slices(trials)]).astype(np.int64)
    with np.load(shared.SWEEP / 'shared/index.npz') as saved:
        np.testing.assert_array_equal(anchors, saved['anchor_index'])
    if len(anchors) != 130507:
        raise RuntimeError('Unexpected window count.')
    scaler = StandardScaler()
    data = torch.from_numpy(scaler.fit_transform(np.sqrt(counts)).astype(np.float32)).cuda()
    references_np = original.pair_endpoints(anchors, trials, 1)
    references = torch.as_tensor(references_np, device='cuda')
    endpoints = torch.as_tensor(anchors, device='cuda')
    model = original.encoder(data.shape[1], 192, 'cuda')
    objective = FixedCosineInfoNCE(temperature=1.).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    batch_size = 1024
    steps = math.ceil(10 * len(anchors) / batch_size)
    assert steps == 1275
    parent_path = shared.SWEEP / 'cebra/d128_t1p0_lag1/config.json'
    config = json.loads(parent_path.read_text())
    config.update(config_id='d192_t1p0_lag1', dimension=192,
                  gpu_name=properties.name, gpu_total_memory=properties.total_memory,
                  slurm_job_id=None, slurm_array_job_id=None, slurm_array_task_id=None,
                  source_sha256=digest(Path(__file__)),
                  parent_config_path=str(parent_path), parent_config_sha256=digest(parent_path),
                  reused_training_helpers_sha256=digest(Path(original.__file__)),
                  host=socket.gethostname(), gpu_uuid=GPU_UUID,
                  only_training_config_change='output dimension 128 -> 192',
                  wandb_mode='disabled; supplementary local check, no external upload')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    (output / 'cebra_192_check_source.py').write_bytes(Path(__file__).read_bytes())
    (output / 'original_cebra_source.py').write_bytes(Path(original.__file__).read_bytes())
    history = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    model.train()
    for index in range(steps):
        reference = references[torch.randint(len(references), (batch_size,), device='cuda')]
        negative = endpoints[torch.randint(len(anchors), (batch_size,), device='cuda')]
        optimizer.zero_grad(set_to_none=True)
        embeddings = model(original.windows(data, torch.cat((reference, reference + 1, negative))))
        result = objective(*embeddings.chunk(3))
        if not all(torch.isfinite(value).all() for value in result):
            raise ValueError('Nonfinite training objective.')
        result[0].backward()
        optimizer.step()
        history.append([float(value.detach()) for value in result])
        if index % 25 == 0 or index == steps - 1:
            sample_memory()
        if index % 100 == 0 or index == steps - 1:
            print(json.dumps(dict(step=index + 1, steps=steps, loss=history[-1][0])), flush=True)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    torch.save(dict(state_dict=model.state_dict(), config=config), output / 'model.pt')
    joblib.dump(scaler, output / 'preprocessing.joblib')
    np.save(output / 'objective_history.npy', np.asarray(history))
    activations = original.transform(model, data, anchors)
    np.savez(output / 'endpoint_activations.npz', activations=activations,
             source_index=anchors, trial_id=trials[anchors])
    restored = original.encoder(data.shape[1], 192, 'cuda')
    restored.load_state_dict(torch.load(output / 'model.pt', weights_only=False)['state_dict'])
    restored_data = torch.as_tensor(joblib.load(output / 'preprocessing.joblib').transform(
        np.sqrt(counts)).astype(np.float32), device='cuda')
    reloaded = original.transform(restored, restored_data, anchors)
    np.testing.assert_allclose(activations, reloaded, atol=1e-6, rtol=1e-6)
    endpoint = int(anchors[0])
    perturbed = data.clone()
    perturbed[endpoint + 1:] += 100
    before = original.transform(model, data, anchors[:1])
    after = original.transform(model, perturbed, anchors[:1])
    np.testing.assert_array_equal(before, after)
    sample_memory()
    summary = dict(status='complete', train_seconds=train_seconds,
                   total_seconds=time.perf_counter() - start, steps_completed=len(history),
                   reference_draws=len(history) * batch_size, epoch_equivalents=10,
                   endpoint_count=len(anchors), embedding_shape=list(activations.shape),
                   finite_history=bool(np.isfinite(history).all()),
                   loss_initial_100_mean=float(np.mean(np.asarray(history)[:100, 0])),
                   loss_final_100_mean=float(np.mean(np.asarray(history)[-100:, 0])),
                   reload_max_abs_error=float(np.max(np.abs(activations - reloaded))),
                   endpoint_future_isolation_max_abs_error=float(np.max(np.abs(before - after))),
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                   sampled_process_gpu_peak_bytes=sampled_peak,
                   gpu_name=properties.name, gpu_uuid=GPU_UUID,
                   test_labels_accessed=False, wandb_uploaded=False)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


@beartype
def score(output: Path) -> None:
    """Score all four frozen development features with both selection criteria."""
    features = shared.load_development_features()
    with np.load(output / 'endpoint_activations.npz') as saved:
        with np.load(shared.SWEEP / 'shared/index.npz') as index:
            np.testing.assert_array_equal(saved['source_index'], index['anchor_index'])
        values = saved['activations']
    if values.shape != (130507, 192):
        raise ValueError(f'Unexpected activation shape {values.shape}')
    candidates, selected, definitions = [], {}, {}
    for name in shared.EXPECTED_FEATURES:
        feature = features[name]
        rows = shared.score_dense(values, feature)
        candidates.extend(dict(feature=name, **row) for row in rows)
        selected[name] = {f'best_{metric}': choose(rows, metric) for metric in ('sel', 'auroc')}
        definitions[name] = dict(feature.definition, reference_activity_rate=feature.reference_rate,
                                 valid_rows=len(feature.rows),
                                 rows_sha256=hashlib.sha256(feature.rows.tobytes()).hexdigest(),
                                 labels_sha256=hashlib.sha256(feature.labels.tobytes()).hexdigest())
    destination = output / 'feature_scores'
    destination.mkdir(exist_ok=False)
    (destination / 'all_candidates.json').write_text(json.dumps(candidates, indent=2))
    report = dict(status='complete', method='CEBRA-Time', dimension=192,
                  config_id='d192_t1p0_lag1', selected=selected, definitions=definitions,
                  development_only=True, test_labels_accessed=False, decoder_fitted=False,
                  activity_rule='Sign by development AUROC, strict development quantile rate matching to displayed NLDisco256 latents; ties unsplit.',
                  eligibility_rule='Varying, TPR >= 0.05, >=20 active-positive trials; no Sel gate.',
                  selection_rule='Maximum eligible Sel or AUROC separately; exact metric ties use lowest latent ID.',
                  training_config_sha256=digest(output / 'config.json'),
                  export_sha256=digest(output / 'endpoint_activations.npz'),
                  comparison_source_sha256=digest(Path(shared.__file__)))
    (destination / 'summary.json').write_text(json.dumps(report, indent=2))
    (destination / 'comparison_features_source.py').write_bytes(Path(shared.__file__).read_bytes())
    print(json.dumps(report, indent=2), flush=True)


@beartype
def main() -> None:
    """Run one explicitly requested training or development-scoring stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('train', 'score'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    (train if args.stage == 'train' else score)(args.output)


if __name__ == '__main__':
    main()
