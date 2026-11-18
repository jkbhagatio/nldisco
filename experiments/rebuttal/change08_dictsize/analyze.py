"""Per-run dictionary-size analysis for the macaque (Sec. 4.2) and Aeon (Sec. 4.3) NLDisco models.

For every run: dead latents, training time, the best-matching latent (paper's selection rule)
for each of the paper's four feature descriptions, and, relative to the paper's D=192 model,
the share of its latents that reappear plus the new and near-duplicate latents.
One JSON (+ one NPZ with per-latent best correlations) is written per run, never overwritten.

    python -m experiments.rebuttal.change08_dictsize.analyze --dataset aeon
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.rebuttal.change08_dictsize import dictsize as ds

CEPH = Path('/ceph/aeon/aeon/nldisco/rebuttal_20261118/change08_dictsize')
TAUS_MATCH = (0.3, 0.5, 0.7, 0.9)
TAUS_DUP = (0.5, 0.7, 0.8, 0.9)


def macaque_runs() -> list:
    """(D, seed, role, activations path, summary dict) for every macaque k=16 two-level run."""
    from experiments.rebuttal.change08_dictsize import macaque_features as mf
    runs = [(d, s, role, mf.activation_path(folder, d),
             json.loads((folder / 'summary.json').read_text()) if (folder / 'summary.json').exists() else {})
            for d, s, folder, role in mf.existing_runs()]
    for folder in sorted((CEPH / 'runs/macaque').glob('d*_k16_l2_seed*')):
        if (folder / 'summary.json').exists():
            summary = json.loads((folder / 'summary.json').read_text())
            runs.append((summary['dimension'], summary['seed'], 'new', mf.activation_path(folder, summary['dimension']), summary))
    return runs


def aeon_runs() -> list:
    """(D, seed, role, activations path, metrics dict) for every Aeon k=12 run."""
    from experiments.rebuttal.change08_dictsize import aeon_features as af
    runs = []
    for d in (128, 192, 256):
        folder = af.SWEEP / f'd{d}/k12_seed0'
        runs.append((d, 0, 'paper' if d == 192 else 'existing', folder / 'activations.npy',
                     json.loads((folder / 'metrics.json').read_text())))
    for folder in sorted((CEPH / 'runs/aeon').glob('d*_k12_seed*')):
        if (folder / 'COMPLETE.json').exists():
            done = json.loads((folder / 'COMPLETE.json').read_text())
            runs.append((done['d'], done['seed'], 'new', folder / 'activations.npy',
                         json.loads((folder / 'metrics.json').read_text())))
    return runs


def session_b_runs() -> list:
    """(D, seed, role, activations path, metrics dict) for Aeon session B (change 2): own D=192 and D=256
    models plus the new D=512/1024 runs."""
    base = Path('/ceph/aeon/aeon/nldisco/rebuttal_20261118/change02_multisession/models/b_only')
    runs = []
    for d in (192, 256):
        folder = base / f'd{d}_k12_seed0'
        runs.append((d, 0, 'paper' if d == 192 else 'existing', folder / 'activations.npy',
                     json.loads((folder / 'metrics.json').read_text())))
    for folder in sorted((CEPH / 'runs/session_b').glob('d*_k12_seed*')):
        if (folder / 'COMPLETE.json').exists():
            done = json.loads((folder / 'COMPLETE.json').read_text())
            runs.append((done['d'], done['seed'], 'new', folder / 'activations.npy',
                         json.loads((folder / 'metrics.json').read_text())))
    return runs


def train_seconds(dataset: str, info: dict) -> float:
    """Wall-clock seconds recorded by the run itself (training + threshold calibration)."""
    if dataset == 'macaque':
        return float(info.get('training_seconds_including_calibration',
                              info.get('training_seconds_including_threshold_calibration', np.nan)))
    return float(info['elapsed_seconds'])


def features_for(dataset: str) -> dict:
    """name -> (rows, condition bool, groups) of the paper's four feature descriptions."""
    if dataset == 'session_b':
        return {}  # shared-latent, dead, new and near-duplicate counts only
    if dataset == 'macaque':
        from experiments.rebuttal.change08_dictsize import macaque_features as mf
        return {name: (f.rows, f.labels.astype(bool), f.trials) for name, f in mf.load_features().items()}
    from experiments.rebuttal.change08_dictsize import aeon_features as af
    return af.load_features()[1]


def analyse_run(dataset: str, d: int, seed: int, role: str, path: Path, info: dict,
                reference: 'ds.sp.csr_matrix', features: dict, out: Path, is_reference: bool) -> None:
    """Write ``out/{dataset}_d{d}_s{seed}.json`` for one run."""
    stem = out / f'{dataset}_d{d}_s{seed}'
    if stem.with_suffix('.json').exists():
        return
    started = time.time()
    x = ds.load_sparse(path)
    if x.shape[1] != d or x.shape[0] != reference.shape[0]:
        raise ValueError(f'{path}: shape {x.shape} does not match D={d} / reference windows')
    live = ds.live_mask(x)
    record = dict(dataset=dataset, D=d, seed=seed, role=role, activations=str(path),
                  n_windows=int(x.shape[0]), dead=int((~live).sum()), dead_fraction=float((~live).mean()),
                  mean_l0=float(x.getnnz() / x.shape[0]), train_seconds=train_seconds(dataset, info),
                  gpu=info.get('gpu_name', ''), is_paper_model=is_reference)
    record['features'] = {}
    for name, (rows, condition, groups) in features.items():
        scores = ds.score_latents(x, rows, condition, groups)
        winners = ds.select_winners(scores)
        record['features'][name] = dict(
            n_eligible=int(scores.eligible.sum()),
            **{key: (None if w is None else {k: float(v) for k, v in w.items()}) for key, w in winners.items()})
    corr_self = ds.pearson_cross(x, x)
    record['near_duplicates'] = {str(t): ds.near_duplicates(corr_self, live, t) for t in TAUS_DUP}
    arrays = {}
    if not is_reference:
        ref_live = ds.live_mask(reference)
        corr = ds.pearson_cross(reference, x)
        record['vs_paper_model'] = {str(t): ds.reappearance(ref_live, corr, live, t) for t in TAUS_MATCH}
        arrays['best_ref_to_big'] = np.where(live[None], corr, -np.inf).max(1)
        arrays['best_big_to_ref'] = np.where(ref_live[:, None], corr, -np.inf).max(0)
    np.fill_diagonal(corr_self, -np.inf)
    arrays['best_self'] = np.where(live[None], corr_self, -np.inf).max(1)
    arrays['live'] = live
    np.savez(stem.with_suffix('.npz'), **arrays)
    record['analysis_seconds'] = time.time() - started
    stem.with_suffix('.json').write_text(json.dumps(record, indent=1))
    print(f'{dataset} D={d} seed={seed}: dead={record["dead"]} analysed in {record["analysis_seconds"]:.0f}s', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('macaque', 'aeon', 'session_b'), required=True)
    parser.add_argument('--output', type=Path, default=CEPH / 'analysis')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = dict(macaque=macaque_runs, aeon=aeon_runs, session_b=session_b_runs)[args.dataset]()
    paper = next(r for r in runs if r[2] == 'paper')
    reference = ds.load_sparse(paper[3])
    features = features_for(args.dataset)
    for d, seed, role, path, info in sorted(runs, key=lambda r: (r[0], r[1])):
        analyse_run(args.dataset, d, seed, role, path, info, reference, features, args.output, role == 'paper')


if __name__ == '__main__':
    main()
