"""Spike-reconstruction comparison on the Section 4.2 macaque data (rebuttal change 10).

A shared L2-regularised Poisson regression maps each method's full endpoint representation
to the spike counts of all channels in the endpoint (last) bin of its ten-bin window. It is
fit on training trials, its regularisation strength is chosen by validation bits per spike,
and it is scored once on test trials (the frozen whole-trial 80/10/10 split). Methods with
their own decoder also get their native endpoint-bin reconstruction scored the same way
(native rates are produced by ``native_nldisco.py`` and ``native_langevinflow.py``).

Adding a method is one entry in ``METHODS`` (or a ``REPRESENTATIONS_READY.json`` marker, for
ST-NDT). Run from the repository root::

    python -m experiments.rebuttal.change10_reconstruction.reconstruction
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from scipy.special import gammaln, xlogy

MAIN = Path('/nfs/nhome/live/jbhagat/nldisco')
SWEEP = Path('/ceph/aeon/aeon/nldisco/churchland/sweeps/20260914_s10_endpoint_v1')
DATA = MAIN / 'experiments/churchland/data/nitschke_20090812'
FROZEN = MAIN / 'experiments/churchland/outputs/s10_endpoint_revision_v4_nice_cutoffs/frozen_spec.json'
OUT = Path('/ceph/aeon/aeon/nldisco/rebuttal_20261118/change10_reconstruction')
STNDT_MARKER = Path('/ceph/aeon/aeon/nldisco/rebuttal_20261118/change01_stndt/REPRESENTATIONS_READY.json')
ROOT = Path(__file__).resolve().parents[3]
TABLE = ROOT / 'paper/iclr_paper/figures/rebuttal/change10_reconstruction'
N_WINDOWS = 130507
ALPHAS = tuple(10. ** np.arange(-6, 1))  # L2 strengths, sklearn PoissonRegressor convention
FLOORS = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1)  # rate floors for native decoders without positive output
SPLITS = dict(train=0, validation=1, test=2)


@dataclass(frozen=True)
class Method:
    """One compared method: its endpoint representation and, optionally, native rates."""

    key: str
    label: str
    representation: str
    native: Optional[str] = None  # (N_WINDOWS, channels) native endpoint-bin rates in counts/bin


_spec = json.loads(FROZEN.read_text())['representations']
METHODS = [
    Method('nldisco', 'NLDisco', _spec['nldisco']['path'], str(OUT / 'native/nldisco_endpoint_rates.npy')),
    Method('cebra', 'CEBRA-Time', _spec['cebra']['path']),
    Method('langevinflow', 'LangevinFlow', _spec['langevinflow']['path'],
           str(OUT / 'native/langevinflow_endpoint_rates.npy')),
]


@beartype
def stndt_method(marker: Path = STNDT_MARKER) -> Optional[Method]:
    """Build the ST-NDT entry from change 1's marker, if it has been written."""
    if not marker.exists():
        return None
    info = json.loads(marker.read_text())
    path = info.get('macaque_representation') or info.get('representation') or info.get('path')
    if path is None:
        paths = [v for v in info.values() if isinstance(v, str) and v.endswith(('.npy', '.npz'))]
        if len(paths) != 1:
            raise ValueError(f'Cannot identify the ST-NDT representation path in {marker}: {info}')
        path = paths[0]
    return Method('stndt', 'ST-NDT', str(path), info.get('native_rates'))


@jaxtyped(typechecker=beartype)
def poisson_nll(rates: Float[np.ndarray, 'n c'], counts: Float[np.ndarray, 'n c']) -> Float[np.ndarray, ' c']:
    """Per-channel Poisson negative log-likelihood in nats (with 0 log 0 = 0)."""
    return (rates - xlogy(counts, rates) + gammaln(counts + 1)).sum(0)


@jaxtyped(typechecker=beartype)
def bits_per_spike(rates: Float[np.ndarray, 'n c'], counts: Float[np.ndarray, 'n c']) -> float:
    """NLB reconstruction score: gain over each channel's evaluation-set mean rate, per spike, in bits."""
    if (rates <= 0).any() and (counts[rates <= 0] > 0).any():
        raise ValueError('Nonpositive predicted rate for an observed spike.')
    null = np.broadcast_to(counts.mean(0, keepdims=True), counts.shape)
    return float((poisson_nll(null, counts).sum() - poisson_nll(rates, counts).sum()) / counts.sum() / np.log(2))


@jaxtyped(typechecker=beartype)
def variance_explained(rates: Float[np.ndarray, 'n c'], counts: Float[np.ndarray, 'n c']) -> tuple:
    """Mean over channels of 1 - SSE / SST; channels without variance are skipped and counted."""
    sst = ((counts - counts.mean(0)) ** 2).sum(0)
    keep = sst > 0
    per_channel = 1 - ((counts - rates) ** 2).sum(0)[keep] / sst[keep]
    return float(per_channel.mean()), int((~keep).sum())


@jaxtyped(typechecker=beartype)
def scores(rates: Float[np.ndarray, 'n c'], counts: Float[np.ndarray, 'n c']) -> dict:
    """Both reconstruction scores; zero is the mean-rate predictor, higher is better."""
    ve, skipped = variance_explained(rates, counts)
    return dict(bits_per_spike=bits_per_spike(rates, counts), variance_explained=ve,
                channels_without_variance=skipped)


@jaxtyped(typechecker=beartype)
def trial_bootstrap(rates: Float[np.ndarray, 'n c'], counts: Float[np.ndarray, 'n c'],
                    trials: Int[np.ndarray, ' n'], repetitions: int = 1000, seed: int = 0) -> dict:
    """95% intervals from resampling whole test trials (windows within a trial overlap).

    Exact and vectorised: both scores depend on a resample only through per-trial sums
    (log-factorials cancel in bits per spike), so each draw is a weighted sum of trial blocks.
    """
    _, inverse = np.unique(trials, return_inverse=True)
    terms = np.stack([counts, rates, xlogy(counts, rates), counts ** 2, (counts - rates) ** 2])
    sums = np.zeros((terms.shape[0], inverse.max() + 1, counts.shape[1]))
    np.add.at(sums, (slice(None), inverse), terms)  # (stat, trial, channel)
    sizes = np.bincount(inverse).astype(np.float64)
    rng = np.random.default_rng(seed)
    weights = np.stack([np.bincount(rng.integers(len(sizes), size=len(sizes)), minlength=len(sizes))
                        for _ in range(repetitions)]).astype(np.float64)  # (draw, trial)
    s, r, l, s2, sse = np.einsum('bt,ktc->kbc', weights, sums)
    n = (weights @ sizes)[:, None]
    m = s / n
    null = n * m - xlogy(s, m)
    bps = ((null - (r - l)).sum(1)) / s.sum(1) / np.log(2)
    sst = s2 - n * m ** 2
    keep = sst > 1e-9
    ve = np.where(keep, 1 - sse / np.where(keep, sst, 1), 0).sum(1) / keep.sum(1)
    low, high = np.quantile(np.stack([bps, ve], 1), [.025, .975], axis=0)
    return dict(bits_per_spike_ci=[float(low[0]), float(high[0])],
                variance_explained_ci=[float(low[1]), float(high[1])], repetitions=repetitions)


@jaxtyped(typechecker=beartype)
def fit_poisson(x: Float[torch.Tensor, 'n d'], y: Float[torch.Tensor, 'n c'], alpha: float,
                init: Optional[tuple] = None, max_iter: int = 1000) -> tuple:
    """Jointly fit independent per-channel L2 Poisson GLMs (log link) with full-batch L-BFGS.

    Minimises, for every channel, mean(exp(eta) - y * eta) + alpha / 2 * ||w||^2 with
    eta = x w + b, the sklearn ``PoissonRegressor`` objective; channels share nothing but x.
    """
    w = torch.zeros(x.shape[1], y.shape[1], dtype=x.dtype, device=x.device) if init is None else init[0].clone()
    b = torch.log(y.mean(0).clamp_min(1e-8)) if init is None else init[1].clone()
    w.requires_grad_(True)
    b.requires_grad_(True)
    optimizer = torch.optim.LBFGS([w, b], lr=1, max_iter=max_iter, tolerance_grad=1e-9,
                                  tolerance_change=1e-12, history_size=20, line_search_fn='strong_wolfe')

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        eta = torch.addmm(b, x, w)
        loss = (torch.exp(eta) - y * eta).mean(0).sum() + alpha / 2 * w.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    grad = max(float(w.grad.abs().max()), float(b.grad.abs().max()))
    return w.detach(), b.detach(), grad


@jaxtyped(typechecker=beartype)
def predict(x: Float[torch.Tensor, 'n d'], params: tuple) -> Float[np.ndarray, 'n c']:
    """Predicted rates (counts per bin) of a fitted GLM."""
    return torch.exp(torch.addmm(params[1], x, params[0])).cpu().numpy()


@jaxtyped(typechecker=beartype)
def shared_readout(values: Float[np.ndarray, 'n d'], counts: Float[np.ndarray, 'n c'],
                   split: Int[np.ndarray, ' n'], device: str, alphas: tuple = ALPHAS) -> tuple:
    """Fit on train, choose alpha by validation bits per spike, return test rates and a record."""
    train, validation, test = (split == SPLITS[s] for s in ('train', 'validation', 'test'))
    mean, std = values[train].mean(0), values[train].std(0)
    std[std == 0] = 1.  # dead dimensions stay zero after standardisation
    to = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)  # noqa: E731
    scaled = (values - mean) / std
    x_train, x_val, x_test = (to(scaled[mask]) for mask in (train, validation, test))
    y_train = to(counts[train])
    grid, params, best = [], None, None
    for alpha in sorted(alphas, reverse=True):  # warm-start along a decreasing path
        params = fit_poisson(x_train, y_train, float(alpha), params)[:2]
        val = bits_per_spike(predict(x_val, params), counts[validation])
        grid.append(dict(alpha=float(alpha), validation_bits_per_spike=val))
        if best is None or val > best[0]:
            best = (val, float(alpha), params)
    _, alpha, params = best
    w, b, grad = fit_poisson(x_train, y_train, alpha, params)  # polish the selected fit
    record = dict(chosen_alpha=alpha, alpha_at_grid_edge=alpha in (min(alphas), max(alphas)),
                  final_max_abs_gradient=grad, grid=grid, standardised_on='train',
                  train_bits_per_spike=bits_per_spike(predict(x_train, (w, b)), counts[train]),
                  validation_bits_per_spike=bits_per_spike(predict(x_val, (w, b)), counts[validation]))
    return predict(x_test, (w, b)), record


@jaxtyped(typechecker=beartype)
def native_readout(rates: Float[np.ndarray, 'n c'], counts: Float[np.ndarray, 'n c'],
                   split: Int[np.ndarray, ' n'], floors: tuple = FLOORS) -> tuple:
    """Score a method's own decoder; a rate floor (chosen on validation) handles nonpositive output."""
    validation, test = split == SPLITS['validation'], split == SPLITS['test']
    nonpositive = float((rates <= 0).mean())
    grid = [dict(floor=float(f), validation_bits_per_spike=bits_per_spike(np.maximum(rates[validation], f),
                                                                          counts[validation])) for f in floors]
    floor = max(grid, key=lambda r: r['validation_bits_per_spike'])['floor'] if nonpositive else 0.
    return (np.maximum(rates[test], floor) if floor else rates[test],
            dict(rate_floor=floor, nonpositive_fraction_all_windows=nonpositive,
                 floor_grid=grid if nonpositive else None))


@beartype
def load_values(path: Path, anchors: np.ndarray) -> np.ndarray:
    """Load an (N_WINDOWS, d) array and verify it is ordered by the shared endpoint index."""
    if path.suffix == '.npz':
        with np.load(path) as archive:
            key = 'activations' if 'activations' in archive.files else archive.files[0]
            values = archive[key]
            index = next((archive[k] for k in ('source_index', 'anchor_index') if k in archive.files), None)
    else:
        values = np.load(path)
        index = None
        for name in ('activation_index.npz', 'index.npz'):
            if (path.parent / name).exists():
                with np.load(path.parent / name) as archive:
                    index = next((archive[k] for k in ('source_index', 'anchor_index') if k in archive.files), None)
                break
    if index is not None:
        np.testing.assert_array_equal(index, anchors)
    if values.ndim != 2 or len(values) != N_WINDOWS or not np.isfinite(values).all():
        raise ValueError(f'Invalid array {path}: shape {values.shape}')
    return np.asarray(values, dtype=np.float64)


@beartype
def load_targets() -> tuple:
    """Endpoint-bin counts, split and trial of every shared window."""
    with np.load(SWEEP / 'shared/index.npz') as index:
        anchors = index['anchor_index']
    with np.load(DATA / 'metadata.npz') as meta:
        split, trial = meta['split'][anchors].astype(np.int64), meta['trial_id'][anchors].astype(np.int64)
        np.testing.assert_array_equal(meta['split'][anchors - 9], split)  # windows never cross splits
    counts = np.load(DATA / 'counts.npy')[anchors].astype(np.float64)
    return anchors, counts, split, trial


@beartype
def latex_table(results: dict, path: Path) -> None:
    """Write the appendix table body (booktabs), best value per column in bold."""
    rows = list(results.values())
    cols = [('readout', 'bits_per_spike'), ('readout', 'variance_explained'),
            ('native', 'bits_per_spike'), ('native', 'variance_explained')]
    best = {c: max((r[c[0]]['test'][c[1]] for r in rows if r.get(c[0])), default=None) for c in cols}
    lines = []
    for r in rows:
        cells = [r['label']]
        for c in cols:
            if not r.get(c[0]):
                cells.append('--')
                continue
            v, (lo, hi) = r[c[0]]['test'][c[1]], r[c[0]]['test'][c[1] + '_ci']
            text = f'{v:.3f}'
            cells.append((rf'\textbf{{{text}}}' if np.isclose(v, best[c]) else text)
                         + rf' {{\scriptsize [{lo:.3f}, {hi:.3f}]}}')
        lines.append(' & '.join(cells) + r' \\')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')


@beartype
def main() -> None:
    """Score every configured method and write JSON, Markdown and the LaTeX table body."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', type=Path, default=OUT / 'results')
    parser.add_argument('--table', type=Path, default=TABLE / 'reconstruction_rows.tex')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    methods = METHODS + [m for m in (stndt_method(),) if m is not None]
    anchors, counts, split, trial = load_targets()
    test = split == SPLITS['test']
    results = {}
    for m in methods:
        values = load_values(Path(m.representation), anchors)
        rates, record = shared_readout(values, counts, split, args.device)
        record['test'] = dict(**scores(rates, counts[test]),
                              **trial_bootstrap(rates, counts[test], trial[test], args.bootstrap))
        entry = dict(label=m.label, method=asdict(m), dimension=values.shape[1], readout=record, native=None)
        if m.native is not None and Path(m.native).exists():
            native = load_values(Path(m.native), anchors)
            if native.shape[1] != counts.shape[1]:
                raise ValueError(f'{m.key} native rates have {native.shape[1]} channels')
            rates, info = native_readout(native, counts, split)
            entry['native'] = dict(**info, test=dict(**scores(rates, counts[test]),
                                                     **trial_bootstrap(rates, counts[test], trial[test],
                                                                       args.bootstrap)))
        elif m.native is not None:
            print(f'Native rates missing for {m.key}: {m.native}', flush=True)
        results[m.key] = entry
        print(json.dumps({m.key: dict(readout=record['test'], alpha=record['chosen_alpha'],
                                      native=entry['native'] and entry['native']['test'])}), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = dict(windows=N_WINDOWS, channels=counts.shape[1],
                rows=dict(train=int((split == 0).sum()), validation=int((split == 1).sum()), test=int(test.sum())),
                test_trials=int(len(np.unique(trial[test]))), test_spikes=float(counts[test].sum()),
                target='spike counts in the endpoint (last) bin of each ten-bin window', alphas=list(ALPHAS),
                floors=list(FLOORS))
    (args.out / 'results.json').write_text(json.dumps(dict(meta=meta, methods=results), indent=2, allow_nan=False))
    lines = ['| Method | readout bps | readout VE | native bps | native VE | alpha |', '|---|---:|---:|---:|---:|---:|']
    for r in results.values():
        n = r['native']['test'] if r['native'] else None
        lines.append(f"| {r['label']} | {r['readout']['test']['bits_per_spike']:.4f} | "
                     f"{r['readout']['test']['variance_explained']:.4f} | "
                     f"{n['bits_per_spike'] if n else float('nan'):.4f} | "
                     f"{n['variance_explained'] if n else float('nan'):.4f} | {r['readout']['chosen_alpha']:g} |")
    (args.out / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    latex_table(results, args.table)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
