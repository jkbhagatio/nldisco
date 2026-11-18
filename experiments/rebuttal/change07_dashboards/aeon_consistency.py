"""Check each Aeon example latent's activating episodes against its feature description.

For the four latents shown in the paper, report how often windows at or above an
activation threshold satisfy the frozen feature condition (within the feature's
comparison population), and whether the dashboard's strongest and sampled episodes do.
Reads frozen masks and activations only; nothing in the paper is rescored.

    python -m experiments.rebuttal.change07_dashboards.aeon_consistency --out <file.json>
"""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.rebuttal.change07_dashboards.dashboard import interval_samples, top_episodes
from experiments.rebuttal.change07_dashboards.datasets import AEON

FEATURES = {"wheel": ("d192/k12_seed0", 52), "area": ("d256/k12_seed0", 107),
            "speed": ("d256/k12_seed0", 109), "direction": ("d256/k12_seed0", 10)}


def summarize(z: np.ndarray, condition: np.ndarray, valid: np.ndarray, latent: int,
              half_width: int = 500, n: int = 6, seed: int = 0) -> dict:
    """Precision by activation threshold, and condition status of the displayed episodes."""
    zv, cv = z[valid], condition[valid]
    cuts = {"z>0": zv > 0}
    cuts.update({f"top {100 * (1 - q):g}%": zv >= np.quantile(zv, q) for q in (.9, .99, .999)})
    edges = np.linspace(0, z.max(), n + 1)
    bands = {f"({lo:.1f}, {hi:.1f}]": (zv > lo) & (zv <= hi) for lo, hi in zip(edges, edges[1:])}
    peaks = top_episodes(z, n, half_width)
    picks, _ = interval_samples(z, n, np.random.default_rng([seed, latent]))
    rate = lambda mask: dict(windows=int(mask.sum()),  # noqa: E731
                             precision=float(cv[mask].mean()) if mask.any() else None)
    shown = lambda rows: dict(in_population=int(valid[rows].sum()),  # noqa: E731
                              in_condition=int((valid & condition)[rows].sum()), shown=len(rows))
    return dict(population=int(valid.sum()), prevalence=float(cv.mean()),
                by_threshold={k: rate(m) for k, m in cuts.items()},
                by_interval={k: rate(m) for k, m in bands.items()},
                strongest=shown(peaks), sampled=shown(picks))


def main() -> None:
    """Summarize all four latents and print or save the result."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    masks = np.load(AEON / "paper_revision_wheelcms_20260921/feature_arrays.npz")
    result = {}
    for name, (run, latent) in FEATURES.items():
        z = np.load(AEON / run / "activations.npy", mmap_mode="r")[:, latent].astype(np.float64)
        result[name] = dict(run=run, latent=latent, **summarize(
            z, masks[f"condition_{name}"], masks[f"valid_{name}"], latent))
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
