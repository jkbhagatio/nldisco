"""Regenerate the Aeon source activity tracebacks as encoder attribution.

Using the same native-active windows and training standardization, this script credits each cluster and lag with its standardized activity times the encoder gradient of the
latent's endpoint pre-activation. Outputs stay under ``experiments/aeon/outputs/``.

Usage: uv run --no-sync python -m experiments.aeon.sweep_20260921.attribution_traceback
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from beartype import beartype  # noqa: E402
from einops import einsum, reduce  # noqa: E402

from experiments.aeon.sweep_20260921 import paper_results as p  # noqa: E402
from experiments.aeon.sweep_20260921.shared_training import make_config  # noqa: E402
from experiments.attribution import encoder_attribution  # noqa: E402
from nldisco.model import Sed, build_sed  # noqa: E402

ATTRIBUTION = Path(__file__).resolve().parents[1] / "outputs/attribution"
FIGURES = Path(__file__).resolve().parents[1] / "outputs/paper"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 8192
WIDTH = 20


@beartype
def load_model(d: int) -> Sed:
    ckpt = torch.load(p.ROOT / f"d{d}/k12_seed0/checkpoint.pt", map_location="cpu",
                      weights_only=True)
    model = build_sed(make_config(d, 12, 90))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(DEVICE).eval()


@beartype
def pooled_attribution(model: Sed, counts: np.ndarray, endpoints: np.ndarray, latent: int,
                       pools: dict, mean: np.ndarray, std: np.ndarray,
                       reference: np.ndarray) -> dict:
    """Mean attribution over every pool in one pass over the union of active windows."""
    union = np.flatnonzero(np.logical_or.reduce(list(pools.values())))
    masks = {name: torch.from_numpy(mask[union].astype(np.float64)) for name, mask in pools.items()}
    sums = {name: torch.zeros(WIDTH, counts.shape[1], dtype=torch.float64) for name in pools}
    offsets = np.arange(-WIDTH + 1, 1)
    checked = False
    for start in range(0, len(union), BATCH):
        rows = union[start : start + BATCH]
        windows = ((counts[endpoints[rows][:, None] + offsets].astype(np.float32) - mean) / std)
        batch = torch.from_numpy(windows).to(DEVICE)
        if not checked:  # The rebuilt model must reproduce the exported native activations.
            occurrence = torch.zeros(len(rows), WIDTH, dtype=torch.bool, device=DEVICE)
            occurrence[:, -1] = True
            with torch.inference_mode():
                out = model(batch, occurrence_mask=occurrence)
            level = model.cfg.n_features
            np.testing.assert_allclose(out.sparse_acts[level][:, -1, latent].cpu().numpy(),
                                       reference[rows], rtol=1e-3, atol=1e-3)
            checked = True
        contribution = encoder_attribution(model, batch, latent).double().cpu()
        for name, mask in masks.items():
            sums[name] += einsum(mask[start : start + BATCH], contribution, "w, w t c -> t c")
    return {name: (sums[name] / max(int(pools[name].sum()), 1)).numpy() for name in pools}


@beartype
def compute() -> dict:
    ATTRIBUTION.mkdir(parents=True, exist_ok=True)
    counts = np.load(p.ROOT / "data/counts.npy")
    arrays = np.load(p.OUT / "feature_arrays.npz")
    endpoints = arrays["endpoint_bins"]
    metadata = pd.read_csv(p.ROOT / "data/cluster_metadata.csv")
    traces = dict(cluster_ids=metadata.cluster_id.to_numpy(),
                  good=metadata.KSLabel.to_numpy() == "good",
                  lags_s=np.arange(-WIDTH + 1, 1) * .02)
    profiles, supports = [], []
    for name, (d, latent, _) in p.FEATURES.items():
        norm = np.load(p.ROOT / f"d{d}/k12_seed0/normalization.npz")
        model = load_model(d)
        values = arrays[f"values_{name}"]
        active = values > 0
        pools = {"all": active}
        if f"condition_{name}" in arrays:
            condition, valid = arrays[f"condition_{name}"], arrays[f"valid_{name}"]
            pools.update(inside=active & valid & condition, outside=active & valid & ~condition)
        print(f"Attributing {name}: D{d}/K12/L{latent}, "
              + ", ".join(f"{k}={int(v.sum()):,}" for k, v in pools.items()), flush=True)
        result = pooled_attribution(model, counts, endpoints, latent, pools, norm["mean"],
                                    norm["std"], values)
        for pool, trace in result.items():
            traces[f"{name}_{pool}"] = trace
            supports.append(dict(feature=name, pool=pool, windows=int(pools[pool].sum())))
            for cluster, label, z in zip(metadata.cluster_id, metadata.KSLabel,
                                         reduce(trace, "lag cluster -> cluster", "mean")):
                profiles.append(dict(feature=name, pool=pool, cluster_id=int(cluster),
                                     KSLabel=label, mean_attribution=float(z)))
        del model
        torch.cuda.empty_cache()
    np.savez_compressed(ATTRIBUTION / "traceback_attribution.npz", **traces)
    pd.DataFrame(supports).to_csv(ATTRIBUTION / "traceback_support.csv", index=False)
    pd.DataFrame(profiles).to_csv(ATTRIBUTION / "cluster_profiles.csv", index=False)
    (ATTRIBUTION / "protocol.txt").write_text(
        "Mean over native-active windows of standardized source count times the gradient of the "
        "latent's endpoint pre-activation with respect to that count (gradient x input through "
        "the encoder). Same windows, pools and standardization as traceback.npz.\n")
    return traces


@beartype
def render(traces: dict) -> None:
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    blue, ids = "#2166ac", traces["cluster_ids"]
    label = "Mean encoder attribution"

    z = reduce(traces["wheel_all"], "lag cluster -> cluster", "mean")
    fig, ax = plt.subplots(figsize=(7.1, 2.3), constrained_layout=True)
    ax.axhline(0, color=".7", lw=.7)
    ax.vlines(ids, 0, z, color=blue, lw=.7)
    ax.scatter(ids, z, color=blue, s=8)
    ax.set(xlabel="Recorded cluster ID", ylabel=label,
           title="Wheel activity latent 52: source activity traceback")
    p.save_figure(fig, ATTRIBUTION, "aeon_wheel_source_traceback")

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.5), constrained_layout=True)
    for ax, (name, (d, latent, _)) in zip(axes.flat, p.FEATURES.items()):
        values = traces[f"{name}_all"]
        vmax = max(1e-4, float(np.quantile(np.abs(values), .99)))
        im = ax.imshow(values, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax,
                       vmax=vmax, extent=[-.5, len(ids) - .5, -.39, .01], interpolation="nearest")
        ax.set(title=f"D{d}/K12 #{latent}: all active windows", xlabel="Recorded cluster ID",
               ylabel="Time from endpoint (s)", yticks=[-.38, -.2, 0])
        ax.set_xticks(np.arange(0, len(ids), 20), ids[::20])
        fig.colorbar(im, ax=ax, label=label, shrink=.85, pad=.015)
    p.save_figure(fig, ATTRIBUTION, "aeon_traceback")

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.3), constrained_layout=True)
    vmax = max(float(np.quantile(np.abs(traces[f"wheel_{q}"]), .99)) for q in ("inside", "outside"))
    for ax, pool, title in zip(axes, ("inside", "outside"), ("Wheel active", "Wheel not active")):
        im = ax.imshow(traces[f"wheel_{pool}"], origin="lower", aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, extent=[-.5, len(ids) - .5, -.39, .01],
                       interpolation="nearest")
        ax.set(title=f"Latent 52 active + {title.lower()}", xlabel="Recorded cluster ID",
               ylabel="Time from endpoint (s)", yticks=[-.38, -.2, 0])
        ax.set_xticks(np.arange(0, len(ids), 20), ids[::20])
    fig.colorbar(im, ax=axes, label=label, shrink=.85, pad=.02)
    p.save_figure(fig, ATTRIBUTION, "aeon_wheel_context_traceback")
    plt.close("all")
    for stem in ("aeon_wheel_source_traceback", "aeon_traceback", "aeon_wheel_context_traceback"):
        for suffix in ("pdf", "png"):
            (FIGURES / f"{stem}.{suffix}").write_bytes((ATTRIBUTION / f"{stem}.{suffix}").read_bytes())


def main() -> None:
    traces = compute()
    render(traces)
    ids = traces["cluster_ids"]
    for name in p.FEATURES:
        z = reduce(traces[f"{name}_all"], "lag cluster -> cluster", "mean")
        order = np.argsort(-z)[:5]
        print(f"{name}: top clusters {ids[order].tolist()} attribution {z[order].round(4).tolist()}")


if __name__ == "__main__":
    main()
