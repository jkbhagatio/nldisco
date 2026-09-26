"""Regenerate the synthetic source activity tracebacks as encoder attribution.

Spatial latents (vanilla SED) and temporal latents (TW-SED) previously plotted the
conditional mean of standardized activity over active bins/windows. Here each source entry is
instead credited with activity times the encoder gradient of the latent's pre-activation, which
for the linear vanilla encoder is exactly encoder weight times activity. The active-bin/window
masks, standardization, and selected latents are unchanged from the existing figures.

Usage: uv run --no-sync python -m experiments.synthetic.scripts.attribution_traceback
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from beartype import beartype
from einops import rearrange, reduce
from jaxtyping import Float, Int, jaxtyped
from matplotlib.lines import Line2D

from experiments.synthetic.scripts import appendix, paper
from experiments.attribution import encoder_attribution, mean_encoder_attribution
from nldisco.config import EncoderConfig, SedConfig
from nldisco.model import Sed, build_sed

DEVICE = "cuda" if t.cuda.is_available() else "cpu"
BATCH = 4096
FIGURES = paper.PAPER


def _batches(windows: Float[np.ndarray, "window time unit"]):
    for start in range(0, len(windows), BATCH):
        yield t.from_numpy(np.ascontiguousarray(windows[start : start + BATCH]))


@beartype
def load_vanilla(seed: int) -> tuple[Sed, np.ndarray, np.ndarray]:
    """Rebuild one vanilla run from its checkpoint plus its training standardization."""
    ckpt = t.load(paper.RUNS / f"vanilla/seed_{seed:02d}/checkpoint.pt", map_location="cpu",
                  weights_only=False)
    c = ckpt["sed_config"]
    cfg = SedConfig(n_neurons=c["n_neurons"], seq_len=c["seq_len"],
                    dsed_topk_map={int(k): v for k, v in c["dsed_topk_map"].items()},
                    encoder=EncoderConfig(type=c["encoder_type"]),
                    inference_sparsity=c["inference_sparsity"])
    model = build_sed(cfg)
    model.load_state_dict({k: v.float() for k, v in ckpt["model_state_dict"].items()})
    pre = ckpt["preprocessing"]
    return model.to(DEVICE).eval(), pre["mean"].numpy(), pre["std"].numpy()


@beartype
def load_window_model(run: Path) -> Sed:
    """Rebuild one TW-SED run from its checkpoint and serialized configuration."""
    ckpt = t.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    c = json.loads(ckpt["config_json"])["sed_config"]
    cfg = SedConfig(n_neurons=c["n_neurons"], seq_len=c["seq_len"],
                    dsed_topk_map={int(k): v for k, v in c["dsed_topk_map"].items()},
                    encoder=EncoderConfig(**c["encoder"]),
                    inference_sparsity=c["inference_sparsity"])
    model = build_sed(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(DEVICE).eval()


@jaxtyped(typechecker=beartype)
def position_attribution_means(
    contributions: Float[np.ndarray, "sample unit"],
    positions: Float[np.ndarray, "sample"],  # noqa: F821
    edges: Float[np.ndarray, "edge"],  # noqa: F821
) -> tuple[Float[np.ndarray, "position unit"], Int[np.ndarray, "position"]]:  # noqa: F821
    """Average equally within position bins; include the last edge and leave empty bins NaN."""
    if (len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0)
            or not np.isfinite(positions).all() or not np.isfinite(contributions).all()):
        raise ValueError("Expected finite arrays and strictly increasing position edges")
    if np.any((positions < edges[0]) | (positions > edges[-1])):
        raise ValueError("Positions must lie within the bin edges")
    indices = np.minimum(np.searchsorted(edges, positions, side="right") - 1, len(edges) - 2)
    counts = np.bincount(indices, minlength=len(edges) - 1)
    totals = np.zeros((len(edges) - 1, contributions.shape[1]), dtype=np.float64)
    np.add.at(totals, indices, contributions)
    means = np.full_like(totals, np.nan)
    np.divide(totals, counts[:, None], out=means, where=counts[:, None] > 0)
    return means, counts


@jaxtyped(typechecker=beartype)
def spatial_attribution(
    raw: Float[np.ndarray, "time unit"], positions: Float[np.ndarray, "time"]  # noqa: F821
) -> dict:
    """Average active-bin encoder attributions overall and within 20 track-position bins."""
    with (paper.RUNS / "vanilla/summary/selected_features.csv").open() as stream:
        selected = list(csv.DictReader(stream))
    edges = np.linspace(0, 1, 21)
    result = {"position_edges_m": edges, "position_centers_m": (edges[:-1] + edges[1:]) / 2}
    for cell in range(4):
        rows = [r for r in selected if r["target"] == f"cell_{cell}"]
        profiles, position_profiles, position_counts = [], [], []
        for row in rows:
            seed, latent = int(row["seed"]), int(row["latent_idx"])
            model, mean, std = load_vanilla(seed)
            inputs = (raw - mean) / std + 1e-8
            with np.load(paper.RUNS / f"vanilla/seed_{seed:02d}/activations.npz") as acts:
                mask = (acts["latent_idx"] == latent) & (acts["activation_value"] > 0)
                bins = np.unique(acts["source_time_idx"][mask])
            windows = inputs[bins][:, None, :].astype(np.float32)
            contributions = np.concatenate([
                encoder_attribution(model, batch.to(DEVICE), latent).cpu().numpy()[:, 0, :]
                for batch in _batches(windows)
            ])
            profiles.append(contributions.mean(axis=0, dtype=np.float64))
            profile, counts = position_attribution_means(contributions, positions[bins], edges)
            position_profiles.append(profile)
            position_counts.append(counts)
            print(f"cell {cell}: seed {seed} latent {latent}: {len(bins):,} active bins", flush=True)
        result[f"cell_{cell}_attribution"] = np.stack(profiles)
        result[f"cell_{cell}_position_attribution"] = np.stack(position_profiles)
        result[f"cell_{cell}_position_active_counts"] = np.stack(position_counts)
        result[f"cell_{cell}_seeds"] = np.array([int(r["seed"]) for r in rows])
        result[f"cell_{cell}_latent_indices"] = np.array([int(r["latent_idx"]) for r in rows])
    return result


@beartype
def spatial_figure(results: dict, traces: dict) -> plt.Figure:
    """Draw one spatial attribution curve per neuron, matching the temporal figure's style."""
    paper._style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), sharex=True, sharey=True,
                             layout="constrained")
    positions = traces["position_centers_m"]
    for cell, ax in enumerate(axes.flat):
        values = traces[f"cell_{cell}_position_attribution"]
        counts = np.isfinite(values).sum(axis=0)
        means = np.divide(np.nansum(values, axis=0), counts,
                          out=np.full(values.shape[1:], np.nan), where=counts > 0)
        sd = np.sqrt(np.divide(np.nansum((values - means) ** 2, axis=0), counts - 1,
                                out=np.full_like(means, np.nan), where=counts > 1))
        center = results["parameters"]["centers_m"][cell]
        ax.plot(positions, means[:, 4:], color="#888888", alpha=0.30, linewidth=0.65)
        for unit, field_center in enumerate(results["parameters"]["centers_m"]):
            color = plt.colormaps["viridis"](field_center)
            ax.fill_between(positions, means[:, unit] - sd[:, unit],
                            means[:, unit] + sd[:, unit], color=color, alpha=0.12, linewidth=0)
            ax.plot(positions, means[:, unit], color=color,
                    linestyle=":" if unit == 3 else "-", linewidth=1.8,
                    label=f"Cell {unit + 1}")
        ax.axhline(0, color="#444444", linestyle="--", linewidth=0.6)
        ax.set(xlabel="Track position (m)", ylabel="Mean encoder attribution", xlim=(0, 1),
               title=f"Recovered field {cell + 1}: center {center:.1f} m")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles + [Line2D([], [], color="#888888", lw=0.8)],
                      labels + ["96 noise units"], frameon=False, fontsize=7, ncol=2)
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="viridis"),
                 ax=list(axes.flat), label="Place-field center (m); noise units gray", fraction=0.025)
    paper._save(fig, "supplement_unit_traceback_attribution")
    return fig


@beartype
def temporal_figure(results: dict, raw: Float[np.ndarray, "time unit"], traces: Optional[dict] = None) -> tuple:
    paper._style()
    counts = reduce(raw, "(time group) unit -> time unit", "sum", group=5).astype(np.float32)
    windows_all = rearrange(np.lib.stride_tricks.sliding_window_view(counts, 20, axis=0),
                            "window unit time -> window time unit")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True, sharey=True,
                             layout="constrained")
    exported = {}
    for ax, (direction, record) in zip(axes, appendix.selected_features(results).items()):
        data = record["artifacts"]
        feature = int(data[f"{direction}_feature"])
        if traces is None:
            model = load_window_model(record["path"])
            with np.load(record["path"] / "scores.npz") as scores:
                starts = scores["analysis_starts"]
                if np.any(starts % 5):
                    raise ValueError("Window starts must align with the 50-ms count bins")
                active = scores["analysis_activations"][:, feature] > 0
            windows = windows_all[starts[active] // 5]
            attribution = mean_encoder_attribution(model, _batches(windows), feature)
        else:
            if int(traces[f"{direction}_latent"]) != feature or int(traces[f"{direction}_seed"]) != record["summary"]["seed"]:
                raise ValueError("Saved attribution identity does not match the selected latent")
            attribution = traces[f"{direction}_attribution"]
            active = np.ones(int(traces[f"{direction}_active_window_count"]), dtype=bool)
        print(f"{direction}: seed {record['summary']['seed']} latent {feature}: "
              f"{int(active.sum()):,} active windows", flush=True)
        times = data["decoder_relative_time_s"] - data["trajectory_relative_time_s"][-1]
        ax.plot(times, attribution[:, 4:], color="#888888", alpha=0.30, linewidth=0.65)
        for cell, center in enumerate(results["parameters"]["centers_m"]):
            ax.plot(times, attribution[:, cell], color=plt.colormaps["viridis"](center),
                    linestyle=":" if cell == 3 else "-", linewidth=1.8, label=f"Cell {cell + 1}")
        ax.axhline(0, color="#444444", linestyle="--", linewidth=0.6)
        ax.set(xlabel="Time from window end (s)", ylabel="Mean encoder attribution",
               xlim=(-1, 0), title=f"{'Forward' if direction == 'right' else 'Backward'}: "
               f"latent {feature}, seed {record['summary']['seed']}\n"
               f"{int(active.sum()):,} active windows")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + [Line2D([], [], color="#888888", lw=0.8)],
                  labels + ["96 noise units"], frameon=False, fontsize=7, ncol=2)
        exported.update({f"{direction}_attribution": attribution,
                         f"{direction}_active_window_count": np.asarray(int(active.sum())),
                         f"{direction}_seed": np.asarray(record["summary"]["seed"]),
                         f"{direction}_latent": np.asarray(feature)})
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="viridis"),
                 ax=list(axes), label="Place-field center (m); noise units gray", fraction=0.025)
    paper._save(fig, "supplement_temporal_coactivity_attribution")
    return fig, dict(exported, relative_time_s=times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial-only", action="store_true", help="Update only the spatial traceback")
    args = parser.parse_args()
    results = paper.load_results()
    raw = np.load(paper.DATA / "spike_matrix.npy").T.astype(np.float32)
    if raw.shape != (60000, 100):
        raise ValueError("Paper simulation must contain 60000 bins and 100 units")
    spatial = spatial_attribution(raw, np.load(paper.DATA / "positions.npy"))
    np.savez_compressed(paper.PAPER / "appendix_unit_attribution.npz", **spatial)
    plt.close(spatial_figure(results, spatial))
    exports = [("supplement_unit_traceback_attribution", "supplement_unit_traceback")]
    if not args.spatial_only:
        fig, temporal = temporal_figure(results, raw)
        plt.close(fig)
        np.savez_compressed(paper.PAPER / "appendix_temporal_attribution.npz", **temporal)
        exports.append(("supplement_temporal_coactivity_attribution", "supplement_temporal_coactivity"))
    for stem, target in exports:
        for suffix in ("pdf", "png"):
            if (paper.PAPER / f"{stem}.{suffix}").exists():
                (FIGURES / f"{target}.{suffix}").write_bytes(
                    (paper.PAPER / f"{stem}.{suffix}").read_bytes())
    print("Top attributed units per spatial field:",
          {c: np.argsort(-spatial[f"cell_{c}_attribution"].mean(0))[:3].tolist() for c in range(4)})


if __name__ == "__main__":
    main()
