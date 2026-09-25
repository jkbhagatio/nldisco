"""Render the PCA appendix from frozen standalone results without refitting."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped

ROOT = Path(__file__).resolve().parents[3]
SAVED = ROOT / "experiments/synthetic/standalone_baselines_20260916/pca"
OUTPUT = ROOT / "experiments/synthetic/outputs/paper"
COLORS = ("#2678b2", "#d17a18", "#269473", "#9457a4")


@jaxtyped(typechecker=beartype)
def display_scale(curve: Float[np.ndarray, "40"]) -> Float[np.ndarray, "40"]:
    """Standardize a spatial curve for display only, preserving its signed shape."""
    return (curve - curve.mean()) / curve.std()


@beartype
def metric_label(value: float) -> str:
    """Follow the appendix's existing floor-to-three-decimal convention."""
    return f"{np.floor(value * 1000) / 1000:.3f}"


@beartype
def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save the publication figure and its raster preview."""
    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


@beartype
def spatial_figure(data: dict, selected: pd.DataFrame) -> None:
    """Show independent winners, distinct assignments, and every candidate."""
    corr = data["shape_correlation"]
    winners = np.abs(corr).argmax(axis=1)
    fig = plt.figure(figsize=(8.0, 5.7), layout="constrained")
    grid = fig.add_gridspec(3, 4, height_ratios=(1.6, 1.1, 1.05))
    for cell in range(4):
        pc = int(winners[cell])
        distinct = int(selected.loc[selected.cell == cell + 1, "pc"].item()) - 1
        sign = float(np.sign(corr[cell, pc]))
        ax = fig.add_subplot(grid[0, cell])
        ax.plot(data["position_centers"], display_scale(data["exact_fields"][cell]),
                "k--", lw=1.1, label="Exact field")
        ax.plot(data["position_centers"],
                display_scale(sign * data["spatial_tuning"][pc]), color=COLORS[cell],
                lw=1.3, label=f"{'+' if sign > 0 else '−'}PC{pc + 1}: r={metric_label(abs(float(corr[cell, pc])))}")
        if distinct != pc:
            distinct_sign = np.sign(corr[cell, distinct])
            ax.plot(data["position_centers"],
                    display_scale(distinct_sign * data["spatial_tuning"][distinct]),
                    color="0.45", ls=":", lw=1.2,
                    label=f"Distinct PC{distinct + 1}: r={metric_label(abs(float(corr[cell, distinct])))}")
        ax.axhline(0, color="0.85", lw=.5)
        ax.set(title=f"{chr(97 + cell)}  Cell {cell + 1}", xlabel="Midpoint position (m)",
               xlim=(0, 1), xticks=(0, .5, 1))
        if cell == 0:
            ax.set_ylabel("Standardized tuning")
        ax.legend(fontsize=5.9, loc="upper right", framealpha=.85,
                  handlelength=1.3, borderpad=.25)
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + .45 * (high - low))

        ax = fig.add_subplot(grid[1, cell])
        energy = data["unit_energy"][pc]
        bars = np.r_[energy[:4], energy[4:].sum()] * 100
        ax.bar(np.arange(5), bars, color=(*COLORS, "0.65"), width=.7)
        if distinct != pc:
            alternative = data["unit_energy"][distinct]
            ax.bar(np.arange(5), np.r_[alternative[:4], alternative[4:].sum()] * 100,
                   width=.4, fill=False, edgecolor="black", lw=.8, ls=":")
        ax.set(title=f"{chr(101 + cell)}  PC{pc + 1} source composition", ylim=(0, 100),
               xticks=np.arange(5), xticklabels=("1", "2", "3", "4", "Noise"),
               xlabel="Source cells")
        if cell == 0:
            ax.set_ylabel("Squared loading (%)")

    ax = fig.add_subplot(grid[2, :])
    image = ax.imshow(np.abs(corr), aspect="auto", cmap="viridis", vmin=0, vmax=1,
                      extent=(.5, 128.5, 4.5, .5), interpolation="nearest")
    ax.set(title="i  Spatial tuning similarity across all 128 principal components",
           xlabel="Principal component", ylabel="Target cell", yticks=(1, 2, 3, 4),
           xticks=(1, 16, 32, 48, 64, 80, 96, 112, 128))
    fig.colorbar(image, ax=ax, label="|Pearson r|", fraction=.025, pad=.02)
    save_figure(fig, "supplement_pca_spatial")


@beartype
def direction_figure(data: dict, fitted: dict, metadata: pd.DataFrame, summary: dict) -> None:
    """Show native PC4 scores, temporal weights, and both trajectory pools."""
    scores = fitted["scores"][:, 3]
    weights = np.zeros(len(scores))
    position_bins = np.clip((metadata.position.to_numpy() * 40).astype(int), 0, 39)
    directions = metadata.direction.to_numpy()
    for record in summary["clean_position_bins"]:
        for sign, count in ((1, record["right"]), (-1, record["left"])):
            mask = data["clean_mask"] & (position_bins == record["bin"]) & (directions == sign)
            weights[mask] = record["weight"] / count
    np.testing.assert_allclose([weights[directions == sign].sum() for sign in (1, -1)],
                               [209, 209])
    fig = plt.figure(figsize=(8.0, 6.5), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.2, 1.2))
    ax = fig.add_subplot(grid[0, 0])
    edges = np.linspace(scores[data["clean_mask"]].min(),
                        scores[data["clean_mask"]].max(), 23)
    for sign, color, label in ((1, COLORS[0], "Forward"), (-1, COLORS[1], "Backward")):
        mask = weights > 0
        mask &= directions == sign
        ax.hist(scores[mask], bins=edges, weights=weights[mask] / weights[mask].sum(),
                histtype="step", lw=1.4, color=color, label=label)
    ax.set(title="a  PC4 scores during clean crossings", xlabel="Native PC4 score",
           ylabel="Matched fraction per bin")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    auc = float(summary["selected_direction"]["best_oriented_clean_auc"])
    ax.text(.02, .96, f"Position-matched AUROC: {metric_label(auc)}", va="top",
            transform=ax.transAxes, fontsize=7)

    ax = fig.add_subplot(grid[0, 1])
    loadings = rearrange(fitted["components"][3], "(time unit) -> time unit", time=20)
    for cell, color in enumerate(COLORS):
        ax.plot(np.arange(20) * .05 + .025, loadings[:, cell], color=color,
                lw=1.2, label=f"Cell {cell + 1}")
    ax.axhline(0, color="0.65", lw=.6)
    ax.set(title="b  PC4 place-cell loadings", xlabel="Time within window (s)",
           ylabel="Signed loading", xlim=(0, 1))
    ax.legend(fontsize=6.4, ncol=2, loc="upper right")

    for row, (name, sign, tail) in enumerate((("forward", 1, "Lowest"),
                                            ("backward", -1, "Highest")), start=1):
        for col, pool in enumerate(("unrestricted", "clean_crossing")):
            ax = fig.add_subplot(grid[row, col])
            trajectories = data[f"{name}_{pool}_trajectories"]
            indices = data[f"{name}_{pool}_indices"]
            preferred = int((data["direction"][indices] == sign).sum())
            assert len(indices) == 16
            np.testing.assert_array_equal(data["direction"][indices],
                                           np.sign(trajectories[:, -1] - trajectories[:, 0]))
            assert np.all(np.diff(np.sort(fitted["starts"][indices])) >= 100)
            ax.plot(np.arange(100) * .01, trajectories.T, color="0.65", lw=.55, alpha=.8)
            ax.plot(np.arange(100) * .01, trajectories.mean(axis=0), color=COLORS[row - 1], lw=1.7)
            for center in (.2, .4):
                ax.axhline(center, color="0.45", ls=":", lw=.7)
            pool_label = "all windows" if col == 0 else "clean crossings"
            ax.set(title=f"{chr(99 + (row - 1) * 2 + col)}  {tail} PC4 scores: {pool_label}\n"
                         f"{name.capitalize()} net motion: {preferred}/16",
                   xlabel="Time within window (s)", ylabel="Position (m)",
                   xlim=(0, 1), ylim=(0, 1), yticks=(0, .2, .4, .6, .8, 1))
    save_figure(fig, "supplement_pca_direction")


@beartype
def main() -> None:
    """Verify frozen inputs, then render only the two new PCA appendix figures."""
    summary = json.loads((SAVED / "summary.json").read_text())
    for relative, expected in summary["hashes"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    with np.load(SAVED / "diagnostics.npz") as saved:
        data = {key: saved[key] for key in saved.files}
    with np.load(SAVED / "pca_fit.npz") as saved:
        fitted = {key: saved[key] for key in saved.files}
    metadata = pd.read_csv(ROOT / "experiments/synthetic/outputs/runs/temporal_transformer/analysis_windows.csv")
    np.testing.assert_array_equal(fitted["starts"], metadata.start)
    np.testing.assert_array_equal(data["clean_mask"], metadata.ramp_clean)
    assert fitted["scores"].shape == (11981, 128)
    assert fitted["components"].shape == (128, 2000)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    spatial_figure(data, pd.read_csv(SAVED / "spatial_selected.csv"))
    direction_figure(data, fitted, metadata, summary)
    inputs = [SAVED / name for name in ("summary.json", "diagnostics.npz", "pca_fit.npz",
                                        "spatial_selected.csv")]
    outputs = sorted(OUTPUT.glob("supplement_pca_*.pdf")) + sorted(OUTPUT.glob("supplement_pca_*.png"))
    manifest = {
        "scope": "Render frozen PCA results; no fitting or changes to original standalone artifacts",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in inputs},
        "outputs": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in outputs},
    }
    (OUTPUT / "pca_supplement_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Rendered PCA spatial/direction supplement figures from verified saved artifacts.")


if __name__ == "__main__":
    main()
