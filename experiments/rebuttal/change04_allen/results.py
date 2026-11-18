"""Score the selected stimulus features on held-out presentations and render outputs.

The four latents and their fixed stimulus conditions were chosen by inspecting the
training split only (``search.py``); this module scores them on both splits and
plots held-out tuning. All conditions label a 50-ms response bin with the stimulus
shown 50 ms earlier (``data.DELAY_S``).

Example:
    python -m experiments.rebuttal.change04_allen.results
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from beartype import beartype
from sklearn.metrics import roc_curve

from experiments.churchland.sweeps.s10_endpoint_v1.publication_style import COLORMAP, COLORS
from experiments.rebuttal.change04_allen.data import CACHE, DELAY_S
from experiments.rebuttal.change04_allen.features import MOVIE_SEGMENT_FRAMES, association
from experiments.rebuttal.change04_allen.train import RUNS

REPO = Path(__file__).resolve().parents[3]
FIGURES = REPO / "paper/iclr_paper/figures/rebuttal/change04_allen"
RUN = "s1_z"
D_K = "320/36"
BLUE, LIGHT = COLORS["nldisco"], "#8ab6d6"
# Additional gabor latents shown as receptive-field maps: latent -> training peak (x, y).
RF_LATENTS = {196: (10.0, 10.0), 84: (0.0, 40.0), 49: (40.0, -10.0), 22: (-40.0, -20.0)}


def _block(bins: pd.DataFrame, family: str) -> np.ndarray:
    return bins.stim.isin([family, f"{family}_gray"]).to_numpy()


def _on(bins: pd.DataFrame, family: str) -> np.ndarray:
    return (bins.stim == family).to_numpy()


@dataclass(frozen=True)
class Feature:
    """A fixed stimulus condition, its comparison population and selected latent."""

    key: str
    title: str
    latent: int
    population: Callable[[pd.DataFrame], np.ndarray]
    condition: Callable[[pd.DataFrame], np.ndarray]


FEATURES = (
    Feature("gabor", "Gabor at (10°, 10°)", 196,
            lambda b: _block(b, "gabors"),
            lambda b: _on(b, "gabors") & (b.x_position == 10).to_numpy()
            & (b.y_position == 10).to_numpy()),
    Feature("temporal_frequency", "Drifting grating at ≥8 Hz", 65,
            lambda b: _block(b, "drifting_gratings"),
            lambda b: _on(b, "drifting_gratings") & (b.temporal_frequency >= 8).to_numpy()),
    Feature("scene", "Natural image 19", 70,
            lambda b: _block(b, "natural_scenes"),
            lambda b: _on(b, "natural_scenes") & (b.frame == 19).to_numpy()),
    Feature("movie", "Natural movie 3, second 115", 112,
            lambda b: _block(b, "natural_movie_three"),
            lambda b: _on(b, "natural_movie_three")
            & (b.frame // MOVIE_SEGMENT_FRAMES == 115).to_numpy()),
)


@beartype
def score(acts: np.ndarray, bins: pd.DataFrame) -> pd.DataFrame:
    """Association metrics for every feature on the train and held-out splits."""
    rows = []
    valid = ~bins.invalid.to_numpy()
    for f in FEATURES:
        for split in ("train", "test"):
            pop = f.population(bins) & valid & (bins.split == split).to_numpy()
            pos = f.condition(bins)[pop]
            z = acts[pop][:, [f.latent]].astype(np.float32)
            m = {k: float(v[0]) for k, v in association(z, pos).items()}
            rows.append(dict(feature=f.key, title=f.title, latent=f.latent, split=split,
                             n_pos=int(pos.sum()), n_neg=int((~pos).sum()), **m))
    return pd.DataFrame(rows)


def _p_active(acts, bins, mask, latent, by):
    return pd.Series(acts[mask, latent] > 0).groupby([np.asarray(bins[c])[mask] for c in by]
                                                     if isinstance(by, list) else
                                                     np.asarray(by)[mask]).mean()


def _letter(ax, letter, x=-0.18, y=1.12):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold")


@beartype
def render_figure(acts: np.ndarray, bins: pd.DataFrame, path: Path) -> dict:
    """Held-out tuning: gabor receptive fields, temporal frequency, image and movie time."""
    test = ((bins.split == "test") & ~bins.invalid).to_numpy()
    tuning = {}
    fig = plt.figure(figsize=(7.2, 4.5), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.1])
    axes = top.subplots(1, len(RF_LATENTS))
    gabor = test & _on(bins, "gabors")
    for i, (ax, (latent, peak)) in enumerate(zip(axes, RF_LATENTS.items())):
        grid = _p_active(acts, bins, gabor, latent, ["y_position", "x_position"]).unstack()
        tuning[f"rf_{latent}"] = grid.to_dict()
        im = ax.imshow(grid.to_numpy(), origin="lower", cmap=COLORMAP, vmin=0, vmax=1,
                       extent=(-45, 45, -45, 45))
        ax.plot(*peak, marker="+", color="w", ms=6, mew=1.2)
        ax.set(xticks=[-40, 0, 40], yticks=[-40, 0, 40], title=f"Latent {latent}",
               xlabel="Gabor azimuth (°)")
        if i == 0:
            ax.set_ylabel("Gabor elevation (°)")
            _letter(ax, "a", -0.45, 1.1)
        else:
            ax.set_yticklabels([])
    top.colorbar(im, ax=axes, shrink=0.85, label="P(latent active)")

    ax_tf, ax_scene, ax_movie = bottom.subplots(1, 3, width_ratios=[1, 1.25, 1.35])
    dg = test & _on(bins, "drifting_gratings") & bins.temporal_frequency.notna().to_numpy()
    by = _p_active(acts, bins, dg, 65, ["orientation", "temporal_frequency"]).unstack()
    for _, row in by.iterrows():
        ax_tf.plot(row.index, row.to_numpy(), color=LIGHT, lw=0.7)
    mean = _p_active(acts, bins, dg, 65, ["temporal_frequency"])
    ax_tf.plot(mean.index, mean.to_numpy(), "o-", color=BLUE, ms=3, lw=1.3, label="All directions")
    gray = test & (_block(bins, "drifting_gratings") & ~dg & ~(bins.stim == "none").to_numpy())
    ax_tf.axhline((acts[gray, 65] > 0).mean(), ls=":", color=".5", lw=0.8)
    ax_tf.text(15, (acts[gray, 65] > 0).mean() + 0.03, "gray/blank", color=".4", fontsize=6,
               ha="right")
    ax_tf.axvspan(6, 20, color=BLUE, alpha=0.08)
    ax_tf.set(xscale="log", xticks=[1, 2, 4, 8, 15], xticklabels=[1, 2, 4, 8, 15], ylim=(0, 1.03),
              xlabel="Temporal frequency (Hz)", ylabel="P(latent active)",
              title="Latent 65: drifting gratings")
    ax_tf.minorticks_off()
    tuning["tf_65"] = {"by_direction": by.to_dict(), "mean": mean.to_dict()}
    _letter(ax_tf, "b")

    train = ((bins.split == "train") & ~bins.invalid).to_numpy()
    ns_on = _on(bins, "natural_scenes")
    order = _p_active(acts, bins, train & ns_on, 70, ["frame"]).sort_values(ascending=False).index
    images = _p_active(acts, bins, test & ns_on, 70, ["frame"]).reindex(order)
    ax_scene.bar(np.arange(len(images)), images.to_numpy(), width=1.0,
                 color=[BLUE if k == 19 else LIGHT for k in images.index])
    ax_scene.set(xlim=(-1, len(images)), ylim=(0, 1.03), xticks=[0, 50, 100],
                 xlabel="Image rank on training split (118 images + blank)",
                 title="Latent 70: natural scenes")
    ax_scene.text(2.5, images.iloc[0], "image 19", color=BLUE, va="center", fontsize=6)
    tuning["scene_70"] = images.to_dict()
    _letter(ax_scene, "c", -0.12)

    second = bins.frame // MOVIE_SEGMENT_FRAMES
    nm = _on(bins, "natural_movie_three")
    seconds = _p_active(acts, bins, test & nm, 112, second)
    reference = _p_active(acts, bins, train & nm, 112, second)
    ax_movie.bar(seconds.index, seconds.to_numpy(), width=1.0,
                 color=[BLUE if s == 115 else LIGHT for s in seconds.index], label="Held-out")
    ax_movie.plot(reference.index, reference.to_numpy(), color=".25", lw=0.6,
                  label="Training mean")
    ax_movie.legend(frameon=False, loc="upper left", fontsize=6)
    ax_movie.set(xlim=(-1, 120), ylim=(0, 1.03), xlabel="Natural movie 3 time (s)",
                 title="Latent 112: natural movie 3")
    tuning["movie_112"] = {"test": seconds.to_dict(), "train": reference.to_dict()}
    _letter(ax_movie, "d", -0.12)
    for ax in (ax_tf, ax_scene, ax_movie):
        ax.spines[["top", "right"]].set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=210)
    plt.close(fig)
    return tuning


@beartype
def render_table(scores: pd.DataFrame, path: Path) -> None:
    """Row group per feature with training and held-out association scores."""
    lines = [r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrrrrr@{}}", r"\toprule",
             r"Split & Latent & \metric{Cov} & \metric{Spec} & \metric{Sel} & \metric{AUROC}"
             r" & Bins in $C$ & Bins in $\neg C$ \\"]
    for f in FEATURES:
        lines += [r"\midrule", r"\multicolumn{8}{l}{\textit{" + f.title.replace("≥", r"$\geq$")
                  .replace("°", r"$^\circ$") + r"}} \\"]
        for split, label in (("train", "Train (80\\%)"), ("test", "Held-out (20\\%)")):
            r = scores[(scores.feature == f.key) & (scores.split == split)].iloc[0]
            lines.append(f"{label} & {f.latent} & {r['cov']:.3f} & {r.spec:.3f} & {r.sel:.3f} & "
                         f"{r.auroc:.3f} & {r.n_pos:,} & {r.n_neg:,} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    FIGURES.mkdir(parents=True, exist_ok=True)
    acts = np.load(RUNS / RUN / "acts.npy")
    bins = pd.read_parquet(CACHE / "bins.parquet")
    scores = score(acts, bins)
    test = ((bins.split == "test") & ~bins.invalid).to_numpy()
    # Held-out scores for the extra receptive-field latents at their training peaks.
    rf_scores = {}
    for latent, (x, y) in RF_LATENTS.items():
        pop = test & _block(bins, "gabors")
        pos = (_on(bins, "gabors") & (bins.x_position == x).to_numpy()
               & (bins.y_position == y).to_numpy())[pop]
        rf_scores[latent] = {k: float(v[0]) for k, v in
                             association(acts[pop][:, [latent]].astype(np.float32), pos).items()}
    rocs = {}
    for f in FEATURES:
        pop = f.population(bins) & test
        fpr, tpr, _ = roc_curve(f.condition(bins)[pop], acts[pop, f.latent].astype(np.float32))
        rocs[f.key] = {"fpr": fpr[::max(1, len(fpr) // 200)].tolist(),
                       "tpr": tpr[::max(1, len(tpr) // 200)].tolist()}
    tuning = render_figure(acts, bins, FIGURES / "allen_stimulus_features")
    render_table(scores, FIGURES / "allen_feature_metrics.tex")
    summary = json.loads((RUNS / RUN / "summary.json").read_text())
    out = dict(run=RUN, delay_s=DELAY_S, activation_rule="native z > 0", model=summary,
               scores=scores.to_dict("records"), rf_latent_test_scores=rf_scores, rocs=rocs,
               tuning={k: {str(i): v for i, v in d.items()} if isinstance(d, dict) else d
                       for k, d in tuning.items()})
    (FIGURES / "allen_feature_metrics.json").write_text(json.dumps(out, indent=1, default=str))
    print(scores.round(3).to_string(index=False))
    print(json.dumps(rf_scores, indent=1))


if __name__ == "__main__":
    main()
