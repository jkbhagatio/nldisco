"""Render the Aeon appendix in the Churchland style from saved analysis results.

Run with ``uv run --no-config --no-sync --with cairosvg python -m experiments.aeon.paper_results``.
This renderer changes presentation only; it does not fit models or change feature
masks. Attribution inputs are gradient-times-input maps, not activity-only means.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from beartype import beartype
from einops import rearrange, reduce

from experiments.aeon.sweep_20260921 import paper_results as source
from experiments.churchland.sweeps.s10_endpoint_v1.publication_style import COLORS, COLORMAP

REPO = Path(__file__).resolve().parents[2]
FIGURES = REPO / "experiments/aeon/outputs/paper"
DECODE = REPO / "experiments/aeon/outputs/appendix_decoders_20260925/results.json"
TITLES = {
    "wheel": "Wheel activity",
    "area": "Smaller area near the patch",
    "speed": "Highest centroid-speed quintile",
    "direction": "Radially outward movement",
}
BLUE = COLORS["nldisco"]


@beartype
def save(fig: plt.Figure, name: str) -> None:
    """Save matching vector and raster figures in the manuscript directory."""
    for extension in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{extension}", dpi=210)
    plt.close(fig)


@beartype
def render_table(scores: pd.DataFrame) -> None:
    """Use one feature row group, with native association scores and saved decoding."""
    decoders = {(row["feature"], row["space"]): row
                for row in json.loads(DECODE.read_text())["results"]}
    lines = [r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrrrr@{}}", r"\toprule",
             r"Method & $D/k$ & Latent & \metric{Cov} & \metric{Spec} & \metric{Sel} & \metric{AUROC} & \shortstack{Single\\Decode} & \shortstack{All\\Decode} \\"]
    records = []
    for name, title in TITLES.items():
        row = scores.loc[name]
        decoding = {space: decoders[name, space]["test"]["balanced_accuracy"]
                    for space in ("single", "all")}
        assert all(decoders[name, space]["D"] == int(row.D)
                   and decoders[name, space]["latent"] == int(row.latent)
                   for space in decoding)
        values = [f"{row.tpr:.4f}", f"{1-row.fpr:.4f}", f"{row.selectivity:.3f}",
                  f"{row.auroc:.3f}", f"{decoding['single']:.4f}", f"{decoding['all']:.4f}"]
        lines.extend([r"\midrule", r"\multicolumn{9}{l}{\textit{" + title + r"}} \\",
                      " & ".join(["NLDisco", f"{int(row.D)}/{int(row.K)}", str(int(row.latent)),
                                  *values]) + r" \\"])
        records.append(dict(feature=name, D=int(row.D), k=int(row.K), latent=int(row.latent),
                            coverage=float(row.tpr), specificity=float(1-row.fpr),
                            selectivity=float(row.selectivity), auroc=float(row.auroc),
                            single_decode=decoding["single"], all_decode=decoding["all"],
                            positive_windows=int(row.n_condition), negative_windows=int(row.n_outside)))
    lines.extend([r"\bottomrule", r"\end{tabular*}", ""])
    (FIGURES / "aeon_feature_metrics.tex").write_text("\n".join(lines))
    (FIGURES / "aeon_appendix_metrics.json").write_text(json.dumps(dict(
        association_source=str(source.OUT / "feature_scores.csv"),
        activation_rule="native z > 0", decode_source=str(DECODE), rows=records), indent=2) + "\n")


@beartype
def panel_letter(ax: plt.Axes, letter: str) -> None:
    """Match the bold lowercase panel labels used in Churchland figures."""
    ax.text(-.12, 1.13, letter, transform=ax.transAxes, fontsize=10, fontweight="bold")


@beartype
def render_features(scores: pd.DataFrame) -> None:
    """Plot the existing area/speed tuning and outward-direction ROC."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), layout="constrained")
    for ax, name, label, letter in zip(axes, ("area", "speed"),
                                     ("Area quintile (within 100 px of patch)", "Centroid-speed quintile"), "ab"):
        with np.load(source.OUT / f"tuning_{name}.npz") as tuning:
            colors = [BLUE if i == (0 if name == "area" else 4) else "#8ab6d6" for i in range(5)]
            ax.bar(np.arange(1, 6), tuning["probability"], color=colors)
        row = scores.loc[name]
        ax.set(xticks=np.arange(1, 6), ylim=(0, 1.03), xlabel=label, ylabel="P(latent active)",
               title=f"{TITLES[name]}: latent {int(row.latent)}")
        panel_letter(ax, letter)
    save(fig, "aeon_additional_features")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), layout="constrained")
    with np.load(source.OUT / "tuning_direction.npz") as tuning:
        centers = (tuning["edges"][:-1] + tuning["edges"][1:]) / 2
        axes[0].axvspan(-45, 45, color=BLUE, alpha=.10)
        axes[0].plot(centers, tuning["probability"], "o-", ms=3, lw=1.2, color=BLUE)
    axes[0].set(xticks=[-180, -90, 0, 90, 180], ylim=(0, 1.03),
                xlabel="Heading relative to outward direction (deg)",
                ylabel="P(latent active)", title="Radially outward movement: latent 10")
    with np.load(source.OUT / "roc_direction.npz") as roc:
        axes[1].plot(roc["fpr"], roc["tpr"], color=BLUE, lw=1.25)
    axes[1].plot([0, 1], [0, 1], ":", color=".7", lw=.7)
    row = scores.loc["direction"]
    axes[1].scatter(row.fpr, row.tpr, color=BLUE, s=20)
    axes[1].text(.97, .08, f"Sel = {row.selectivity:.3f}\nAUROC = {row.auroc:.3f}",
                 ha="right", transform=axes[1].transAxes)
    axes[1].set(xlim=(-.01, 1.01), ylim=(-.01, 1.01), xlabel="False-positive rate",
                ylabel="True-positive rate", title="Outward versus other fast movement")
    for ax, letter in zip(axes, "ab"):
        panel_letter(ax, letter)
    save(fig, "aeon_direction")


@beartype
def render_patch_control() -> None:
    """Recompute both temporal panels from native latent and behavior states."""
    from .sweep_20260921.temporal_ticks import prepare
    states = prepare(FIGURES)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.4), layout='constrained')
    rows = []
    for ax, (lo, hi), letter in zip(axes, ((1.2, 1.3), (11., 11.1)), 'ab'):
        selected = (states['hours'] >= lo) & (states['hours'] < hi)
        counts = {}
        for y, name, color in ((2, 'patch', '.45'), (1, 'wheel', BLUE), (0, 'latent', COLORS['cebra'])):
            ax.hlines(y, lo, hi, color='.85', lw=.6)
            for active, ink in ((~states[f'valid_{name}'], '.82'), (states[name], color)):
                take = active & selected
                ax.plot(states['hours'][take], np.full(int(take.sum()), y), ls='', marker='|',
                        markersize=10, markeredgewidth=.3, color=ink, rasterized=True)
            counts[name] = int((states[name] & selected).sum())
        ax.set(xlim=(lo, hi), ylim=(-.5, 2.5), yticks=[0, 1, 2],
               yticklabels=['Latent active', 'Wheel active', 'At patch'],
               xlabel='Time from recording start (hours)', title=f'Hours {lo:g}–{hi:g}')
        ax.text(-.08, 1.08, letter, transform=ax.transAxes, fontweight='bold')
        ax.grid(axis='x', alpha=.15)
        rows.append(dict(hours=[lo, hi], active_endpoint_counts=counts))
    save(fig, 'aeon_wheel_vs_patch')
    (FIGURES/'wheel_tick_comparison.json').write_text(json.dumps(dict(
        regenerated=rows, activation_rule='native z > 0',
        note='Regenerated from frozen native activations. Tick positions differ from the current manuscript SVG; paper is unchanged.'), indent=2)+'\n')


@beartype
def render_tracebacks() -> None:
    """Pair saved encoder attributions with decoder dictionaries in the S14 style."""
    with np.load(source.OUT / "attribution/traceback_attribution.npz") as data:
        traces = dict(data)
    ids = traces["cluster_ids"]
    weights = {d: torch.load(source.ROOT / f"d{d}/k12_seed0/checkpoint.pt", map_location="cpu",
                            weights_only=True)["state_dict"]["decoder.weight"].numpy()
               for d in (192, 256)}
    z = reduce(traces["wheel_all"], "lag cluster -> cluster", "mean")
    fig, ax = plt.subplots(figsize=(7.2, 2.3), layout="constrained")
    ax.axhline(0, color=".7", lw=.7)
    ax.vlines(ids, 0, z, color=BLUE, lw=.7)
    ax.scatter(ids, z, color=BLUE, s=8)
    ax.set(xlabel="Recorded cluster ID", ylabel="Mean encoder attribution",
           title="Wheel activity: latent 52")
    save(fig, "aeon_wheel_source_traceback")

    fig, axes = plt.subplots(4, 2, figsize=(7.2, 7.8), layout="constrained")
    for row_axes, (name, title) in zip(axes, TITLES.items()):
        d, latent, _ = source.FEATURES[name]
        attribution = rearrange(traces[f"{name}_all"], "lag cluster -> cluster lag")
        for ax, array, heading, label in zip(row_axes, (attribution, weights[d][latent]),
                                             ("Encoder attribution", "Temporal decoder weights"),
                                             ("Mean attribution", "Decoder weight")):
            assert array.shape == (len(ids), 20) and np.isfinite(array).all()
            limit = float(np.quantile(abs(array), .99))
            im = ax.imshow(array, aspect="auto", origin="lower", cmap=COLORMAP,
                           vmin=-limit, vmax=limit, extent=[-.39, .01, -.5, len(ids)-.5],
                           interpolation="nearest")
            ax.set_title(f"{title}: {latent}\n{heading}", fontsize=7)
            ax.set(xlabel="Time relative to endpoint (s)", ylabel="Recorded cluster ID", xticks=[-.38, -.2, 0])
            ax.set_yticks(np.arange(0, len(ids), 20), ids[::20])
            fig.colorbar(im, ax=ax, shrink=.8).set_label(label)
    save(fig, "aeon_traceback")


@beartype
def verify() -> dict:
    """Recalculate all native association and decoder scores from saved observations."""
    from experiments.decoder_metrics import binary_metrics
    from sklearn.metrics import roc_auc_score
    scores=pd.read_csv(source.OUT/'feature_scores.csv').set_index('feature')
    records=[]
    behavior=pd.read_parquet(source.ROOT/'data/behavior.parquet')
    with np.load(source.OUT/'feature_arrays.npz') as data:
        conditions, _=source.definitions(behavior, data['endpoint_bins'])
        for name in TITLES:
            np.testing.assert_array_equal(conditions[name][0], data[f'condition_{name}'])
            np.testing.assert_array_equal(conditions[name][1], data[f'valid_{name}'])
            valid=data[f'valid_{name}'];labels=data[f'condition_{name}'][valid]
            values=data[f'values_{name}'][valid];active=values>0
            cov=float(active[labels].mean());fpr=float(active[~labels].mean())
            measured=[cov,1-fpr,cov/(cov+fpr),roc_auc_score(labels,values)]
            row=scores.loc[name]
            np.testing.assert_allclose(measured,[row.tpr,1-row.fpr,row.selectivity,row.auroc],atol=1e-12)
            records.append(dict(feature=name,coverage=cov,specificity=1-fpr,selectivity=measured[2],auroc=measured[3]))
    for row in json.loads(DECODE.read_text())['results']:
        path=DECODE.parent/row['feature']/f"{row['space']}_test_predictions.npz"
        with np.load(path) as saved:
            predicted=saved['probabilities']>=row['threshold']
            measured=binary_metrics(saved['labels'],predicted)
        np.testing.assert_allclose(measured['balanced_accuracy'],row['test']['balanced_accuracy'],atol=1e-12)
    report=dict(association=records,decoder_scores_verified=True,
                wheel_tick_plot='Regenerated from native activations; differs from current manuscript tick geometry, which is unchanged.')
    FIGURES.mkdir(parents=True,exist_ok=True)
    (FIGURES/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


@beartype
def render_main() -> None:
    """Replot Figure 5 using the saved numerical tuning and ROC arrays."""
    import shutil
    from unittest.mock import patch
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    from experiments.plot_style import format_plot_label
    for name in ('feature_scores.csv','tuning_wheel.npz','roc_wheel.npz','wheel_roc_marker.json'):
        shutil.copyfile(source.OUT/name,FIGURES/name)
    original=Figure.savefig
    def save_with_typography(figure,*args,**kwargs):
        for text in figure.findobj(match=Text):text.set_text(format_plot_label(text.get_text()))
        return original(figure,*args,**kwargs)
    with patch.object(Figure,'savefig',save_with_typography):source.render_main(FIGURES)


@beartype
def main() -> None:
    """Rebuild Figure 5, Table S3, and all five Aeon appendix figures."""
    verify();render_main()
    plt.rcParams.update({'font.size':7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    scores=pd.read_csv(source.OUT/'feature_scores.csv').set_index('feature')
    render_table(scores);render_features(scores);render_patch_control();render_tracebacks()


if __name__=='__main__':
    main()
