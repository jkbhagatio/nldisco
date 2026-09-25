"""Render the consolidated Churchland appendix from existing results.

Run from the repository root with::

    uv run --no-sync python -m paper.iclr_paper.scripts.render_churchland_appendix

Table 1 supplies displayed scalar scores for its features. Saved results supply
the supplementary scores and decoding intervals. Empirical ROC vertices and
native activation markers are retained without transformations. The JSON export
keeps measured and displayed quantities separate for subsequent manuscript review.
"""

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype
from matplotlib.lines import Line2D

from experiments.churchland.sweeps.s10_endpoint_v1 import feature_decoding as decoder
from experiments.churchland.sweeps.s10_endpoint_v1 import nice_cutoffs_revision as revision
from experiments.churchland.sweeps.s10_endpoint_v1.matched192_tables import floor_three
from experiments.churchland.sweeps.s10_endpoint_v1.paper_figures import probability_curve
from experiments.churchland.sweeps.s10_endpoint_v1.publication_style import COLORS

ROOT = Path(__file__).resolve().parents[3]
FIGURES = ROOT / "paper/iclr_paper/figures"
SOURCE = ROOT / "experiments/churchland/outputs/s10_endpoint_revision_v4_nice_cutoffs"
DECODE = ROOT / "experiments/outputs/decoder_balanced_accuracy_20260923/churchland"
FEATURE_ORDER = ("recent_braking", "fast_target_specific", "early_leftward", "late_target_specific")
MAIN = ("recent_braking", "fast_target_specific")
TITLES = dict(zip(decoder.FEATURES, (
    "Recent deceleration", "Early leftward movement",
    "Fast leftward target reach", "Late, target-specific reach",
)))
MARKERS = {"illustrative": "o", "sel": "^", "auroc": "D"}
MAIN_ORDER = (("nldisco", "illustrative"), ("cebra", "sel"), ("cebra", "auroc"),
              ("langevinflow", "sel"), ("langevinflow", "auroc"))


@beartype
def selected_rows(selections: list[dict]) -> list[dict]:
    """Select the manuscript's specified coordinates, preserving feature order."""
    index = {(r["feature"], r["method"], r["criterion"]): r for r in selections}
    return [index[feature, method, criterion]
            for feature in FEATURE_ORDER
            for method in COLORS
            for criterion in (("illustrative",) if method == "nldisco" and feature in MAIN
                              else ("sel", "auroc"))]


@beartype
def main_table_values(path: Path) -> dict:
    """Read all six metrics directly from the existing two-feature main table."""
    lines = [line for line in path.read_text().splitlines()
             if line.startswith(("NLDisco &", "CEBRA-Time (", "LangevinFlow ("))]
    if len(lines) != len(MAIN_ORDER):
        raise ValueError("Unexpected Table 1 rows; inspect its selection layout.")
    values = {}
    for line, (method, criterion) in zip(lines, MAIN_ORDER):
        cells = line.removesuffix(r" \\").split(" & ")
        if len(cells) != 13 or not cells[0].startswith(decoder.NAMES[method]):
            raise ValueError(f"Unexpected Table 1 row: {cells[0]}")
        for i, feature in enumerate(MAIN):
            scores = [re.sub(r"\\textbf\{([^{}]+)\}", r"\1", cell)
                      for cell in cells[1 + 6*i:7 + 6*i]]
            if not all(0 <= float(value) <= 1 for value in scores):
                raise ValueError("A displayed metric is outside [0, 1].")
            values[feature, method, criterion] = scores
    return values


@beartype
def render_table(figures: Path = FIGURES, decode_output: Path = DECODE) -> list[dict]:
    """Combine association and decoding scores, with maxima at display precision."""
    selections = json.loads((SOURCE / "selection.json").read_text())["selections"]
    probes = json.loads((decode_output / "results.json").read_text())["results"]
    probes = {(r["feature"], r["method"], r["latent_id"]): r for r in probes}
    main = main_table_values(figures / "churchland_comparison_main.tex")
    records = []
    for row in selected_rows(selections):
        feature, method = row["feature"], row["method"]
        single = probes[feature, method, row["latent_id"]]
        full = probes[feature, method, None]
        measured = [row["tpr"], 1-row["fpr"], row["sel"], row["auroc"],
                    single["test"]["balanced_accuracy"], full["test"]["balanced_accuracy"]]
        scores = main.get((feature, method, row["criterion"]), [
            f"{measured[0]:.4f}", f"{measured[1]:.4f}", floor_three(measured[2]),
            floor_three(measured[3]), f"{measured[4]:.4f}", f"{measured[5]:.4f}",
        ])
        records.append(dict(selection=row, displayed_scores=scores, measured_scores=measured,
                            score_source="Table 1" if feature in MAIN else "saved results",
                            intervals=[single["interval"], full["interval"]]))

    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Method / selector & Latent & \metric{Cov} & \metric{Spec} & \metric{Sel} & \metric{AUROC} & \shortstack{Single\\Decode} & \shortstack{All\\Decode} \\"]
    for feature in FEATURE_ORDER:
        entries = [r for r in records if r["selection"]["feature"] == feature]
        maxima = np.max([[float(s) for s in r["displayed_scores"]] for r in entries], axis=0)
        lines.extend([r"\midrule", r"\multicolumn{8}{l}{\textit{" + TITLES[feature] + r"}} \\"])
        for record in entries:
            row = record["selection"]
            label = decoder.NAMES[row["method"]]
            if row["criterion"] != "illustrative":
                label += r" (best \metric{" + ("Sel" if row["criterion"] == "sel" else "AUROC") + "})"
            cells = [label, str(row["latent_id"])]
            for i, score in enumerate(record["displayed_scores"]):
                cell = r"\textbf{" + score + "}" if float(score) == maxima[i] else score
                cells.append(cell)
            lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (figures / "churchland_feature_metrics.tex").write_text("\n".join(lines))
    (figures / "churchland_appendix_metrics.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


@beartype
def save_figure(fig: plt.Figure, stem: str, figures: Path) -> None:
    """Write matching PDF and PNG assets."""
    for extension in ("pdf", "png"):
        fig.savefig(figures / f"{stem}.{extension}", dpi=210)
    plt.close(fig)


@beartype
def render_rocs(records: list[dict], figures: Path = FIGURES) -> None:
    """Filter the saved empirical curves without refitting or altering vertices."""
    curves = json.loads((SOURCE / "roc_plot_data.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), layout="constrained")
    provenance = {}
    for ax, feature in zip(axes.flat, FEATURE_ORDER):
        rows = [r["selection"] for r in records if r["selection"]["feature"] == feature]
        seen = set()
        provenance[feature] = []
        for row in rows:
            key = row["method"], row["latent_id"]
            if key in seen:
                continue
            seen.add(key)
            criteria = [r["criterion"] for r in rows if (r["method"], r["latent_id"]) == key]
            curve = next(c for c in curves[feature] if (c["method"], c["latent_id"]) == key)
            criterion = "illustrative" if "illustrative" in criteria else "auroc" if "auroc" in criteria else "sel"
            style = {"illustrative": ":", "sel": "--", "auroc": "-"}[criterion]
            selection = "illustrative" if criterion == "illustrative" else "best " + "/".join(
                "Sel" if c == "sel" else "AUROC" for c in criteria)
            ax.plot(curve["fpr"], curve["tpr"], style, color=COLORS[row["method"]], lw=1.25,
                    label=f'{decoder.NAMES[row["method"]]} {row["latent_id"]} ({selection})')
            ax.scatter(row["fpr"], row["tpr"], marker=MARKERS[criterion],
                       color=COLORS[row["method"]], s=20)
            provenance[feature].append(dict(method=key[0], latent_id=key[1], criteria=criteria,
                                            empirical_auroc=curve["auroc"],
                                            native_marker=[row["fpr"], row["tpr"]]))
        ax.plot([0, 1], [0, 1], ":", color=".7", lw=.7)
        ax.set(title=TITLES[feature], xlabel="False-positive rate", ylabel="True-positive rate",
               xlim=(-.01, 1.01), ylim=(-.01, 1.01))
        ax.legend(fontsize=6, loc="lower right", frameon=False)
    save_figure(fig, "churchland_feature_rocs", figures)
    (figures / "churchland_appendix_rocs.json").write_text(json.dumps(provenance, indent=2) + "\n")


@beartype
def render_supplementary_features(records: list[dict], figures: Path = FIGURES) -> None:
    """Retain movement-aligned curves and show only the selected feature latents."""
    features = revision.features()
    spec = json.loads((SOURCE / "frozen_spec.json").read_text())
    latents = np.load(spec["representations"]["nldisco"]["path"], mmap_mode="r")
    with np.load(decoder.SWEEP / "shared/index.npz") as data:
        anchors = data["anchor_index"]
    with np.load(decoder.DATA / "metadata.npz") as data:
        meta = {key: data[key] for key in ("timestamps", "vel_x", "target_x", "target_y", "trial_id")}
    with (decoder.DATA / "trials.csv").open() as stream:
        trials = {int(row["trial_id"]): row for row in csv.DictReader(stream)}
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 6.3))
    fig.subplots_adjust(left=.075, right=.975, top=.90, bottom=.24, wspace=.55, hspace=1.25)
    for row_axes, name in zip(axes, (name for name in FEATURE_ORDER if name not in MAIN)):
        ax, bx, cx = row_axes
        feature = features[name]
        source = anchors[feature.rows]
        trial_ids = meta["trial_id"][source]
        if name == "early_leftward":
            alignment = np.array([float(trials[int(t)]["movement_onset"]) for t in trial_ids])
            category = meta["vel_x"][source] < 0
            edges = np.arange(-.1, .401, .025)
            xlabel, labels = "Time from movement onset (s)", ("Leftward", "Non-leftward")
        else:
            alignment = np.array([float(trials[int(t)]["movement_end"]) for t in trial_ids])
            category = (meta["target_x"][source] == 125) & (meta["target_y"][source] == -18)
            edges = np.arange(-.6, .651, .05)
            xlabel, labels = "Time from movement end (s)", ("Target (125, −18) mm", "Other targets")
        x = meta["timestamps"][source] - alignment
        selected = [r for r in records if r["selection"]["feature"] == name]
        for record, color in zip(selected[:2], (COLORS["nldisco"], "#8ab6d6")):
            row = record["selection"]
            criterion, latent = row["criterion"], row["latent_id"]
            active = latents[feature.rows, latent] > 0
            criterion_label = "Sel" if criterion == "sel" else "AUROC"
            for mask, style, label in ((category, "-", labels[0]), (~category, "--", labels[1])):
                ax.plot((edges[:-1]+edges[1:])/2, probability_curve(x[mask], active[mask], edges),
                        style, color=color, lw=1.2, label=f"{latent} ({criterion_label}): {label}")
            offset = -.16 if criterion == "sel" else .16
            bx.bar(np.arange(2)+offset, [active[feature.labels].mean(), active[~feature.labels].mean()],
                   width=.3, color=color, label=f"Latent {latent}: best {criterion_label}")
        ax.set(xlabel=xlabel, ylabel="P(latent active)", ylim=(-.02, 1.03))
        bx.set(ylim=(0, 1.03), ylabel="P(latent active)")
        bx.set_xticks([0, 1], ["Feature present", "Feature absent"], fontsize=6)
        ax.legend(fontsize=5.2, frameon=False, loc="upper left",
                  bbox_to_anchor=(0, -.48), borderaxespad=0)
        bx.legend(fontsize=5.5, frameon=False, loc="upper left",
                  bbox_to_anchor=(0, -.48), borderaxespad=0)
        ax.text(0, 1.06, TITLES[name], transform=ax.transAxes, fontsize=7.5, fontweight="bold")
        for record in selected:
            row = record["selection"]
            criterion, color = row["criterion"], COLORS[row["method"]]
            cx.scatter(float(record["displayed_scores"][3]), float(record["displayed_scores"][2]),
                       marker=MARKERS[criterion], s=65 if criterion == "auroc" else 38,
                       facecolors="none" if criterion == "auroc" else color,
                       edgecolors=color, linewidths=1, zorder=6 if criterion == "auroc" else 5,
                       clip_on=False)
        cx.set(xlim=(.5, 1), ylim=(.7, 1), xlabel="AUROC", ylabel="Sel")
        cx.set_xticks([.5, .75, 1], ["0.50", "0.75", "1.00"])
        cx.set_yticks([.7, .85, 1], ["0.70", "0.85", "1.00"])
        cx.grid(alpha=.15)
    for letter, ax, title in zip("abc", axes[0], ("Interpretation", "Activation probability", "Evaluation metrics")):
        bounds = ax.get_position()
        fig.text((bounds.x0+bounds.x1)/2, .975, title, va="top", ha="center", fontsize=8)
        fig.text(bounds.x0-.025, .975, letter, va="top", ha="left", fontsize=10, fontweight="bold")
    handles = [Line2D([], [], marker="o", color=c, ls="", markersize=4, label=decoder.NAMES[m])
               for m, c in COLORS.items()]
    handles += [Line2D([], [], marker=MARKERS[c], color=".25", ls="", markersize=4,
                       markerfacecolor="none" if c == "auroc" else ".25", label="Best " + label)
                for c, label in (("sel", "Sel"), ("auroc", "AUROC"))]
    fig.legend(handles=handles, ncol=5, loc="lower center", fontsize=6, frameon=False,
               handlelength=.8, handletextpad=.4, columnspacing=1.5)
    save_figure(fig, "churchland_additional_features", figures)


@beartype
def main() -> None:
    """Regenerate the table and two feature-comparison figures."""
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    records = render_table()
    render_rocs(records)
    render_supplementary_features(records)
    print(f"Rendered {len(records)} selected feature rows and Churchland appendix figures.")


if __name__ == "__main__":
    main()
