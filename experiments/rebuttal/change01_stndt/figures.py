"""Paper assets that add ST-NDT to the Section 4.2 comparison.

Existing results are never regenerated. Figure 4 is the printed PDF with ST-NDT markers
and legend entries overlaid as vector graphics; the ROC and training-time figures redraw
the saved curves and timings unchanged and add ST-NDT; table rows are produced for
insertion by hand. Run with the ST-NDT overlay interpreter (needs PyMuPDF):

    $STNDT_PY -m experiments.rebuttal.change01_stndt.figures
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype

from experiments.churchland.sweeps.s10_endpoint_v1.matched192_tables import floor_three
from experiments.churchland.sweeps.s10_endpoint_v1.publication_style import COLORS, TITLES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PAPER = ROOT / "paper/iclr_paper/figures"
DESTINATION = PAPER / "rebuttal/change01_stndt"
FROZEN = ROOT / "experiments/churchland/outputs/s10_endpoint_revision_v4_nice_cutoffs"
TIMING = ROOT / "experiments/churchland/outputs/training_time_h100_3649999"
FEATURE_ORDER = ("recent_braking", "fast_target_specific", "early_leftward",
                 "late_target_specific")
MAIN = FEATURE_ORDER[:2]
NAMES = dict(nldisco="NLDisco", cebra="CEBRA-Time", langevinflow="LangevinFlow",
             stndt="ST-NDT")
COLOR = "#9467bd"
MARKERS = {"illustrative": "o", "sel": "^", "auroc": "D"}
SELECTOR = dict(sel="Sel", auroc="AUROC")
RC = {"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
      "pdf.fonttype": 42, "ps.fonttype": 42}


@beartype
def displayed(results: dict) -> dict:
    """Format measured ST-NDT scores exactly as the two existing tables print them."""
    probes = {(p["feature"], p["latent_id"]): p["test"]["balanced_accuracy"]
              for p in results["decoding"]}
    rows = {}
    for row in results["selections"]:
        measured = [row["tpr"], 1 - row["fpr"], row["sel"], row["auroc"],
                    probes[row["feature"], row["latent_id"]], probes[row["feature"], None]]
        rows[row["feature"], row["criterion"]] = dict(
            latent_id=row["latent_id"], measured=measured,
            main=[f"{measured[0]:.3f}", f"{measured[1]:.3f}", floor_three(measured[2]),
                  floor_three(measured[3]), f"{measured[4]:.3f}", f"{measured[5]:.3f}"],
            appendix=[f"{measured[0]:.4f}", f"{measured[1]:.4f}", floor_three(measured[2]),
                      floor_three(measured[3]), f"{measured[4]:.4f}", f"{measured[5]:.4f}"])
    return rows


@beartype
def rebold(lines: list, columns: slice) -> list:
    """Bold the column maxima (ties included) among consecutive table rows."""
    cells = [line.removesuffix(r" \\").split(" & ") for line in lines]
    plain = [[re.sub(r"\\textbf\{([^{}]+)\}", r"\1", c) for c in row[columns]] for row in cells]
    maxima = np.max(np.array(plain, dtype=float), axis=0)
    return [" & ".join([*row[:columns.start],
                        *(r"\textbf{" + v + "}" if float(v) == m else v
                          for v, m in zip(values, maxima))]) + r" \\"
            for row, values in zip(cells, plain)]


@beartype
def tables(rows: dict) -> dict:
    """Return both tables with ST-NDT rows added and every existing number untouched."""
    label = {c: r"ST-NDT (best \metric{" + SELECTOR[c] + "})" for c in SELECTOR}
    main = (PAPER / "churchland_comparison_main.tex").read_text().splitlines()
    body = [i for i, line in enumerate(main)
            if line.startswith(("NLDisco &", "CEBRA-Time (", "LangevinFlow (", "ST-NDT ("))]
    kept = [main[i] for i in body if not main[i].startswith("ST-NDT (")]
    added = [" & ".join([label[c], *rows[MAIN[0], c]["main"], *rows[MAIN[1], c]["main"]])
             + r" \\" for c in SELECTOR]
    main[body[0]:body[-1] + 1] = rebold([*kept, *added], slice(1, 13))
    appendix = (PAPER / "churchland_feature_metrics.tex").read_text().splitlines()
    result, block = [], []

    def flush(feature: str | None) -> None:
        if block:
            new = [" & ".join([label[c], str(rows[feature, c]["latent_id"]),
                               *rows[feature, c]["appendix"]]) + r" \\" for c in SELECTOR]
            result.extend(rebold([*block, *new], slice(2, 8)))
            block.clear()

    feature = None
    titles = {TITLES[name]: name for name in FEATURE_ORDER}
    for line in appendix:
        heading = re.match(r"\\multicolumn\{8\}\{l\}\{\\textit\{(.+)\}\} \\\\", line)
        if line.startswith(("NLDisco", "CEBRA-Time", "LangevinFlow")):
            block.append(line)
            continue
        if line.startswith("ST-NDT"):
            continue
        flush(feature)
        if heading:
            feature = titles[heading[1]]
        result.append(line)
    return dict(main="\n".join(main) + "\n", appendix="\n".join(result) + "\n")


@beartype
def figure4(rows: dict, destination: Path) -> None:
    """Overlay ST-NDT markers and legend entries on the printed Figure 4, as vectors."""
    import fitz

    document = fitz.open(PAPER / "figure4.pdf")
    page = document[0]
    width, height = page.rect.width, page.rect.height
    panels = sorted((d["rect"] for d in page.get_drawings()
                     if d["type"] == "f" and d.get("fill") == (1., 1., 1.)
                     and abs(d["rect"].width - 105.22) < .1
                     and abs(d["rect"].height - 60.43) < .1),
                    key=lambda r: (round(r.y0), r.x0))
    if len(panels) != 4:
        raise ValueError("Expected the four metric panels of the printed Figure 4")
    limits = (((.5, 1.), (0., 1.)), ((.7, 1.), (.7, 1.)))  # (x, y) limits of b and c
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(width / 72, height / 72))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, width), ylim=(height, 0))
        ax.axis("off")
        for row, feature in enumerate(MAIN):
            for criterion in SELECTOR:
                cov, spec, sel, auroc = map(float, rows[feature, criterion]["main"][:4])
                for (x, y), rect, ((x0, x1), (y0, y1)) in zip(
                        ((spec, cov), (auroc, sel)), panels[2 * row:2 * row + 2], limits):
                    ax.scatter(rect.x0 + (x - x0) / (x1 - x0) * rect.width,
                               rect.y1 - (y - y0) / (y1 - y0) * rect.height,
                               marker=MARKERS[criterion], s=36 if criterion == "auroc" else 25,
                               facecolors="none" if criterion == "auroc" else COLOR,
                               edgecolors=COLOR, linewidths=.85, clip_on=False, zorder=5)
        # Legend continues the printed column-major order: third column, then a fourth.
        for criterion, x, baseline in (("sel", 336.35, 254.048), ("auroc", 435.6, 245.235)):
            ax.plot(x - 5.1, baseline - 2.1, MARKERS[criterion], color=COLOR, markeredgewidth=1,
                    markersize=4.8 if criterion == "auroc" else 4,
                    markerfacecolor="none" if criterion == "auroc" else COLOR, clip_on=False)
            ax.text(x, baseline, f"ST-NDT (best {SELECTOR[criterion]})", fontsize=6,
                    va="baseline", ha="left")
        with tempfile.TemporaryDirectory() as scratch:
            fig.savefig(Path(scratch) / "overlay.pdf", transparent=True)
            plt.close(fig)
            overlay = fitz.open(Path(scratch) / "overlay.pdf")
            if abs(overlay[0].rect.width - width) > .01 or abs(overlay[0].rect.height - height) > .01:
                raise ValueError("Overlay and printed figure sizes differ")
            page.show_pdf_page(page.rect, overlay, 0)
            document.save(destination / "figure4.pdf", garbage=3, deflate=True)
    fitz.open(destination / "figure4.pdf")[0].get_pixmap(dpi=210).save(
        destination / "figure4.png")


@beartype
def rocs(results: dict, destination: Path) -> None:
    """Redraw the saved empirical ROC curves unchanged and add the ST-NDT curves."""
    selections = json.loads((FROZEN / "selection.json").read_text())["selections"]
    index = {(r["feature"], r["method"], r["criterion"]): r for r in selections}
    curves = json.loads((FROZEN / "roc_plot_data.json").read_text())
    added = json.loads((HERE / "outputs/roc_plot_data.json").read_text())
    colors = dict(COLORS, stndt=COLOR)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), layout="constrained")
        for ax, feature in zip(axes.flat, FEATURE_ORDER):
            rows = [index[feature, method, criterion] for method in COLORS
                    for criterion in (("illustrative",) if method == "nldisco"
                                      and feature in MAIN else ("sel", "auroc"))]
            rows += [r for r in results["selections"] if r["feature"] == feature]
            seen = set()
            for row in rows:
                key = row["method"], row["latent_id"]
                if key in seen:
                    continue
                seen.add(key)
                criteria = [r["criterion"] for r in rows
                            if (r["method"], r["latent_id"]) == key]
                curve = next(c for c in [*curves[feature], *added[feature]]
                             if (c["method"], c["latent_id"]) == key)
                criterion = ("illustrative" if "illustrative" in criteria
                             else "auroc" if "auroc" in criteria else "sel")
                style = {"illustrative": ":", "sel": "--", "auroc": "-"}[criterion]
                selection = "illustrative" if criterion == "illustrative" else "best " + "/".join(
                    SELECTOR[c] for c in criteria)
                ax.plot(curve["fpr"], curve["tpr"], style, color=colors[row["method"]], lw=1.25,
                        label=f'{NAMES[row["method"]]} {row["latent_id"]} ({selection})')
                ax.scatter(row["fpr"], row["tpr"], marker=MARKERS[criterion],
                           color=colors[row["method"]], s=20)
            ax.plot([0, 1], [0, 1], ":", color=".7", lw=.7)
            ax.set(title=TITLES[feature], xlabel="False-positive rate",
                   ylabel="True-positive rate", xlim=(-.01, 1.01), ylim=(-.01, 1.01))
            ax.legend(fontsize=6, loc="lower right", frameon=False)
        for extension in ("pdf", "png"):
            fig.savefig(destination / f"churchland_feature_rocs.{extension}", dpi=210)
        plt.close(fig)


@beartype
def training_times(new: Path, destination: Path) -> dict:
    """Plot the fifteen saved fits unchanged beside the five ST-NDT fits."""
    rows = json.loads((TIMING / "results.json").read_text())["runs"]
    added = json.loads((new / "results.json").read_text())["runs"]
    if len(added) != 5 or any(r["foreign_gpu_pids"] or r["status"] != "complete"
                              or "H100" not in r["gpu_name"] for r in added):
        raise ValueError("Require five uncontended H100 ST-NDT fits")
    rows = [*rows, *added]
    methods = ("nldisco", "cebra", "langevinflow", "stndt")
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", COLOR)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(6.2, 3.5), layout="constrained")
        summaries, top = {}, max(r["train_seconds"] for r in rows)
        for position, (method, color) in enumerate(zip(methods, colors), 1):
            group = sorted((r for r in rows if r["method"] == method), key=lambda r: r["seed"])
            values = np.array([r["train_seconds"] for r in group])
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            summaries[method] = dict(median=float(median), q1=float(q1), q3=float(q3),
                                     min=float(values.min()), max=float(values.max()),
                                     parameters=group[0]["parameter_count"])
            ax.boxplot([values], positions=[position], widths=.46, patch_artist=True,
                       showfliers=False,
                       boxprops=dict(facecolor=color, alpha=.2, edgecolor=color),
                       medianprops=dict(color=color, linewidth=2),
                       whiskerprops=dict(color=color), capprops=dict(color=color))
            ax.scatter(position + np.linspace(-.12, .12, 5), values, c=color, s=34,
                       edgecolors="white", linewidths=.6, zorder=3)
            ax.text(position, values.max() + top * .055, f"{median:.1f} s", ha="center",
                    fontweight="bold", color=color)
        ax.set(xticks=range(1, 5), xticklabels=[NAMES[m] for m in methods],
               ylabel="Training time (s)", title="LVM training time comparison",
               ylim=(0, 1.15 * top))
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=.18)
        ax.set_axisbelow(True)
        fig.savefig(destination / "training_time_h100.pdf", bbox_inches="tight")
        fig.savefig(destination / "training_time_h100.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
    return summaries


@beartype
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", type=Path, help="ST-NDT timing output directory")
    args = parser.parse_args()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    results = json.loads((HERE / "outputs/results.json").read_text())
    rows = displayed(results)
    text = tables(rows)
    record = dict(rows={f"{feature}:{criterion}": row for (feature, criterion), row in rows.items()},
                  tables=text)
    figure4(rows, DESTINATION)
    rocs(results, DESTINATION)
    if args.timing:
        record["training_time"] = training_times(args.timing, DESTINATION)
    (HERE / "outputs/paper_assets.json").write_text(json.dumps(record, indent=2) + "\n")
    print(text["main"], text["appendix"], sep="\n")


if __name__ == "__main__":
    main()
