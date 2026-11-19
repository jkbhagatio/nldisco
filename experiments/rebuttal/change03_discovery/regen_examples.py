"""Re-draws change 3's example figure with a layout that fits the paper.

Same data, selection rule and scores as `paper.examples_figure` on branch
rebuttal/c03-discovery (one validated description per method, the one with the
highest held-out AUROC); only titles and annotation placement differ.

    PYTHONPATH=<change-3 worktree> python regen_examples.py DATA RESULTS OUT.pdf
"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.rebuttal.change03_discovery import data, rules  # noqa: E402
from experiments.rebuttal.change03_discovery.part_a import LABELS, METHODS  # noqa: E402

BLUE, GREY = "#2166ac", "#9a9a9a"


def main(data_root: Path, results: Path, out: Path) -> None:
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    answers = pd.read_csv(results / "part_b/part_b_answers.csv")
    key = json.loads((results / "part_b/key/part_b_key.json").read_text())
    session = data.load_session(data_root)
    chosen = [answers[(answers.method == m) & answers.passed].sort_values(
        "auroc", ascending=False).head(1) for m in METHODS]
    chosen = [c.iloc[0] for c in chosen if len(c)]
    fig, grid = plt.subplots(2, 3, figsize=(5.5, 4.1), constrained_layout=True)
    axes = [grid.ravel()[:len(chosen)]]
    for spare in grid.ravel()[len(chosen):]:
        spare.axis("off")
    records = []
    for letter, ax, row in zip("abcdefgh", axes[0], chosen):
        values = np.load(key[row.card]["path"], mmap_mode="r")
        mask, valid = rules.evaluate(row.rule, session.variables, session.support)
        held = session.heldout & valid
        sign = 1 if row.direction == "high" else -1
        column = sign * np.asarray(values[np.flatnonzero(held), int(row.latent)], dtype=float)
        inside = mask[held]
        lo, hi = np.quantile(column, [0.001, 0.999])
        bins = np.linspace(lo, hi, 60) if hi > lo else 10
        for flag, color, label in ((~inside, GREY, "rule does not hold"), (inside, BLUE, "rule holds")):
            weights = np.full(int(flag.sum()), 1 / max(int(flag.sum()), 1))
            ax.hist(np.clip(column[flag], lo, hi), bins=bins, weights=weights, color=color,
                    alpha=.65, label=label)
        ax.axvline(np.clip(row.threshold, lo, hi), color="k", ls="--", lw=.8)
        ax.set_yscale("log")
        ax.set_xlabel("Latent value" + (" (sign-reversed)" if sign < 0 else ""), fontsize=7.5)
        ax.tick_params(labelsize=7)
        rule = textwrap.wrap(row.rule, 28, break_long_words=False)
        rule = "\n".join(rule + [" "] * (2 - len(rule)))  # equal title heights across panels
        scores = (f"Cov {row['cov']:.2f}, Spec {row.spec:.2f}\n"
                  f"Sel {row.sel:.2f}, AUROC {row.auroc:.2f}")
        ax.set_title(f"{LABELS[row.method]}, latent {int(row.latent)}\n{scores}\n{rule}",
                     fontsize=7, linespacing=1.25)
        ax.text(-.2, 1.04, letter, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, fontweight="bold")
        records.append(dict(method=row.method, latent=int(row.latent), rule=row.rule,
                            direction=row.direction, rater=int(row.rater), cov=float(row["cov"]),
                            spec=float(row.spec), sel=float(row.sel), auroc=float(row.auroc),
                            threshold=float(row.threshold), n_inside=int(inside.sum()),
                            n_outside=int((~inside).sum())))
    for ax in grid[:, 0]:
        ax.set_ylabel("Fraction of held-out windows", fontsize=7.5)
    grid.ravel()[-1].legend(*axes[0][0].get_legend_handles_labels(), frameon=False, fontsize=7.5,
                            loc="center")
    fig.savefig(out, bbox_inches="tight")
    print(json.dumps(records, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
