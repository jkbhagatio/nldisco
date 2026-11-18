"""Draw the multi-session SED schematic for the appendix."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

FIGURES = Path(__file__).resolve().parents[3] / "paper/iclr_paper/figures/rebuttal/change02_multisession"
FROZEN, TRAINED, SHARED, INK = "#1f77b4", "#ff7f0e", "#d9e6f2", "#222222"


def box(ax, xy, w, h, text, face, edge, fontsize=7.5, weight="normal", ls="-"):
    """Rounded labelled box."""
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=face, ec=edge, lw=1.1, ls=ls, zorder=2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=INK, fontweight=weight, zorder=3, linespacing=1.15)


def arrow(ax, start, end, color=INK, ls="-", lw=1.0, rad=0.):
    """Arrow between two points."""
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, lw=lw, color=color,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=1,
                                 shrinkA=1, shrinkB=1))


def raster(ax, x, y, w, h, n, seed, color):
    """Small spike-count window icon: n unit rows by time."""
    rng = np.random.default_rng(seed)
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=".4", lw=.6, zorder=2))
    for row in range(n):
        times = rng.random(rng.integers(2, 6)) * w * .94 + x + w * .03
        yy = y + h * (row + .5) / n
        ax.vlines(times, yy - h / n * .32, yy + h / n * .32, color=color, lw=.7, zorder=3)


def probe(ax, x, y, h, depths_a, depths_b):
    """Probe shank with the units of both sessions at their depths, matched by position."""
    ax.add_patch(Rectangle((x - .035, y), .07, h, fc=".88", ec=".5", lw=.6, zorder=1))
    ax.add_patch(plt.Polygon([[x - .035, y], [x + .035, y], [x, y - .12]], fc=".88", ec=".5", lw=.6))
    for depth_a, depth_b in zip(depths_a, depths_b):
        ax.plot([x - .2, x + .2], [y + h * depth_a, y + h * depth_b], color=".55", lw=.6, ls=":", zorder=2)
    ax.scatter(np.full(len(depths_a), x - .2), y + h * np.asarray(depths_a), s=13, color=FROZEN, zorder=3)
    ax.scatter(np.full(len(depths_b), x + .2), y + h * np.asarray(depths_b), s=13, color=TRAINED, zorder=3)
    ax.text(x - .2, y + h + .09, "A", ha="center", fontsize=7, color=FROZEN, fontweight="bold")
    ax.text(x + .2, y + h + .09, "B", ha="center", fontsize=7, color=TRAINED, fontweight="bold")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.2, 2.75))
    ax.set(xlim=(0, 10), ylim=(0, 3.9))
    ax.axis("off")
    top, bottom = 2.65, .55
    # Inputs.
    raster(ax, .55, top, .8, .75, 7, 0, FROZEN)
    raster(ax, .55, bottom, .8, .75, 6, 1, TRAINED)
    ax.text(.95, top + .86, "Session A\n$N_A$ units", ha="center", va="bottom", fontsize=7.5, color=FROZEN)
    ax.text(.95, bottom - .1, "Session B\n$N_B$ units", ha="center", va="top", fontsize=7.5, color=TRAINED)
    # Input mappings.
    box(ax, (1.85, top + .05), 1.25, .65, "Input mapping\n$E_A$", "#e8f0f8", FROZEN)
    box(ax, (1.85, bottom + .05), 1.25, .65, "Input mapping\n$E_B$", "#fdebd8", TRAINED)
    arrow(ax, (1.37, top + .375), (1.83, top + .375))
    arrow(ax, (1.37, bottom + .375), (1.83, bottom + .375))
    # Probe-position initialisation.
    probe(ax, 2.47, 1.72, .62, [.1, .38, .62, .9], [.13, .36, .66, .88])
    ax.text(2.95, 2.0, "initialise from\nprobe position", fontsize=6.5, color=".3", va="center")
    # Shared core.
    box(ax, (3.75, 1.3), 1.45, 1.3, "Shared\ntemporal core\n(Transformer)", SHARED, FROZEN, weight="bold")
    arrow(ax, (3.12, top + .375), (3.75, 2.35), rad=-.15)
    arrow(ax, (3.12, bottom + .375), (3.75, 1.55), rad=.15)
    # Latents.
    lx, ly, cell = 5.75, 1.42, .135
    for i in range(8):
        ax.add_patch(Rectangle((lx, ly + (i + 2.4) * cell), .22, cell * .9, fc=FROZEN if i in (2, 5) else "white",
                               ec=FROZEN, lw=.7, zorder=3))
    for i in range(2):
        ax.add_patch(Rectangle((lx, ly + i * cell), .22, cell * .9, fc=TRAINED if i == 0 else "white",
                               ec=TRAINED, lw=.7, zorder=3))
    ax.text(lx + .11, ly + 10.9 * cell, "Shared latents\n$z_1,\\dots,z_D$", ha="center", va="bottom",
            fontsize=7.5, color=FROZEN)
    ax.text(lx + .11, ly - .08, "B-only\nlatents", ha="center", va="top", fontsize=7.5, color=TRAINED)
    arrow(ax, (5.22, 1.95), (5.72, 1.95))
    # Decoders.
    box(ax, (6.75, top + .05), 1.25, .65, "Decoder\n$W_A$", "#e8f0f8", FROZEN)
    box(ax, (6.75, bottom + .05), 1.25, .65, "Decoder\n$W_B$", "#fdebd8", TRAINED)
    arrow(ax, (6.0, 2.25), (6.75, top + .3), rad=-.15)
    arrow(ax, (6.0, 1.65), (6.75, bottom + .45), rad=.15)
    raster(ax, 8.5, top, .8, .75, 7, 0, FROZEN)
    raster(ax, 8.5, bottom, .8, .75, 6, 1, TRAINED)
    arrow(ax, (8.02, top + .375), (8.48, top + .375))
    arrow(ax, (8.02, bottom + .375), (8.48, bottom + .375))
    ax.text(8.9, top + .86, "Reconstruction\nof session A", ha="center", va="bottom", fontsize=7.5, color=FROZEN)
    ax.text(8.9, bottom - .1, "Reconstruction\nof session B", ha="center", va="top", fontsize=7.5, color=TRAINED)
    # Legend.
    ax.add_patch(Rectangle((2.2, .12), .16, .13, fc="#e8f0f8", ec=FROZEN, lw=.9))
    ax.text(2.43, .185, "trained on session A, then frozen", fontsize=7, va="center")
    ax.add_patch(Rectangle((5.05, .12), .16, .13, fc="#fdebd8", ec=TRAINED, lw=.9))
    ax.text(5.28, .185, "trained on session B's neural activity", fontsize=7, va="center")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for extension in ("pdf", "png"):
        fig.savefig(FIGURES / f"multisession_schematic.{extension}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
