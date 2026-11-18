"""Throwaway latent dashboards from saved activations and behavior (rebuttal change 7).

One view per latent: activation as a function of chosen behavioral or environmental
variables, 2-D activation maps, and activating episodes alongside behavior (the
strongest ones, and ones sampled across the activation range). `write_dashboard` renders every active latent and an `index.html` that
steps between them with the arrow keys; `Dashboard.figure` gives one static view.
"""

import html
import json
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype
from einops import rearrange
from jaxtyping import Bool, Float, Int, jaxtyped
from scipy import sparse

BLUE, GREY, BAD = "#1f77b4", "#7f7f7f", "#eeeeee"
STYLE = {"font.size": 6.5, "axes.spines.top": False, "axes.spines.right": False,
         "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
         "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42, "ps.fonttype": 42}


@dataclass(frozen=True)
class Variable:
    """One behavioral or environmental variable, with one value per example (NaN: invalid)."""

    label: str
    values: np.ndarray
    edges: np.ndarray
    log: bool = False
    mark: Optional[float] = None  # optional reference line


@dataclass(frozen=True)
class Map:
    """Two variables binned jointly, or (with `points`) a categorical set of locations."""

    label: str
    x: np.ndarray
    y: np.ndarray
    xedges: np.ndarray
    yedges: np.ndarray
    xlabel: str = ""
    ylabel: str = ""
    points: Optional[np.ndarray] = None  # [m, 2] locations; x then holds codes in [0, m)
    marker: Optional[tuple] = None  # (x, y, label) landmark
    invert_y: bool = False


@dataclass(frozen=True)
class Dataset:
    """Everything a dashboard needs: activations plus aligned behavior, in recorded order."""

    title: str
    activations: np.ndarray  # [n, d] nonnegative sparse activations
    bin_seconds: float
    segments: np.ndarray  # [n] ids of temporally contiguous runs (trials or gap-free stretches)
    variables: tuple
    maps: tuple
    timeline: Variable  # drawn wide: recording time or task-aligned time
    traces: dict  # label -> [n] behavior shown under each episode
    events: dict = field(default_factory=dict)  # label -> [n] bool markers within episodes
    episode_labels: Optional[np.ndarray] = None  # [n] strings used as episode titles
    half_width: int = 250  # bins shown either side of an episode's peak
    n_episodes: int = 6
    n_intervals: int = 6  # equal-width activation intervals sampled once each; 0 disables
    seed: int = 0  # for the sampled episodes
    min_count: int = 50  # examples a bin needs before its statistic is drawn


@jaxtyped(typechecker=beartype)
def digitize(values: Float[np.ndarray, " n"], edges: Float[np.ndarray, " e"]) -> Int[np.ndarray, " n"]:
    """Bin index per example; -1 outside the edges or non-finite. The last bin is closed."""
    index = np.searchsorted(edges, values, side="right") - 1
    index[values == edges[-1]] = len(edges) - 2
    index[~np.isfinite(values) | (index < 0) | (index >= len(edges) - 1)] = -1
    return index.astype(np.int64)


@jaxtyped(typechecker=beartype)
def binned_activity(
    activations: Float[np.ndarray, "n d"], bins: Int[np.ndarray, " n"], n_bins: int,
    active: Optional[Float[np.ndarray, "n d"]] = None,
) -> tuple[Float[np.ndarray, " b"], Float[np.ndarray, "b d"], Float[np.ndarray, "b d"]]:
    """Per bin and latent: example count, P(activation > 0) and mean activation.

    One sparse indicator product covers every latent at once; empty bins give NaN.
    `active` optionally passes a cached `(activations > 0)` in the activations' dtype.
    """
    rows = np.flatnonzero(bins >= 0)
    onehot = sparse.csr_matrix((np.ones(len(rows), activations.dtype), (bins[rows], rows)),
                               shape=(n_bins, len(bins)))
    counts = np.bincount(bins[rows], minlength=n_bins).astype(np.float64)
    active = (activations > 0).astype(activations.dtype) if active is None else active
    with np.errstate(invalid="ignore", divide="ignore"):
        return counts, (onehot @ active) / counts[:, None], (onehot @ activations) / counts[:, None]


@jaxtyped(typechecker=beartype)
def top_episodes(
    activation: Float[np.ndarray, " n"], k: int, half_width: int,
) -> Int[np.ndarray, " m"]:
    """Indices of the k strongest positive peaks whose +/- half_width windows are disjoint."""
    remaining, peaks = activation.astype(np.float64), []
    for _ in range(k):
        peak = int(remaining.argmax())
        if not remaining[peak] > 0:
            break
        peaks.append(peak)
        remaining[max(peak - 2 * half_width, 0):peak + 2 * half_width + 1] = -np.inf
    return np.array(peaks, dtype=np.int64)


@jaxtyped(typechecker=beartype)
def interval_samples(
    activation: Float[np.ndarray, " n"], n_intervals: int, rng: np.random.Generator,
) -> tuple[Int[np.ndarray, " m"], Float[np.ndarray, "m 2"]]:
    """One uniformly drawn example per equal-width interval of (0, max], highest first.

    Returns the sampled indices and each one's (low, high] interval; empty intervals
    are skipped. Intervals follow the activation value rather than its quantiles, so
    the columns span the latent's whole activation range.
    """
    edges = np.linspace(0, max(float(activation.max()), 0.), n_intervals + 1)
    order = np.argsort(activation, kind="stable")
    cuts = np.searchsorted(activation[order], edges, side="right")
    picks = [(order[rng.integers(cuts[j], cuts[j + 1])], edges[j], edges[j + 1])
             for j in reversed(range(n_intervals)) if cuts[j + 1] > cuts[j]]
    return (np.array([p[0] for p in picks], dtype=np.int64),
            np.array([p[1:] for p in picks], dtype=np.float64).reshape(-1, 2))


@jaxtyped(typechecker=beartype)
def episode_bounds(
    peaks: Int[np.ndarray, " m"], segments: Int[np.ndarray, " n"], half_width: int,
) -> Int[np.ndarray, "m 2"]:
    """Half-open [start, stop) rows around each peak, clipped to the peak's segment."""
    starts = np.r_[0, np.flatnonzero(np.diff(segments)) + 1, len(segments)]
    run = np.searchsorted(starts, peaks, side="right") - 1
    return np.stack([np.maximum(peaks - half_width, starts[run]),
                     np.minimum(peaks + half_width + 1, starts[run + 1])], axis=1)


class Dashboard:
    """Precompute every latent's statistics once, then draw single-latent views on demand."""

    @beartype
    def __init__(self, data: Dataset):
        self.data, acts = data, data.activations
        self.rate = (acts > 0).mean(0)
        # Display scale: each latent's 99th-percentile positive activation.
        self.scale = np.array([np.quantile(col[col > 0], .99) if (col > 0).any() else 1.
                               for col in acts.T])
        active = (acts > 0).astype(acts.dtype)
        self.tuning = [binned_activity(acts, digitize(v.values.astype(np.float64), v.edges),
                                       len(v.edges) - 1, active)
                       for v in (*data.variables, data.timeline)]
        self.grids = []
        for m in data.maps:
            if m.points is not None:
                self.grids.append(binned_activity(acts, m.x.astype(np.int64), len(m.points), active))
                continue
            ix, iy = digitize(m.x.astype(np.float64), m.xedges), digitize(m.y.astype(np.float64), m.yedges)
            nx = len(m.xedges) - 1
            flat = np.where((ix >= 0) & (iy >= 0), iy * nx + ix, -1)
            self.grids.append(binned_activity(acts, flat, nx * (len(m.yedges) - 1), active))

    @property
    def active_latents(self) -> np.ndarray:
        """Latents with at least one positive activation."""
        return np.flatnonzero(self.rate > 0)

    @beartype
    def figure(self, latent: int, letters: bool = False) -> plt.Figure:
        """Draw one latent's view.

        Rows: activation by variable (a), maps and timeline (b), the strongest episodes (c),
        and episodes sampled across the activation range (d), each above its behavior.
        """
        d, z, scale = self.data, self.data.activations[:, latent], self.scale[latent]
        n_var, n_map, n_trace = len(d.variables), len(d.maps), len(d.traces)
        # Row heights in inches, with spacer rows so label room does not scale with content.
        block = .36 * (1 + n_trace)
        heights = [1., .78, 1.35, .86, block] + ([.84, block] if d.n_intervals else [])
        top, bottom = .52, .46
        total = sum(heights) + top + bottom
        with plt.rc_context(STYLE):
            fig = plt.figure(figsize=(7.2, total))
            outer = fig.add_gridspec(len(heights), 1, height_ratios=heights, left=.085,
                                     right=.985, top=1 - top / total, bottom=bottom / total,
                                     hspace=0)
            fig.text(.085, 1 - .05 / total, f"{d.title}  |  latent {latent}", va="top",
                     fontsize=8.5, fontweight="bold")
            fig.text(.085, 1 - .23 / total, f"active in {100 * self.rate[latent]:.2f}% of "
                     f"{len(z):,} examples  |  99th-percentile activation {scale:.2f}",
                     va="top", fontsize=7)

            # (a) activation as a function of each variable.
            axes = outer[0].subgridspec(1, n_var, wspace=.28).subplots(sharey=True)
            for ax, v, stats in zip(np.atleast_1d(axes), d.variables, self.tuning):
                self._tuning(ax, v, stats, latent)
            axes = np.atleast_1d(axes)
            axes[0].set_ylabel("P(active)  /  mean act.")
            fig.legend(*axes[0].get_legend_handles_labels(), fontsize=6, frameon=False,
                       loc="upper right", bbox_to_anchor=(.99, 1.), handlelength=1.5)

            # (b) joint maps and the wide timeline.
            row = outer[2].subgridspec(1, n_map + 1, wspace=.62,
                                       width_ratios=[1] * n_map + [max(3.4 - n_map, 1.2)])
            map_axes = [fig.add_subplot(row[i]) for i in range(n_map)]
            for ax, m, stats in zip(map_axes, d.maps, self.grids):
                self._map(ax, m, stats, latent)
            wide = fig.add_subplot(row[n_map])
            self._tuning(wide, d.timeline, self.tuning[-1], latent)
            wide.set_ylabel("P(active)  /  mean act.")

            # (c) strongest episodes; (d) one episode per equal-width activation interval.
            peaks = top_episodes(z, d.n_episodes, d.half_width)
            blocks = [(outer[4], peaks, [f"#{i + 1}: peak act. {z[p]:.1f}"
                                         for i, p in enumerate(peaks)],
                       "Strongest activating episodes", "Time from peak (s)", d.n_episodes)]
            if d.n_intervals:
                picks, bounds = interval_samples(z, d.n_intervals,
                                                 np.random.default_rng([d.seed, latent]))
                blocks.append((outer[6], picks,
                               [f"act. {z[p]:.1f} in ({lo:.1f}, {hi:.1f}]"
                                for p, (lo, hi) in zip(picks, bounds)],
                               "Episodes sampled across the activation range, one per "
                               "equal-width interval", "Time from sampled example (s)",
                               d.n_intervals))
            for spec, centers, titles, header, xlabel, n_columns in blocks:
                self._episodes(fig, spec, z, centers, titles, n_columns)
                box = spec.get_position(fig)
                fig.text(.085, box.y1 + .3 / total, header, fontsize=7, fontweight="bold",
                         va="bottom")
                fig.text((box.x0 + box.x1) / 2, box.y0 - .3 / total, xlabel, ha="center",
                         va="top")
            colors = plt.get_cmap("Dark2")
            handles = [plt.Line2D([], [], color=colors(j), lw=.8, ls="--", label=name)
                       for j, name in enumerate(d.events)]
            if handles:
                fig.legend(handles=handles, loc="lower right", ncol=len(handles), fontsize=6,
                           frameon=False, bbox_to_anchor=(.985, -.003))
            if letters:
                lifts = [.06, .06] + [.3] * (len(blocks))
                for letter, spec, lift in zip("abcd", list(outer)[::2], lifts):
                    fig.text(.008, spec.get_position(fig).y1 + lift / total, letter,
                             fontsize=10, fontweight="bold", va="bottom")
        return fig

    def _episodes(self, fig, spec, z: np.ndarray, centers: np.ndarray, titles: list,
                  n_columns: int) -> None:
        """One column per episode: latent activation above each behavior trace."""
        d = self.data
        bounds = episode_bounds(centers, d.segments, d.half_width)
        grid = spec.subgridspec(1 + len(d.traces), n_columns, wspace=.12, hspace=.25)
        axes = grid.subplots(sharex=True, sharey="row", squeeze=False)
        colors = plt.get_cmap("Dark2")
        for col, (center, (lo, hi)) in enumerate(zip(centers, bounds)):
            t = (np.arange(lo, hi) - center) * d.bin_seconds
            axes[0, col].fill_between(t, z[lo:hi], step="mid", color=BLUE, lw=0)
            title = titles[col]
            if d.episode_labels is not None:
                title += f"\n{d.episode_labels[center]}"
            axes[0, col].set_title(title, fontsize=5.6, pad=2, linespacing=1.05)
            for i, values in enumerate(d.traces.values()):
                axes[1 + i, col].plot(t, values[lo:hi], color=GREY, lw=.7)
            for j, flags in enumerate(d.events.values()):
                for when in t[flags[lo:hi]]:
                    for ax in axes[:, col]:
                        ax.axvline(when, color=colors(j), lw=.6, ls="--", zorder=0)
        for col in range(len(centers), n_columns):
            for ax in axes[:, col]:
                ax.set_visible(False)
        axes[0, 0].set_ylabel("Latent\nact.", fontsize=5.8)
        for i, name in enumerate(d.traces):
            axes[1 + i, 0].set_ylabel(name, fontsize=5.8)
        fig.align_ylabels(axes[:, 0])

    def _tuning(self, ax, v: Variable, stats: tuple, latent: int) -> None:
        """P(active) and scaled mean activation per bin; sparsely populated bins are hidden."""
        counts, active, mean = stats
        lo, hi = v.edges[:-1], v.edges[1:]
        centers = np.sqrt(np.where(lo > 0, lo, hi / 2) * hi) if v.log else (lo + hi) / 2
        keep = counts >= self.data.min_count
        ax.plot(centers[keep], active[keep, latent], "-o", color=BLUE, ms=2, lw=.9,
                label="P(active)")
        ax.plot(centers[keep], np.minimum(mean[keep, latent] / self.scale[latent], 1.05), "-",
                color=GREY, lw=.9, label="mean act. / 99th pct.")
        if v.mark is not None:
            ax.axvline(v.mark, color="k", lw=.6, ls=":")
        if v.log:
            ax.set_xscale("log")
        ax.set(ylim=(-.03, 1.05), xlabel=v.label)
        ax.set_yticks([0, .5, 1])
        ax.grid(alpha=.25, lw=.4)

    def _map(self, ax, m: Map, stats: tuple, latent: int) -> None:
        """P(active) over two variables, or at categorical locations."""
        counts, active, _ = stats
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(BAD)
        rate = np.where(counts >= self.data.min_count, active[:, latent], np.nan)
        # Each map has its own color scale, so weakly active latents still show structure.
        vmax = max(np.ceil(10 * np.nanmax(rate, initial=0)) / 10, .1)
        if m.points is not None:
            im = ax.scatter(*m.points.T, c=rate, cmap=cmap, vmin=0, vmax=vmax, s=38,
                            edgecolors="k", linewidths=.4, plotnonfinite=True)
            ax.margins(.18)
        else:
            grid = rearrange(rate, "(y x) -> y x", x=len(m.xedges) - 1)
            im = ax.pcolormesh(m.xedges, m.yedges, np.ma.masked_invalid(grid), cmap=cmap,
                               vmin=0, vmax=vmax, shading="flat", rasterized=True)
            ax.set_facecolor(BAD)
        if m.marker is not None:
            ax.plot(*m.marker[:2], marker="x", color="r", ms=4, mew=.9)
            ax.annotate(m.marker[2], m.marker[:2], xytext=(3, 3), textcoords="offset points",
                        color="r", fontsize=5.5)
        if m.invert_y:
            ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.set(xlabel=m.xlabel, ylabel=m.ylabel)
        ax.set_title(m.label, fontsize=6.5, pad=3)
        bar = ax.figure.colorbar(im, cax=ax.inset_axes([1.06, 0, .06, 1]), ticks=[0, vmax])
        bar.set_label("P(active)", labelpad=-5)
        bar.outline.set_linewidth(.5)


PAGE = """<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>body{{font:14px sans-serif;margin:12px;background:#fafafa}}img{{max-width:100%;
border:1px solid #ccc;background:#fff}}button,input,select{{font:inherit;margin-right:6px}}
#bar{{position:sticky;top:0;background:#fafafa;padding:6px 0}}</style>
<div id="bar"><b>{title}</b> &nbsp;
<button id="prev">&larr; prev</button><button id="next">next &rarr;</button>
latent <input id="jump" type="number" style="width:5em">
order <select id="order"><option value="index">latent index</option>
<option value="rate">activation frequency</option></select>
<span id="info"></span> &nbsp; <small>(arrow keys step between latents)</small></div>
<img id="view">
<script>
const latents = {latents}, rate = {rate};
let order = [...latents], i = 0;
const show = () => {{ const l = order[i];
  document.getElementById('view').src = 'views/latent_' + String(l).padStart(4, '0') + '.png';
  document.getElementById('jump').value = l;
  document.getElementById('info').textContent = (i + 1) + ' / ' + order.length +
    ' active latents; active in ' + (100 * rate[l]).toFixed(2) + '% of examples';
  location.hash = l; }};
const step = d => {{ i = (i + d + order.length) % order.length; show(); }};
const go = l => {{ const j = order.indexOf(l); if (j >= 0) {{ i = j; show(); }} }};
document.getElementById('prev').onclick = () => step(-1);
document.getElementById('next').onclick = () => step(1);
document.getElementById('jump').onchange = e => go(+e.target.value);
document.getElementById('order').onchange = e => {{ const l = order[i];
  order = [...latents].sort((a, b) => e.target.value == 'rate' ? rate[b] - rate[a] : a - b);
  go(l); }};
document.onkeydown = e => {{ if (e.target.tagName == 'INPUT') return;
  if (e.key == 'ArrowRight') step(1); if (e.key == 'ArrowLeft') step(-1); }};
go(+location.hash.slice(1)); show();
</script>"""


_SHARED: dict = {}  # board inherited by forked render workers


def _render(job: tuple) -> None:
    latent, path, dpi = job
    fig = _SHARED["board"].figure(latent)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


@beartype
def write_dashboard(board: Dashboard, out: Path, latents: Optional[list] = None,
                    dpi: int = 130, workers: int = 1) -> Path:
    """Render one PNG per active latent and the `index.html` that steps through them."""
    views = out / "views"
    views.mkdir(parents=True, exist_ok=True)
    chosen = [int(i) for i in (board.active_latents if latents is None else latents)]
    jobs = [(latent, views / f"latent_{latent:04d}.png", dpi) for latent in chosen]
    _SHARED["board"] = board
    if workers > 1:
        with get_context("fork").Pool(workers) as pool:
            pool.map(_render, jobs, chunksize=1)
    else:
        list(map(_render, jobs))
    page = PAGE.format(title=html.escape(board.data.title), latents=json.dumps(chosen),
                       rate=json.dumps([round(float(r), 6) for r in board.rate]))
    (out / "index.html").write_text(page)
    return out / "index.html"


@jaxtyped(typechecker=beartype)
def segment_ids(positions: Int[np.ndarray, " n"],
                groups: Optional[Int[np.ndarray, " n"]] = None) -> Int[np.ndarray, " n"]:
    """Number contiguous runs: a new run starts at any time gap or change of group."""
    breaks = np.diff(positions) != 1
    if groups is not None:
        breaks |= np.diff(groups) != 0
    return np.r_[0, np.cumsum(breaks)].astype(np.int64)


@jaxtyped(typechecker=beartype)
def trailing_slope(values: Float[np.ndarray, "n s"], seconds: float) -> Float[np.ndarray, " n"]:
    """Ordinary-least-squares slope across each row's evenly spaced samples."""
    t = np.arange(values.shape[1]) * seconds
    t -= t.mean()
    return values @ t / (t @ t)


@jaxtyped(typechecker=beartype)
def as_flags(n: int, positions: Int[np.ndarray, " m"]) -> Bool[np.ndarray, " n"]:
    """Boolean event markers at the given in-range row positions."""
    flags = np.zeros(n, dtype=bool)
    flags[positions[(positions >= 0) & (positions < n)]] = True
    return flags
