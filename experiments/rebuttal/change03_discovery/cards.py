"""Part B: identical text summary cards for latents of any method.

A card is built from the discovery part only and never names the method or the
latent index. The same code path produces every card.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from beartype import beartype
from scipy.stats import rankdata

from experiments.rebuttal.change03_discovery.data import BIN_SECONDS

HIGH_FRACTION = 0.05
FLOOR_MAJORITY = 0.5
MIN_BIN_WINDOWS = 200
MAP_CELL = 100.0
MAP_MIN_WINDOWS = 500
INF = float("inf")

FIXED_EDGES = {
    "patch_distance": [0, 50, 100, 200, 400, 600, 800, INF],
    "speed": [0, 2, 5, 20, 50, 100, 200, INF],
    "acceleration": [-INF, -400, -200, -50, 50, 200, 400, INF],
    "radial_velocity": [-INF, -100, -20, -5, 5, 20, 100, INF],
    "wheel_speed": [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, INF],
    "time_since_pellet": [0, 0.5, 1, 2, 5, 15, 60, INF],
    "time_to_pellet": [0, 0.5, 1, 2, 5, 15, 60, INF],
}
UNITS = {"x": "px", "y": "px", "patch_distance": "px", "speed": "px/s",
         "acceleration": "px/s^2", "heading": "deg", "heading_error_to_patch": "deg",
         "radial_velocity": "px/s", "area": "px^2", "wheel_speed": "turns/s",
         "time_since_pellet": "s", "time_to_pellet": "s"}
SECTORS = ("right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right")


@dataclass
class Binning:
    """Bin index of every discovery window for one variable (-1 = not shown)."""

    name: str
    labels: list
    index: np.ndarray


def _label(lo: float, hi: float) -> str:
    text = lambda v: f"{v:g}"
    if lo == -INF:
        return f"<{text(hi)}"
    if hi == INF:
        return f">={text(lo)}"
    return f"{text(lo)}..{text(hi)}"


@beartype
def build_binnings(variables: dict, support: dict) -> list[Binning]:
    """Bin every rule variable over the given (discovery) windows.

    Edges are fixed, except x, y (eight equal bins over the central 99% of visited
    positions) and area (quantiles), which depend on behavior only.
    """
    camera, wheel = support["camera"], support["wheel"]
    binnings = []

    def cut(name, edges, valid):
        values = variables[name]
        index = np.digitize(values, edges[1:-1]).astype(np.int16)
        index[~(valid & np.isfinite(values)) | (values < edges[0]) | (values > edges[-1])] = -1
        binnings.append(Binning(name, [_label(a, b) for a, b in zip(edges[:-1], edges[1:])], index))

    for name in ("x", "y"):
        lo, hi = np.quantile(variables[name][camera], [0.005, 0.995])
        edges = np.round(np.linspace(lo, hi, 9), -1)
        cut(name, [-INF, *edges[1:-1], INF], camera)
    cut("patch_distance", FIXED_EDGES["patch_distance"], camera)
    cut("speed", FIXED_EDGES["speed"], camera)
    cut("acceleration", FIXED_EDGES["acceleration"], camera)
    heading = variables["heading"]
    sector = (np.floor((np.nan_to_num(heading) + 22.5) / 45) % 8).astype(np.int16)
    sector[~np.isfinite(heading)] = 8
    sector[~camera] = -1
    binnings.append(Binning("heading", [*SECTORS, "not moving"], sector))
    error = np.abs(variables["heading_error_to_patch"])
    index = np.digitize(np.nan_to_num(error), [30, 60, 90, 120, 150]).astype(np.int16)
    index[~np.isfinite(error)] = 6
    index[~camera] = -1
    binnings.append(Binning("abs(heading_error_to_patch)",
                            ["0..30", "30..60", "60..90", "90..120", "120..150", "150..180",
                             "not moving"], index))
    cut("radial_velocity", FIXED_EDGES["radial_velocity"], camera)
    edges = np.unique(np.round(np.quantile(variables["area"][camera],
                                           [.05, .2, .4, .6, .8, .95])))
    cut("area", [-INF, *edges, INF], camera)
    cut("wheel_speed", FIXED_EDGES["wheel_speed"], wheel)
    everything = support["all"]
    for name in ("time_since_pellet", "time_to_pellet"):
        cut(name, FIXED_EDGES[name], everything)
    return binnings


@beartype
def states(values: np.ndarray) -> dict:
    """HIGH and LOW states of a latent, by one rule for every latent.

    If the latent sits at its minimum in at least half of the windows, HIGH is every
    window above that minimum and there is no LOW state. Otherwise HIGH is the top
    5% of values and LOW the bottom 5%.
    """
    minimum = float(values.min())
    floor = float((values == minimum).mean())
    if floor >= FLOOR_MAJORITY:
        return dict(floor=floor, high=values > minimum, low=None, high_threshold=minimum,
                    low_threshold=None,
                    high_text=f"value above its floor ({minimum:.4g}); the latent sits at "
                              f"this floor in {100 * floor:.1f}% of windows")
    low_threshold, high_threshold = np.quantile(values, [HIGH_FRACTION, 1 - HIGH_FRACTION])
    return dict(floor=floor, high=values > high_threshold, low=values < low_threshold,
                high_threshold=float(high_threshold), low_threshold=float(low_threshold),
                high_text=f"value > {high_threshold:.4g} (top 5% of values)")


def _episodes(state: np.ndarray, endpoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start index and length of each run of consecutive windows in a state."""
    joined = state[1:] & state[:-1] & (np.diff(endpoints) == 1)
    starts = np.flatnonzero(state & ~np.r_[False, joined])
    ends = np.flatnonzero(state & ~np.r_[joined, False])
    return starts, ends - starts + 1


def _percent(count: np.ndarray, total: np.ndarray, minimum: int) -> list:
    return ["-" if n < minimum else f"{100 * c / n:.0f}" if 100 * c / n >= 9.5
            else f"{100 * c / n:.1f}" for c, n in zip(count, total)]


def _row(label: str, cells: list) -> str:
    return f"| {label} | " + " | ".join(str(c) for c in cells) + " |"


@beartype
def position_map(x: np.ndarray, y: np.ndarray, valid: np.ndarray,
                 state: Optional[np.ndarray] = None) -> str:
    """Text map over 100-px cells: % of windows in the state, or occupancy if None."""
    cx, cy = np.floor(x / MAP_CELL).astype(int), np.floor(y / MAP_CELL).astype(int)
    xs = np.arange(cx[valid].min(), cx[valid].max() + 1)
    ys = np.arange(cy[valid].min(), cy[valid].max() + 1)
    shape = (len(ys), len(xs))
    flat = (cy[valid] - ys[0]) * len(xs) + (cx[valid] - xs[0])
    total = np.bincount(flat, minlength=shape[0] * shape[1]).reshape(shape)
    if state is None:
        cells = np.where(total > 0, np.char.mod("%.1f", 100 * total / total.sum()), ".")
    else:
        hits = np.bincount(flat, weights=state[valid], minlength=total.size).reshape(shape)
        cells = np.where(total >= MAP_MIN_WINDOWS,
                         np.char.mod("%.0f", 100 * hits / np.maximum(total, 1)), ".")
    lines = ["y\\x | " + " ".join(f"{int(v * MAP_CELL):>4d}" for v in xs)]
    lines += [f"{int(v * MAP_CELL):>4d}| " + " ".join(f"{c:>4s}" for c in row)
              for v, row in zip(ys, cells)]
    return "\n".join(lines)


@beartype
def render_card(card_id: str, values: np.ndarray, variables: dict, support: dict,
                binnings: list, endpoints: np.ndarray, blocks: np.ndarray,
                rng: np.random.Generator) -> str:
    """Render one latent's card from its values over the discovery windows.

    Args:
        card_id: anonymous identifier printed on the card.
        values: latent value per discovery window.
        variables, support: endpoint variables and support over the same windows.
        binnings: output of `build_binnings` for the same windows.
        endpoints: last bin of each window (to find consecutive windows).
        blocks: chronological block index of each window.
        rng: generator used to draw example windows.
    """
    values = np.asarray(values, dtype=np.float64)
    state = states(values)
    high, low = state["high"], state["low"]
    rank = 100 * (rankdata(values) - 0.5) / len(values)
    quantiles = np.quantile(values, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    lines = [f"# Latent card {card_id}", "",
             f"Summarises {len(values):,} windows (0.4 s each, one every 20 ms, "
             f"{len(values) * BIN_SECONDS / 3600:.1f} h in total).", "",
             "## Values",
             "- percentiles (min, 1, 5, 25, 50, 75, 95, 99, max): "
             + ", ".join(f"{q:.3g}" for q in quantiles),
             f"- HIGH state: {state['high_text']}; {100 * high.mean():.2f}% of windows"]
    lines.append(f"- LOW state: value < {state['low_threshold']:.4g} (bottom 5% of values); "
                 f"{100 * low.mean():.2f}% of windows" if low is not None else
                 "- LOW state: none (the latent has a floor)")
    for name, flag in (("HIGH", high), ("LOW", low)):
        if flag is None:
            continue
        _, lengths = _episodes(flag, endpoints)
        seconds = lengths * BIN_SECONDS
        per_block = np.bincount(blocks, weights=flag) / np.maximum(np.bincount(blocks), 1)
        lines.append(f"- {name} episodes: {len(lengths):,}; duration median "
                     f"{np.median(seconds) if len(seconds) else 0:.2f} s, 90th percentile "
                     f"{np.quantile(seconds, .9) if len(seconds) else 0:.2f} s")
        lines.append(f"- % of windows {name} in each consecutive block of the recording: "
                     + " ".join(f"{100 * p:.1f}" for p in per_block[np.bincount(blocks) > 0]))
    lines += ["", "## Tuning to each behavior variable",
              "Rows: % of all windows in the bin; mean rank of the latent value in the bin "
              "(0-100, 50 = no preference); % of the bin's windows that are HIGH"
              + ("; % that are LOW." if low is not None else "."), ""]
    for binning in binnings:
        keep = binning.index >= 0
        n = len(binning.labels)
        total = np.bincount(binning.index[keep], minlength=n)
        mean_rank = np.bincount(binning.index[keep], weights=rank[keep], minlength=n) \
            / np.maximum(total, 1)
        unit = UNITS.get(binning.name.replace("abs(", "").rstrip(")"), "")
        lines.append(_row(f"{binning.name} [{unit}]", binning.labels))
        lines.append("|" + "---|" * (n + 1))
        lines.append(_row("% windows", [f"{100 * t / keep.sum():.1f}" for t in total]))
        lines.append(_row("mean rank", ["-" if t < MIN_BIN_WINDOWS else f"{m:.0f}"
                                        for t, m in zip(total, mean_rank)]))
        lines.append(_row("% HIGH", _percent(
            np.bincount(binning.index[keep], weights=high[keep], minlength=n), total,
            MIN_BIN_WINDOWS)))
        if low is not None:
            lines.append(_row("% LOW", _percent(
                np.bincount(binning.index[keep], weights=low[keep], minlength=n), total,
                MIN_BIN_WINDOWS)))
        lines.append("")
    camera = support["camera"]
    for name, flag in (("HIGH", high), ("LOW", low)):
        if flag is None:
            continue
        lines += [f"## Position map: % of windows {name} in each 100-px cell "
                  "('.' = too few windows)", "```",
                  position_map(variables["x"], variables["y"], camera, flag), "```", ""]
    fields = ("x", "y", "patch_distance", "speed", "heading", "area", "wheel_speed",
              "time_since_pellet", "time_to_pellet")
    for name, flag, count in (("HIGH", high, 8), ("LOW", low, 4)):
        if flag is None:
            continue
        starts, lengths = _episodes(flag, endpoints)
        if not len(starts):
            continue
        chosen = np.sort(rng.choice(len(starts), size=min(count, len(starts)), replace=False))
        lines += [f"## Example {name} windows (random episodes; values at the middle of the "
                  "episode)", _row("episode length [s]", fields), "|" + "---|" * (len(fields) + 1)]
        for start, length in zip(starts[chosen], lengths[chosen]):
            i = start + length // 2
            lines.append(_row(f"{length * BIN_SECONDS:.2f}",
                              ["-" if not np.isfinite(variables[f][i]) else
                               f"{variables[f][i]:.0f}" if abs(variables[f][i]) >= 100 else
                               f"{variables[f][i]:.3g}" for f in fields]))
        lines.append("")
    return "\n".join(lines)
