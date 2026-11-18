"""Session loading, the fixed discovery/held-out split and endpoint behavior variables.

Everything here is independent of any model. The split and the behavior variables
were fixed before session B existed (see `PROTOCOL.md` beside this file).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from scipy.ndimage import binary_closing, binary_opening

SEQ_LEN = 20
BIN_SECONDS = 0.02
N_BLOCKS = 20
HELDOUT_BLOCKS = (3, 7, 11, 15, 19)
VALIDATION_FRACTION = 0.1
FIT, VALIDATION, HELDOUT, DROPPED = 0, 1, 2, -1
WHEEL_ACTIVE = 0.05  # turns/s, above encoder jitter (as in the paper's Aeon analysis).
MOVING = 20.0  # px/s, the speed above which the prepared table defines a heading.

# Variables a rule may use; all are read at the last bin of the 0.4 s window.
VARIABLES = {
    "x": "centroid x position (px; image coordinates, larger = right)",
    "y": "centroid y position (px; image coordinates, larger = down)",
    "patch_distance": "distance from the centroid to the foraging-patch centre (px)",
    "speed": "centroid speed (px/s)",
    "acceleration": "rate of change of centroid speed (px/s^2; negative = slowing)",
    "heading": "movement direction (deg in (-180, 180]; 0 = +x/right, 90 = +y/down, "
               "-90 = up, +-180 = left); undefined below 20 px/s",
    "heading_error_to_patch": "signed angle between movement direction and the direction "
                              "to the patch (deg; 0 = straight at the patch, +-180 = "
                              "straight away); undefined below 20 px/s",
    "radial_velocity": "velocity component along the line to the patch (px/s; negative "
                       "= approaching the patch)",
    "area": "segmented body area in the top camera (px^2)",
    "wheel_speed": "unsigned running-wheel speed (turns/s)",
    "time_since_pellet": "seconds since the most recent food-pellet trigger",
    "time_to_pellet": "seconds until the next food-pellet trigger",
}
CAMERA_VARIABLES = ("x", "y", "patch_distance", "speed", "acceleration", "heading",
                    "heading_error_to_patch", "radial_velocity", "area")
MOVEMENT_VARIABLES = ("heading", "heading_error_to_patch")


@jaxtyped(typechecker=beartype)
def bin_parts(n_bins: int) -> tuple[Int[np.ndarray, "time"], Int[np.ndarray, "time"]]:
    """Assign every time bin to a part and a contiguous segment.

    The recording is cut into `N_BLOCKS` equal blocks. Blocks in `HELDOUT_BLOCKS`
    form the held-out part. In every other block the first 90% of bins are the fit
    portion and the last 10% the validation portion of the discovery part.

    Returns:
        part (FIT, VALIDATION or HELDOUT) and a segment identifier that changes at
        every boundary between portions.
    """
    edges = np.round(np.arange(N_BLOCKS + 1) * n_bins / N_BLOCKS).astype(np.int64)
    part = np.empty(n_bins, dtype=np.int64)
    segment = np.empty(n_bins, dtype=np.int64)
    for block, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if block in HELDOUT_BLOCKS:
            part[lo:hi], segment[lo:hi] = HELDOUT, 2 * block
            continue
        cut = hi - int(round((hi - lo) * VALIDATION_FRACTION))
        part[lo:cut], segment[lo:cut] = FIT, 2 * block
        part[cut:hi], segment[cut:hi] = VALIDATION, 2 * block + 1
    return part, segment


@jaxtyped(typechecker=beartype)
def window_parts(starts: Int[np.ndarray, "windows"], n_bins: int,
                 seq_len: int = SEQ_LEN) -> Int[np.ndarray, "windows"]:
    """Part of every window; windows touching two segments are DROPPED."""
    part, segment = bin_parts(n_bins)
    ends = starts + seq_len - 1
    return np.where(segment[starts] == segment[ends], part[starts], DROPPED)


@jaxtyped(typechecker=beartype)
def complete_support(valid: Bool[np.ndarray, "time"],
                     width: int = SEQ_LEN) -> Bool[np.ndarray, "time"]:
    """True where the `width` bins ending at a bin are all valid."""
    cumulative = np.r_[0, np.cumsum(valid, dtype=np.int64)]
    result = np.zeros(len(valid), dtype=bool)
    result[width - 1:] = cumulative[width:] - cumulative[:-width] == width
    return result


@jaxtyped(typechecker=beartype)
def event_clocks(event_bins: Int[np.ndarray, "events"], n_bins: int,
                 ) -> tuple[Float[np.ndarray, "time"], Float[np.ndarray, "time"]]:
    """Seconds since the latest event and until the next one (NaN where none)."""
    since = np.full(n_bins, np.nan)
    until = np.full(n_bins, np.nan)
    events = np.unique(event_bins[(event_bins >= 0) & (event_bins < n_bins)])
    if len(events):
        bins = np.arange(n_bins)
        previous = np.searchsorted(events, bins, side="right") - 1
        following = np.searchsorted(events, bins, side="left")
        has_previous, has_following = previous >= 0, following < len(events)
        since[has_previous] = (bins[has_previous] - events[previous[has_previous]]) * BIN_SECONDS
        until[has_following] = (events[following[has_following]] - bins[has_following]) * BIN_SECONDS
    return since, until


@jaxtyped(typechecker=beartype)
def wheel_bouts(wheel_speed: Float[np.ndarray, "time"], valid: Bool[np.ndarray, "time"],
                ) -> tuple[Int[np.ndarray, "bouts"], Int[np.ndarray, "bouts"]]:
    """First bin of each wheel bout and first bin after it.

    A bout is wheel speed above `WHEEL_ACTIVE`, with pauses shorter than 0.2 s
    filled and bouts shorter than 0.4 s removed.
    """
    raw = valid & np.isfinite(wheel_speed) & (np.abs(np.nan_to_num(wheel_speed)) > WHEEL_ACTIVE)
    moving = binary_opening(binary_closing(raw, structure=np.ones(10)),
                            structure=np.ones(20)) & valid
    change = np.diff(np.r_[False, moving, False].astype(np.int8))
    return np.flatnonzero(change == 1), np.flatnonzero(change == -1)


@dataclass
class Session:
    """One prepared Aeon session and its fixed window split."""

    root: Path
    counts: np.ndarray
    neural_valid: np.ndarray
    starts: np.ndarray
    endpoints: np.ndarray
    part: np.ndarray
    behavior: pd.DataFrame
    variables: dict
    support: dict

    @property
    def discovery(self) -> np.ndarray:
        """Windows of the discovery part (fit and validation portions)."""
        return (self.part == FIT) | (self.part == VALIDATION)

    @property
    def heldout(self) -> np.ndarray:
        """Windows of the held-out part."""
        return self.part == HELDOUT


@beartype
def endpoint_variables(behavior: pd.DataFrame, pellet_bins: np.ndarray,
                       endpoints: np.ndarray) -> tuple[dict, dict]:
    """Rule variables at window endpoints and their 20-bin support masks.

    Returns:
        variables: name -> float array over windows (NaN where undefined).
        support: 'camera', 'wheel', 'all' -> bool arrays over windows that are true
            when the whole window has valid tracking / wheel samples.
    """
    n_bins = len(behavior)
    camera = behavior["camera_analysis_valid"].to_numpy(dtype=bool) if \
        "camera_analysis_valid" in behavior else behavior["camera_valid"].to_numpy(dtype=bool)
    wheel_valid = behavior["wheel_valid"].to_numpy(dtype=bool)
    raw = {name: behavior[name].to_numpy(dtype=float) for name in
           ("x", "y", "patch_distance", "speed", "acceleration", "radial_velocity", "area",
            "wheel_speed")}
    raw["heading"] = np.degrees(behavior["heading"].to_numpy(dtype=float))
    raw["heading_error_to_patch"] = np.degrees(
        behavior["heading_error_to_patch"].to_numpy(dtype=float))
    raw["time_since_pellet"], raw["time_to_pellet"] = event_clocks(
        pellet_bins.astype(np.int64), n_bins)
    always = ("x", "y", "patch_distance", "speed", "area")
    camera = camera & np.logical_and.reduce([np.isfinite(raw[name]) for name in always])
    wheel_valid = wheel_valid & np.isfinite(raw["wheel_speed"])
    support = {"camera": complete_support(camera)[endpoints],
               "wheel": complete_support(wheel_valid)[endpoints]}
    support["all"] = np.ones(len(endpoints), dtype=bool)
    variables = {name: raw[name][endpoints] for name in VARIABLES}
    onsets, offsets = wheel_bouts(raw["wheel_speed"], wheel_valid)
    variables["_time_since_wheel_onset"] = event_clocks(onsets, n_bins)[0][endpoints]
    variables["_time_since_wheel_offset"] = event_clocks(offsets, n_bins)[0][endpoints]
    return variables, support


@beartype
def load_session(root: Path) -> Session:
    """Load a prepared session (format of `prepare_sweep_18h.py`) with its split."""
    counts = np.load(root / "counts.npy", mmap_mode="r")
    starts = np.asarray(np.load(root / "valid_starts.npy"), dtype=np.int64)
    if np.any(np.diff(starts) <= 0):
        raise ValueError("Window starts must be strictly chronological")
    behavior = pd.read_parquet(root / "behavior.parquet")
    if len(behavior) != len(counts):
        raise ValueError("Behavior table and counts differ in length")
    pellets = pd.read_csv(root / "pellet_times.csv")
    pellet_bins = np.floor(pellets["relative_seconds"].to_numpy() / BIN_SECONDS).astype(np.int64)
    endpoints = starts + SEQ_LEN - 1
    variables, support = endpoint_variables(behavior, pellet_bins, endpoints)
    neural_valid = np.asarray(np.load(root / "neural_valid.npy"), dtype=bool)
    return Session(root, counts, neural_valid, starts, endpoints,
                   window_parts(starts, len(counts)), behavior, variables, support)


@beartype
def load_neural(root: Path) -> Session:
    """Load counts and the window split only; runners never read behavior."""
    counts = np.load(root / "counts.npy", mmap_mode="r")
    starts = np.asarray(np.load(root / "valid_starts.npy"), dtype=np.int64)
    if np.any(np.diff(starts) <= 0):
        raise ValueError("Window starts must be strictly chronological")
    neural_valid = np.asarray(np.load(root / "neural_valid.npy"), dtype=bool)
    return Session(root, counts, neural_valid, starts, starts + SEQ_LEN - 1,
                   window_parts(starts, len(counts)), pd.DataFrame(), {}, {})


@jaxtyped(typechecker=beartype)
def fit_bins(session: Session) -> Bool[np.ndarray, "time"]:
    """Valid neural bins of the fit portion, for normalization statistics."""
    return session.neural_valid & (bin_parts(len(session.counts))[0] == FIT)


@jaxtyped(typechecker=beartype)
def lagged_pairs(starts: Int[np.ndarray, "windows"], member: Bool[np.ndarray, "windows"],
                 lag: int) -> Int[np.ndarray, "pairs"]:
    """Indices i of member windows whose window `lag` bins later is also a member."""
    index = np.flatnonzero(member[:-lag] & member[lag:] & (starts[lag:] - starts[:-lag] == lag))
    return index.astype(np.int64)
