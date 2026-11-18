"""Align the processed Allen session 732592105 with its NWB stimulus tables.

The processed ``spike_counts.csv`` stores 50-ms counts of 201 units scaled by a
constant (``COUNT_SCALE``); row ``i`` covers ``[0.05 i, 0.05 (i + 1))`` s on the NWB
session clock (verified exactly against the NWB spike times). This module builds a
per-bin stimulus table from the NWB presentation tables, labelling each response
bin with the stimulus shown ``DELAY_S`` earlier, and a presentation-level 80/20
split of the recording.

Run ``python -m experiments.rebuttal.change04_allen.data`` to write the cache.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped

ROOT = Path("/nfs/nhome/live/jbhagat/nldisco/data/allen_data")
SESSION = ROOT / "sessions_proc/732592105"
NWB = ROOT / "session_732592105/session_732592105.nwb"
CACHE = Path("/ceph/aeon/aeon/nldisco/rebuttal_20261118/change04_allen/cache")
BIN_S = 0.05
DELAY_S = 0.05
COUNT_SCALE = 0.04345703
SPONTANEOUS_CHUNK_S = 10.0
TRAIN_FRACTION = 0.8
SPLIT_SEED = 0
PARAMS = ("orientation", "temporal_frequency", "spatial_frequency", "phase", "color",
          "x_position", "y_position", "frame")


def _decode(values: np.ndarray) -> np.ndarray:
    return np.array([v.decode() if isinstance(v, bytes) else v for v in values], dtype=object)


def _float(values: np.ndarray) -> np.ndarray:
    """Parse scalar NWB strings ('0.25', 'N/A', '[0.0, 0.0]') to floats; lists become NaN."""
    out = np.full(len(values), np.nan)
    for i, v in enumerate(_decode(values)):
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            pass
    return out


@beartype
def load_presentations(nwb: Path = NWB) -> tuple[pd.DataFrame, np.ndarray]:
    """Return all stimulus presentations (sorted by start) and invalid intervals."""
    frames = []
    with h5py.File(nwb, "r") as f:
        for name, group in f["intervals"].items():
            if name == "invalid_times":
                invalid = np.stack([group["start_time"][:], group["stop_time"][:]], 1)
                continue
            table = {"start": group["start_time"][:], "stop": group["stop_time"][:]}
            table["stim"] = name.removesuffix("_presentations")
            for param in PARAMS:
                table[param] = (_float(group[param][:]) if param in group
                                else np.full(len(table["start"]), np.nan))
            frames.append(pd.DataFrame(table))
    pres = pd.concat(frames).sort_values("start", ignore_index=True)
    return pres, invalid


@beartype
def split_units(pres: pd.DataFrame, chunk_s: float = SPONTANEOUS_CHUNK_S) -> np.ndarray:
    """Group presentations into split units.

    Each discrete presentation (grating, gabor, flash, scene) is one unit; a natural
    movie repeat (frames from 0 to the next 0) is one unit; spontaneous intervals are
    cut into ``chunk_s`` pieces, handled by :func:`bin_table`. Returns one integer
    unit id per presentation row.
    """
    movie = pres.stim.str.startswith("natural_movie").to_numpy()
    new_unit = ~movie | (pres.frame.to_numpy() == 0) | (pres.stim != pres.stim.shift()).to_numpy()
    return np.cumsum(new_unit) - 1


@jaxtyped(typechecker=beartype)
def label_times(
    times: Float[np.ndarray, " n"], starts: Float[np.ndarray, " p"], stops: Float[np.ndarray, " p"]
) -> tuple[Int[np.ndarray, " n"], Bool[np.ndarray, " n"]]:
    """Index of the latest presentation starting at or before each time.

    Returns the presentation index (-1 before the first) and whether the time lies
    inside that presentation (``start <= time < stop``) rather than in the gray gap
    after it. ``starts`` must be sorted.
    """
    idx = np.searchsorted(starts, times, side="right") - 1
    inside = (idx >= 0) & (times < stops[np.clip(idx, 0, None)])
    return idx, inside


@beartype
def bin_table(pres: pd.DataFrame, invalid: np.ndarray, n_bins: int,
              bin_s: float = BIN_S, delay_s: float = DELAY_S,
              chunk_s: float = SPONTANEOUS_CHUNK_S, train_fraction: float = TRAIN_FRACTION,
              seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Label every response bin with the stimulus shown ``delay_s`` before its midpoint.

    Bins whose stimulus time falls in the gray gap after a presentation get
    ``stim='<name>_gray'`` with the preceding presentation's split unit. Split units
    are assigned to train/test at random, stratified by stimulus name.
    """
    t0 = np.arange(n_bins) * bin_s
    tau = t0 + bin_s / 2 - delay_s
    idx, inside = label_times(tau, pres.start.to_numpy(), pres.stop.to_numpy())
    has = idx >= 0
    row = pres.iloc[np.clip(idx, 0, None)].reset_index(drop=True)
    df = pd.DataFrame({"bin": np.arange(n_bins), "t": t0, "stim_time": tau})
    df["pres_idx"] = np.where(has, idx, -1)
    df["stim"] = np.where(~has, "none", np.where(inside, row.stim, row.stim + "_gray"))
    df["onset"] = np.where(has, row.start, np.nan)
    df["since_onset"] = tau - df.onset
    for param in PARAMS:
        df[param] = np.where(has & inside, row[param], np.nan)
    unit = split_units(pres)[np.clip(idx, 0, None)]
    spont = has & (row.stim == "spontaneous").to_numpy()
    # Spontaneous intervals (incl. their tails) are chunked so they split like other stimuli.
    key = np.where(spont, "s" + df.pres_idx.astype(str) + "_" +
                   np.floor(df.since_onset.fillna(0) / chunk_s).astype(int).astype(str),
                   "u" + pd.Series(unit).astype(str))
    df["split_unit"] = pd.factorize(np.where(has, key, "none"))[0]
    df["invalid"] = ((t0[:, None] < invalid[:, 1]) & (t0[:, None] + bin_s > invalid[:, 0])).any(1)
    # Stratified random assignment of split units.
    units = df[has].groupby("split_unit").stim.first().str.removesuffix("_gray")
    rng = np.random.default_rng(seed)
    test_units = set()
    for _, members in units.groupby(units):
        ids = rng.permutation(members.index.to_numpy())
        test_units.update(ids[: int(round(len(ids) * (1 - train_fraction)))].tolist())
    df["split"] = np.where(~has, "none", np.where(df.split_unit.isin(test_units), "test", "train"))
    return df


@beartype
def load_counts() -> tuple[Float[np.ndarray, "t n"], np.ndarray]:
    """Processed scaled counts (as used by the model-evaluation figure) and unit ids."""
    path = CACHE / "spike_counts.npy"
    if not path.exists():
        csv = SESSION / "spike_counts.csv"
        np.save(path, pd.read_csv(csv).to_numpy(np.float32))
        np.save(CACHE / "unit_ids.npy",
                pd.read_csv(csv, nrows=0).columns.astype(int).to_numpy())
    return np.load(path), np.load(CACHE / "unit_ids.npy")


@beartype
def unit_areas(unit_ids: np.ndarray, nwb: Path = NWB) -> np.ndarray:
    """CCF area acronym of each unit's peak channel."""
    with h5py.File(nwb, "r") as f:
        e, u = f["general/extracellular_ephys/electrodes"], f["units"]
        loc = dict(zip(e["id"][:], _decode(e["location"][:])))
        peak = dict(zip(u["id"][:], u["peak_channel_id"][:]))
    return np.array([loc[peak[i]] for i in unit_ids])


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    counts, ids = load_counts()
    pres, invalid = load_presentations()
    df = bin_table(pres, invalid, len(counts))
    df.to_parquet(CACHE / "bins.parquet")
    pd.Series(unit_areas(ids), index=ids, name="area").to_csv(CACHE / "unit_areas.csv")
    print(df.groupby(["stim", "split"]).size().unstack(fill_value=0))
    print("invalid bins", int(df.invalid.sum()))


if __name__ == "__main__":
    main()
