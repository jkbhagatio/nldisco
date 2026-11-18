"""Fixed block split of the 18-hour Aeon recording, defined from timing and behavior only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from beartype import beartype

BLOCK_BINS = 45_000  # 15 minutes of 20-ms bins
HELDOUT_OFFSET = 2  # blocks with index % 5 == 2 are held out (14 of 72 blocks)
WINDOW = 20
N_BINS = 3_240_000


@beartype
def heldout_bins(n_bins: int = N_BINS, block: int = BLOCK_BINS, offset: int = HELDOUT_OFFSET) -> np.ndarray:
    """Boolean mask over recording bins that belong to held-out blocks."""
    return (np.arange(n_bins) // block) % 5 == offset


@beartype
def partition_windows(starts: np.ndarray, held: np.ndarray, width: int = WINDOW) -> np.ndarray:
    """Label each window 0 (training), 1 (held-out) or -1 (crosses a block boundary).

    A window is training if none of its bins is held-out, held-out if all of them are.
    """
    cumulative = np.r_[0, np.cumsum(held, dtype=np.int64)]
    inside = cumulative[starts + width] - cumulative[starts]
    label = np.full(len(starts), -1, dtype=np.int8)
    label[inside == 0] = 0
    label[inside == width] = 1
    return label


@beartype
def write_split(path: Path, starts: np.ndarray) -> dict:
    """Persist the split definition and summary beside the experiment outputs."""
    held = heldout_bins()
    label = partition_windows(starts, held)
    summary = dict(block_minutes=BLOCK_BINS * 0.02 / 60, rule=f"block index mod 5 == {HELDOUT_OFFSET}",
                   n_blocks=N_BINS // BLOCK_BINS, n_heldout_blocks=int(held.reshape(-1, BLOCK_BINS)[:, 0].sum()),
                   heldout_bin_fraction=float(held.mean()), n_windows=len(starts),
                   n_train=int((label == 0).sum()), n_heldout=int((label == 1).sum()),
                   n_dropped_crossing=int((label == -1).sum()))
    path.write_text(json.dumps(summary, indent=2))
    return summary
