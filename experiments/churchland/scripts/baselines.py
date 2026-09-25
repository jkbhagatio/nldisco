"""Neural input and trial-boundary helpers for the CEBRA training runner."""
from pathlib import Path
import numpy as np
from beartype import beartype
from jaxtyping import Int, jaxtyped
ROOT = Path(__file__).resolve().parents[1]

@jaxtyped(typechecker=beartype)
def trial_slices(trial_id: Int[np.ndarray, "time"]) -> list:
    """Return contiguous slices and reject disjoint repeated trial identifiers."""
    boundaries = np.r_[0, np.flatnonzero(np.diff(trial_id)) + 1, len(trial_id)]
    if len(np.unique(trial_id)) != len(boundaries) - 1:
        raise ValueError("Trial identifiers must occupy exactly one contiguous block.")
    return [slice(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:])]


@beartype
def load_data(path: Path) -> tuple:
    """Load only neural inputs and trial boundaries, never behavioral labels."""
    counts = np.load(path / "counts.npy")
    with np.load(path / "metadata.npz") as metadata:
        trials = metadata["trial_id"]
    if counts.ndim != 2 or counts.dtype != np.float32:
        raise ValueError("Expected float32 [time, neuron] counts.")
    if len(trials) != len(counts) or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Invalid neural counts or trial alignment.")
    trial_slices(trials)
    return counts, trials

