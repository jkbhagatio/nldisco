"""Input normalization and fingerprints used by the window SED experiment."""
import hashlib
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from beartype import beartype
from jaxtyping import Float, jaxtyped

@beartype
def file_sha256(path: Path) -> str:
    """Hash an input without loading the entire file a second time."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@jaxtyped(typechecker=beartype)
def prepare_inputs(
    counts: Float[np.ndarray, "time neuron"],
    trial_ids: np.ndarray,
    smoothing_sigma_bins: float,
    normalization: str,
) -> tuple:
    """Transform counts using neural data only, retaining every trial row."""
    if counts.shape[0] != len(trial_ids) or not np.isfinite(counts).all():
        raise ValueError("Counts and trial identities must be finite and aligned")
    if np.any(counts < 0) or smoothing_sigma_bins < 0:
        raise ValueError("Counts and smoothing width must be nonnegative")
    values = counts.copy()
    boundaries = np.r_[0, np.flatnonzero(np.diff(trial_ids) != 0) + 1, len(trial_ids)]
    if len(np.unique(trial_ids)) != len(boundaries) - 1:
        raise ValueError("A trial must occupy one contiguous row interval")
    if smoothing_sigma_bins:
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            values[start:end] = gaussian_filter1d(
                values[start:end], sigma=smoothing_sigma_bins, axis=0, mode="reflect"
            )
    if normalization in {"unit_std", "unit_zscore"}:
        scale = values.std(axis=0, keepdims=True)
    elif normalization == "global_max":
        scale = np.full((1, values.shape[1]), values.max(), dtype=np.float32)
    elif normalization == "none":
        scale = np.ones((1, values.shape[1]), dtype=np.float32)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    scale = np.maximum(scale, 1e-8).astype(np.float32)
    mean = (
        values.mean(axis=0, keepdims=True)
        if normalization == "unit_zscore"
        else np.zeros_like(scale)
    )
    return np.asarray((values - mean) / scale, dtype=np.float32), scale, mean

