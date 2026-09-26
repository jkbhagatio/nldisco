"""Load sorter exports and preprocess activity in [timebin, unit] order.

Kilosort loading uses the exported sample clock, without behavioral-clock
synchronization or stitching recordings. Other loaders can reuse ``bin_spikes``
and the normalization functions without depending on Kilosort itself.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Sequence
from jaxtyping import Float, Integer, Real, jaxtyped

PathLike = Union[str, Path]
Method = Literal["zscore", "minmax"]
UnitIds = Union[Sequence[int], Integer[np.ndarray, "unit"]]  # noqa: F821
Seconds = Union[int, float]


@dataclass(frozen=True)
class BinnedSpikes:
    """Integer counts, bin edges in seconds, and original cluster IDs by column.

    Bins are left-inclusive and right-exclusive, including the final bin.
    The last bin can be shorter than the requested bin size.
    """

    counts: Integer[np.ndarray, "timebin unit"]
    bin_edges: Float[np.ndarray, "edge"]  # noqa: F821
    unit_ids: Integer[np.ndarray, "unit"]  # noqa: F821

    @property
    def timestamps(self) -> np.ndarray:
        """Return bin centers in seconds on the exported sample clock."""
        return (self.bin_edges[:-1] + self.bin_edges[1:]) / 2


def _positive_chunk_size(chunk_size: int) -> None:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")


def _sample_boundary(seconds: float, sampling_rate: float, name: str) -> int:
    samples = seconds * sampling_rate
    if not np.isfinite(samples) or samples < 0 or samples >= np.iinfo(np.int64).max:
        raise ValueError(f"{name} must be finite, nonnegative, and fit in int64 samples")
    rounded = round(samples)
    if abs(samples - rounded) > 1e-6:
        raise ValueError(f"{name} must align to an integer sample boundary")
    return rounded


def _output(shape: tuple, dtype: type, output_path: Optional[PathLike]) -> np.ndarray:
    if output_path is None:
        return np.empty(shape, dtype=dtype)
    path = Path(output_path)
    # Reserve the name exclusively: never truncate existing recordings or outputs.
    with path.open("xb"):
        pass
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _flush(array: np.ndarray) -> None:
    if isinstance(array, np.memmap):
        array.flush()


@jaxtyped(typechecker=beartype)
def bin_spikes(
    spike_samples: Integer[np.ndarray, "spike"],  # noqa: F821
    spike_units: Integer[np.ndarray, "spike"],  # noqa: F821
    *,
    sampling_rate: Seconds,
    bin_size: Seconds,
    stop_time: Seconds,
    start_time: Seconds = 0.0,
    unit_ids: Optional[UnitIds] = None,
    chunk_size: int = 1_000_000,
    output_path: Optional[PathLike] = None,
) -> BinnedSpikes:
    """Bin integer sample indices into a [timebin, unit] count matrix.

    Args:
        spike_samples: Nonnegative sample indices; sorting is not required.
        spike_units: Nonnegative cluster IDs paired with the sample indices.
        sampling_rate: Samples per second, strictly positive.
        bin_size: Bin duration in seconds, an integer number of samples.
        stop_time: Exclusive recording/selection end in seconds. Required to
            preserve trailing silence rather than infer duration from spikes.
        start_time: Inclusive recording/selection start in seconds.
        unit_ids: Optional unique IDs to retain, including silent units. Output
            columns are sorted by ID. By default retain all IDs in the input,
            even when a unit has no spikes in the selected interval.
        chunk_size: Maximum spikes processed at once.
        output_path: Optional new .npy file for memory-mapped int64 counts.

    Returns:
        Counts with bin edges and an explicit column-to-cluster mapping.

    Raises:
        ValueError: Invalid timing, IDs, or chunk size.
        FileExistsError: The output file already exists.
    """
    _positive_chunk_size(chunk_size)
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be finite and positive")
    first = _sample_boundary(start_time, sampling_rate, "start_time")
    last = _sample_boundary(stop_time, sampling_rate, "stop_time")
    width = _sample_boundary(bin_size, sampling_rate, "bin_size")
    if width < 1 or last <= first:
        raise ValueError("bin_size must be positive and stop_time must exceed start_time")

    ids = np.empty(0, dtype=np.int64)
    if unit_ids is not None:
        ids = np.asarray(unit_ids)
        if ids.ndim != 1 or (ids.size and ids.dtype.kind not in "iu"):
            raise ValueError("unit_ids must be an integer vector")
        if (
            np.any(ids < 0)
            or np.any(ids > np.iinfo(np.int64).max)
            or len(np.unique(ids)) != len(ids)
        ):
            raise ValueError("unit_ids must be unique nonnegative integers")
        ids = np.sort(ids.astype(np.int64))
    for start in range(0, len(spike_samples), chunk_size):
        samples = spike_samples[start : start + chunk_size]
        units = spike_units[start : start + chunk_size]
        if np.any(samples < 0) or np.any(samples > np.iinfo(np.int64).max) or np.any(units < 0):
            raise ValueError("Spike samples and IDs must be nonnegative int64 values")
        if np.any(units > np.iinfo(np.int64).max):
            raise ValueError("Spike IDs must fit in int64")
        if unit_ids is None:
            ids = np.union1d(ids, np.unique(units).astype(np.int64))

    n_bins = (last - first + width - 1) // width
    counts = _output((n_bins, len(ids)), np.int64, output_path)
    counts[:] = 0
    for start in range(0, len(spike_samples), chunk_size):
        samples = spike_samples[start : start + chunk_size]
        units = spike_units[start : start + chunk_size]
        keep = (samples >= first) & (samples < last) & np.isin(units, ids)
        bins = (samples[keep].astype(np.int64) - first) // width
        columns = np.searchsorted(ids, units[keep])
        np.add.at(counts, (bins, columns), 1)
    _flush(counts)
    edges = np.empty(n_bins + 1, dtype=np.float64)
    edges[:-1] = (first + np.arange(n_bins, dtype=np.int64) * width) / sampling_rate
    edges[-1] = last / sampling_rate
    return BinnedSpikes(counts, edges, ids)


def _load_spike_vector(path: Path) -> np.ndarray:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError(f"{path.name} must be an integer vector or an (n, 1) column")
    return values


@beartype
def load_kilosort(
    directory: PathLike,
    *,
    sampling_rate: Seconds,
    bin_size: Seconds,
    stop_time: Seconds,
    start_time: Seconds = 0.0,
    unit_ids: Optional[UnitIds] = None,
    labels: Optional[Sequence[str]] = None,
    chunk_size: int = 1_000_000,
    output_path: Optional[PathLike] = None,
) -> BinnedSpikes:
    """Load and bin Kilosort/Phy exports without installing a spike sorter.

    Reads spike_times.npy and spike_clusters.npy with memory mapping. Timing and
    output arguments follow ``bin_spikes``. No quality filtering is implicit.
    ``labels`` optionally selects cluster labels (e.g. ["good", "mua"]),
    intersected with ``unit_ids`` if supplied. Phy's cluster_group.tsv takes
    precedence over cluster_KSLabel.tsv; missing labels never fall back to stale
    sorter labels. Missing metadata raises an error when filtering is requested.
    Sampling rate is explicit; params.py is never executed and ops.npy is not
    needed. Exported spike assignments are used as-is, including Phy edits.
    """
    directory = Path(directory)
    samples = _load_spike_vector(directory / "spike_times.npy")
    units = _load_spike_vector(directory / "spike_clusters.npy")
    if samples.shape != units.shape:
        raise ValueError("spike_times.npy and spike_clusters.npy must have matching lengths")
    if labels is not None:
        path = directory / "cluster_group.tsv"
        column = "group"
        if not path.exists():
            path = directory / "cluster_KSLabel.tsv"
            column = "KSLabel"
        frame = pd.read_csv(path, sep="\t")
        # Uncurated Kilosort exports may copy KSLabel verbatim into cluster_group.tsv.
        if column == "group" and column not in frame.columns and "KSLabel" in frame.columns:
            column = "KSLabel"
        if not {"cluster_id", column}.issubset(frame.columns):
            raise ValueError(f"{path.name} requires cluster_id and {column} columns")
        if frame.cluster_id.dtype.kind not in "iu" or frame.cluster_id.duplicated().any():
            raise ValueError("Label metadata must have unique integer cluster IDs")
        selected = frame.loc[frame[column].isin(labels), "cluster_id"].tolist()
        unit_ids = (
            selected
            if unit_ids is None
            else np.intersect1d(unit_ids, np.asarray(selected, dtype=np.int64))
        )
    return bin_spikes(
        samples,
        units,
        sampling_rate=sampling_rate,
        bin_size=bin_size,
        start_time=start_time,
        stop_time=stop_time,
        unit_ids=unit_ids,
        chunk_size=chunk_size,
        output_path=output_path,
    )


def _validate_activity(data: np.ndarray, chunk_size: int) -> None:
    _positive_chunk_size(chunk_size)
    if not all(data.shape):
        raise ValueError("Activity must contain at least one timebin and one unit")


def _block(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Activity must contain only finite values")
    return values


@dataclass(frozen=True)
class NormalizationStats:
    """Per-unit statistics fitted across time; zero scales are replaced with 1.

    Z-scoring uses population standard deviation (ddof=0). A constant training
    unit maps to zero on training data; unseen deviations remain visible.
    Min-max transforms are not clipped, so held-out values may fall outside [0, 1].
    """

    method: Method
    offset: Float[np.ndarray, "unit"]  # noqa: F821
    scale: Float[np.ndarray, "unit"]  # noqa: F821

    @jaxtyped(typechecker=beartype)
    def transform(
        self,
        data: Real[np.ndarray, "timebin unit"],
        *,
        chunk_size: int = 100_000,
        output_path: Optional[PathLike] = None,
    ) -> Float[np.ndarray, "timebin unit"]:
        """Apply fixed statistics to matching unit columns, returning float64.

        ``output_path`` optionally writes a new memory-mapped .npy file. Inputs
        are never modified. Select/reorder unit columns before calling this.
        """
        _validate_activity(data, chunk_size)
        if self.offset.shape != (data.shape[1],) or self.scale.shape != self.offset.shape:
            raise ValueError("Statistics must match the number of unit columns")
        if not np.isfinite(self.offset).all() or not np.isfinite(self.scale).all():
            raise ValueError("Statistics must be finite")
        if np.any(self.scale <= 0):
            raise ValueError("Scales must be positive")
        result = _output(data.shape, np.float64, output_path)
        for start in range(0, len(data), chunk_size):
            stop = start + chunk_size
            result[start:stop] = (_block(data[start:stop]) - self.offset) / self.scale
        _flush(result)
        return result


@jaxtyped(typechecker=beartype)
def fit_normalizer(
    data: Real[np.ndarray, "timebin unit"],
    method: Method = "zscore",
    *,
    chunk_size: int = 100_000,
) -> NormalizationStats:
    """Fit per-unit statistics on training rows only, in bounded memory.

    Exclude invalid/gap rows before fitting. Reuse the returned statistics on
    validation/test activity with the same unit-column order. Float64 blockwise
    central moments avoid subtracting two large, nearly equal raw moments.
    """
    _validate_activity(data, chunk_size)
    offset = np.zeros(data.shape[1], dtype=np.float64)
    moment = np.zeros_like(offset)
    low = np.full_like(offset, np.inf)
    high = np.full_like(offset, -np.inf)
    n = 0
    for start in range(0, len(data), chunk_size):
        values = _block(data[start : start + chunk_size])
        if method == "minmax":
            low = np.minimum(low, values.min(axis=0))
            high = np.maximum(high, values.max(axis=0))
        else:
            size = len(values)
            mean = values.mean(axis=0)
            delta = mean - offset
            moment += np.square(values - mean).sum(axis=0) + delta**2 * (n * size / (n + size))
            offset += delta * (size / (n + size))
            n += size
    if method == "minmax":
        offset, scale = low, high - low
    else:
        scale = np.sqrt(moment / n)
    scale[scale == 0] = 1.0
    return NormalizationStats(method, offset, scale)


@jaxtyped(typechecker=beartype)
def normalize_activity(
    data: Real[np.ndarray, "timebin unit"],
    method: Method = "zscore",
    *,
    axis: Literal[0, 1] = 0,
    chunk_size: int = 100_000,
    output_path: Optional[PathLike] = None,
) -> Float[np.ndarray, "timebin unit"]:
    """Normalize activity without changing [timebin, unit] orientation.

    Axis 0 fits across time independently for each unit. For held-out analyses,
    use fit_normalizer(training_rows).transform(...) instead. Axis 1 normalizes
    across units independently within each timebin, requiring no training fit.
    Constant slices map to zero. Z-score uses ddof=0; min-max maps to [0, 1].
    Returns float64, optionally in a new memory-mapped .npy file.
    """
    _validate_activity(data, chunk_size)
    if axis == 0:
        return fit_normalizer(data, method, chunk_size=chunk_size).transform(
            data,
            chunk_size=chunk_size,
            output_path=output_path,
        )
    result = _output(data.shape, np.float64, output_path)
    for start in range(0, len(data), chunk_size):
        stop = start + chunk_size
        values = _block(data[start:stop])
        if method == "zscore":
            offset, scale = values.mean(axis=1, keepdims=True), values.std(axis=1, keepdims=True)
        else:
            offset = values.min(axis=1, keepdims=True)
            scale = values.max(axis=1, keepdims=True) - offset
        scale[scale == 0] = 1.0
        result[start:stop] = (values - offset) / scale
    _flush(result)
    return result
