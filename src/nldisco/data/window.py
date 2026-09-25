"""Boundary-safe construction of fixed-length spike-count windows."""

from collections import namedtuple
from typing import Collection, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch as t
from torch import Tensor
from torch.utils.data import Dataset


class WindowSample(namedtuple(
    "WindowSample", "values window_index source_indices occurrence_mask anchor_index "
    "trial_code session_code targets",
)):
    """Aligned input/target windows and their shared source-row identity.

    ``values`` remains the input for existing callers. Omitted targets reuse
    values, including in the seven-argument legacy constructor. Every field is
    a tensor, so PyTorch's default named-tuple collation works in both cases.
    """

    __slots__ = ()

    values: Tensor  # [seq_len, n_neurons]
    window_index: Tensor  # scalar index among valid windows
    source_indices: Tensor  # [seq_len]
    occurrence_mask: Tensor  # [seq_len], temporal positions eligible for sparse selection
    anchor_index: Tensor  # scalar source index for lag 0
    trial_code: Tensor  # scalar dataset-local trial code, or -1
    session_code: Tensor  # scalar dataset-local session code, or -1
    targets: Tensor  # [seq_len, n_output_neurons]

    def __new__(
        cls, values: Tensor, window_index: Tensor, source_indices: Tensor,
        occurrence_mask: Tensor, anchor_index: Tensor, trial_code: Tensor,
        session_code: Tensor, targets: Optional[Tensor] = None,
    ):
        return super().__new__(
            cls, values, window_index, source_indices, occurrence_mask,
            anchor_index, trial_code, session_code, values if targets is None else targets,
        )

    @property
    def target(self) -> Tensor:
        """Supervision tensor, identical to values for an autoencoder."""
        return self.targets


def _aligned_metadata(name, shared, target, n_rows):
    """Use shared metadata, checking optional independently supplied target rows."""
    if target is None:
        return shared
    target_values = np.asarray(target)
    if target_values.shape != (n_rows,):
        raise ValueError(f"target_{name} must have shape [timebin]")
    if shared is None:
        return target
    shared_values = np.asarray(shared)
    if shared_values.shape != (n_rows,):
        raise ValueError(f"{name} must have shape [timebin]")
    # Compare only present values: object-array equality containing pd.NA can
    # otherwise collapse to a scalar instead of returning rowwise comparisons.
    shared_missing, target_missing = pd.isna(shared_values), pd.isna(target_values)
    present = ~(shared_missing | target_missing)
    if not np.array_equal(shared_values[present], target_values[present]) or (
        name == "source_indices" and not np.array_equal(shared_missing, target_missing)
    ):
        raise ValueError(f"target_{name} must be temporally aligned with {name}")
    if np.any(target_missing & ~shared_missing):
        # A missing identity on either side invalidates the shared row instead of
        # supplying an otherwise valid input identity to the target population.
        shared_values = shared_values.astype(float if name == "timestamps" else object)
        shared_values[shared_missing | target_missing] = np.nan
        return shared_values
    return shared


def _is_missing(value: object) -> bool:
    try:
        missing = pd.isna(value)
        return bool(missing) if np.ndim(missing) == 0 else False
    except (TypeError, ValueError, OverflowError):
        return False


def _encode_groups(values: Optional[np.ndarray], n_rows: int) -> np.ndarray:
    if values is None:
        return np.full(n_rows, -1, dtype=np.int64)
    if len(values) != n_rows:
        raise ValueError("Group arrays must have the same number of rows as spike_counts")

    result = np.full(n_rows, -1, dtype=np.int64)
    mapping = {}
    next_code = 0
    for idx, value in enumerate(values):
        if _is_missing(value):
            continue
        if value not in mapping:
            mapping[value] = next_code
            next_code += 1
        result[idx] = mapping[value]
    return result


class SpikeWindowDataset(Dataset):
    """View spike-count rows as valid within-trial, within-session windows.

    The dataset never creates a window that crosses a supplied trial/session
    boundary or a discontinuity in timestamps. The final row is the window's
    lag-0 anchor. Optional targets have the same rows and time grid, with an
    independent unit width. Shared metadata describes both populations; target
    metadata, when supplied separately, must agree on present rows. Missing
    metadata or nonfinite activity on either side invalidates the shared row.
    """

    def __init__(
        self,
        spike_counts: Tensor,
        seq_len: int,
        stride: int = 1,
        *,
        targets: Optional[Tensor] = None,
        source_indices: Optional[Sequence[int]] = None,
        trial_ids: Optional[Sequence[object]] = None,
        session_ids: Optional[Sequence[object]] = None,
        timestamps: Optional[Sequence[float]] = None,
        expected_bin_size: Optional[float] = None,
        occurrence_support: Optional[Tuple[int, int]] = None,
        allowed_rows: Optional[Sequence[bool]] = None,
        allowed_trial_ids: Optional[Collection[object]] = None,
        allowed_session_ids: Optional[Collection[object]] = None,
        valid_rows: Optional[Sequence[bool]] = None,
        target_valid_rows: Optional[Sequence[bool]] = None,
        target_source_indices: Optional[Sequence[int]] = None,
        target_trial_ids: Optional[Sequence[object]] = None,
        target_session_ids: Optional[Sequence[object]] = None,
        target_timestamps: Optional[Sequence[float]] = None,
    ) -> None:
        if spike_counts.ndim != 2 or spike_counts.shape[1] < 1:
            raise ValueError("spike_counts must have shape [timebin, neuron]")
        if targets is not None and (
            targets.ndim != 2 or targets.shape[0] != spike_counts.shape[0]
            or targets.shape[1] < 1
        ):
            raise ValueError("targets must have shape [timebin, output_unit] with matching rows")
        if seq_len < 1 or stride < 1:
            raise ValueError("seq_len and stride must be positive")

        if occurrence_support is not None:
            required_left, required_right = occurrence_support
            if required_left < 0 or required_right < 0:
                raise ValueError("occurrence_support values must be nonnegative")
            usable_positions = seq_len - required_left - required_right
            if usable_positions < 1:
                raise ValueError("occurrence_support must leave at least one usable position")
            if stride > usable_positions:
                raise ValueError(
                    "stride cannot exceed the number of usable positions when "
                    "occurrence_support is supplied"
                )

        self.spike_counts = spike_counts
        self.targets = spike_counts if targets is None else targets
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        n_rows = spike_counts.shape[0]

        resolved_source_indices = np.arange(n_rows) if source_indices is None else source_indices
        _aligned_metadata("source_indices", resolved_source_indices, target_source_indices, n_rows)
        if len(resolved_source_indices) != n_rows:
            raise ValueError("source_indices must match spike_counts rows")
        self.source_indices = t.as_tensor(resolved_source_indices, dtype=t.long)

        trial_ids = _aligned_metadata("trial_ids", trial_ids, target_trial_ids, n_rows)
        session_ids = _aligned_metadata("session_ids", session_ids, target_session_ids, n_rows)
        timestamps = _aligned_metadata("timestamps", timestamps, target_timestamps, n_rows)
        trial_values = None if trial_ids is None else np.asarray(trial_ids, dtype=object)
        session_values = None if session_ids is None else np.asarray(session_ids, dtype=object)
        self.trial_codes = _encode_groups(trial_values, n_rows)
        self.session_codes = _encode_groups(session_values, n_rows)

        if timestamps is not None:
            if len(timestamps) != n_rows:
                raise ValueError("timestamps must match spike_counts rows")
            self.timestamps = np.asarray(timestamps, dtype=float)
            if expected_bin_size is None:
                raise ValueError("expected_bin_size is required when timestamps are supplied")
        else:
            self.timestamps = None
        if expected_bin_size is not None and (
            not np.isfinite(expected_bin_size) or expected_bin_size <= 0
        ):
            raise ValueError("expected_bin_size must be finite and positive")
        self.expected_bin_size = expected_bin_size
        if allowed_rows is None:
            self.allowed_rows = np.ones(n_rows, dtype=bool)
        else:
            if len(allowed_rows) != n_rows:
                raise ValueError("allowed_rows must match spike_counts rows")
            self.allowed_rows = np.asarray(allowed_rows, dtype=bool).copy()

        self.valid_rows = np.ones(n_rows, dtype=bool)
        for name, row_mask in (("valid_rows", valid_rows), ("target_valid_rows", target_valid_rows)):
            if row_mask is not None:
                mask = np.asarray(row_mask)
                if mask.shape != (n_rows,) or mask.dtype != np.bool_:
                    raise ValueError(f"{name} must contain one boolean per timebin")
                self.valid_rows &= mask
        for values in (self.spike_counts, self.targets):
            self.valid_rows &= t.isfinite(values).all(dim=1).cpu().numpy()
        if self.timestamps is not None:
            self.valid_rows &= np.isfinite(self.timestamps)
        if trial_values is not None:
            self.valid_rows &= self.trial_codes >= 0
        if session_values is not None:
            self.valid_rows &= self.session_codes >= 0
        self.allowed_rows &= self.valid_rows

        allowed_trials = None if allowed_trial_ids is None else set(allowed_trial_ids)
        allowed_sessions = None if allowed_session_ids is None else set(allowed_session_ids)
        candidate_starts = []
        for start in range(0, max(0, n_rows - self.seq_len + 1)):
            stop = start + self.seq_len
            valid_start = self.allowed_rows[start:stop].all()
            valid_start = valid_start and self._has_constant_group(
                self.trial_codes[start:stop], trial_values, allowed_trials, start
            )
            valid_start = valid_start and self._has_constant_group(
                self.session_codes[start:stop], session_values, allowed_sessions, start
            )
            if valid_start and self.timestamps is not None and self.seq_len > 1:
                diffs = np.diff(self.timestamps[start:stop])
                expected = self.expected_bin_size
                assert expected is not None
                if expected <= 0 or not np.allclose(diffs, expected, rtol=1e-5, atol=1e-8):
                    valid_start = False
            if not valid_start:
                continue
            candidate_starts.append(start)

        candidate_runs = []
        for start in candidate_starts:
            if not candidate_runs or start != candidate_runs[-1][-1] + 1:
                candidate_runs.append([start])
            else:
                candidate_runs[-1].append(start)

        self.valid_starts = []
        self.occurrence_masks = []
        for run in candidate_runs:
            selected_starts = run[:: self.stride]
            if occurrence_support is not None and selected_starts[-1] != run[-1]:
                selected_starts.append(run[-1])

            covered_sources = set()
            for start in selected_starts:
                occurrence_mask = t.ones(self.seq_len, dtype=t.bool)
                if occurrence_support is not None:
                    occurrence_mask.zero_()
                    required_left, required_right = occurrence_support
                    stop = self.seq_len - required_right
                    for time_idx in range(required_left, stop):
                        source_idx = int(self.source_indices[start + time_idx])
                        if source_idx not in covered_sources:
                            occurrence_mask[time_idx] = True
                            covered_sources.add(source_idx)
                self.valid_starts.append(start)
                self.occurrence_masks.append(occurrence_mask)

    def _has_constant_group(
        self,
        codes: np.ndarray,
        original: Optional[np.ndarray],
        allowed: Optional[set],
        start: int,
    ) -> bool:
        if original is None:
            return allowed is None
        if codes[0] < 0 or np.any(codes != codes[0]):
            return False
        if allowed is not None and original[start] not in allowed:
            return False
        return True

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, index: int) -> WindowSample:
        start = self.valid_starts[index]
        stop = start + self.seq_len
        source_indices = self.source_indices[start:stop]
        return WindowSample(
            values=self.spike_counts[start:stop],
            targets=self.targets[start:stop],
            window_index=t.tensor(index, dtype=t.long),
            source_indices=source_indices,
            occurrence_mask=self.occurrence_masks[index],
            anchor_index=source_indices[-1],
            trial_code=t.tensor(self.trial_codes[start], dtype=t.long),
            session_code=t.tensor(self.session_codes[start], dtype=t.long),
        )
