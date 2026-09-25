"""Fixed, reproducible selection of position-tuned SED features."""

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from jaxtyping import Float, jaxtyped
from numpy.typing import NDArray


@dataclass(frozen=True)
class SelectionConfig:
    """Prespecified spatial templates and hit criteria."""

    n_position_bins: int = 40
    track_range: Tuple[float, float] = (0.0, 1.0)
    cell_centers: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
    left_widths: Tuple[float, ...] = (0.05, 0.10, 0.07, 0.07)
    right_widths: Tuple[float, ...] = (0.10, 0.05, 0.07, 0.07)
    peak_amplitudes_hz: Tuple[float, ...] = (10.0, 28.0, 20.0, 20.0)
    baseline_rate_hz: float = 0.1
    field_width_multiplier: float = 2.0
    min_shape_score: float = 0.6
    max_outside_mean: float = 0.4
    max_peak_distance: float = 0.075
    min_active_samples: int = 100
    min_active_per_field: int = 20
    min_conditional_zscore: float = 0.3
    min_decoder_loading: float = 0.1

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable configuration dictionary."""

        return asdict(self)


def position_grid(config: SelectionConfig) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return fixed position-bin edges and centers."""

    edges = np.linspace(*config.track_range, config.n_position_bins + 1, dtype=np.float64)
    return edges, (edges[:-1] + edges[1:]) / 2


@jaxtyped(typechecker=None)
def compute_tuning_curves(
    activation_table: pd.DataFrame,
    evaluation_index: pd.DataFrame,
    positions: Float[NDArray, " time"],
    n_features: int,
    config: SelectionConfig,
) -> Tuple[Float[NDArray, "feature position_bin"], Float[NDArray, " position_bin"]]:
    """Compute mean activation per eligible time bin and position bin."""

    required_activations = {"source_time_idx", "latent_idx", "activation_value"}
    required_index = {"source_time_idx"}
    if missing := required_activations.difference(activation_table.columns):
        raise ValueError(f"activation_table is missing columns: {sorted(missing)}")
    if missing := required_index.difference(evaluation_index.columns):
        raise ValueError(f"evaluation_index is missing columns: {sorted(missing)}")
    if n_features < 1:
        raise ValueError("n_features must be positive")

    positions = np.asarray(positions, dtype=np.float64)
    eligible_indices = np.sort(evaluation_index["source_time_idx"].unique().astype(np.int64))
    if (
        eligible_indices.size == 0
        or eligible_indices.min() < 0
        or eligible_indices.max() >= len(positions)
    ):
        raise ValueError("evaluation_index must contain valid source_time_idx values")

    edges, centers = position_grid(config)
    eligible_bins = np.searchsorted(edges, positions[eligible_indices], side="right") - 1
    eligible_bins = np.clip(eligible_bins, 0, config.n_position_bins - 1)
    occupancy = np.bincount(eligible_bins, minlength=config.n_position_bins).astype(np.float64)

    collapsed = (
        activation_table.loc[:, ["source_time_idx", "latent_idx", "activation_value"]]
        .groupby(["source_time_idx", "latent_idx"], as_index=False)["activation_value"]
        .max()
    )
    source_indices = collapsed["source_time_idx"].to_numpy(dtype=np.int64)
    latent_indices = collapsed["latent_idx"].to_numpy(dtype=np.int64)
    if source_indices.size and (
        source_indices.min() < 0
        or source_indices.max() >= len(positions)
        or latent_indices.min() < 0
        or latent_indices.max() >= n_features
    ):
        raise ValueError("activation_table contains an out-of-range index")
    if not np.isin(source_indices, eligible_indices).all():
        raise ValueError("activation_table includes sources outside evaluation_index")

    source_bins = np.searchsorted(edges, positions[source_indices], side="right") - 1
    source_bins = np.clip(source_bins, 0, config.n_position_bins - 1)
    flat_indices = latent_indices * config.n_position_bins + source_bins
    sums = np.bincount(
        flat_indices,
        weights=collapsed["activation_value"].to_numpy(dtype=np.float64),
        minlength=n_features * config.n_position_bins,
    ).reshape(n_features, config.n_position_bins)
    curves = np.divide(
        sums,
        occupancy[None, :],
        out=np.zeros_like(sums),
        where=occupancy[None, :] > 0,
    )
    return curves, centers


@jaxtyped(typechecker=None)
def normalize_tuning_curves(
    tuning_curves: Float[NDArray, "feature position_bin"],
) -> Float[NDArray, "feature position_bin"]:
    """Normalize every feature independently to unit peak activation."""

    tuning_curves = np.asarray(tuning_curves, dtype=np.float64)
    if tuning_curves.ndim != 2 or not np.isfinite(tuning_curves).all():
        raise ValueError("tuning_curves must be a finite [feature, position_bin] array")
    peaks = tuning_curves.max(axis=1, keepdims=True)
    return np.divide(tuning_curves, peaks, out=np.zeros_like(tuning_curves), where=peaks > 0)


def ground_truth_curves(positions: NDArray, config: SelectionConfig) -> NDArray:
    """Evaluate exact generative rates, including baseline, then normalize peaks."""
    displacement = np.asarray(positions)[None, :] - np.asarray(config.cell_centers)[:, None]
    widths = np.where(
        displacement < 0,
        np.asarray(config.left_widths)[:, None],
        np.asarray(config.right_widths)[:, None],
    )
    rates = config.baseline_rate_hz + np.asarray(config.peak_amplitudes_hz)[:, None] * np.exp(
        -0.5 * (displacement / widths) ** 2
    )
    return normalize_tuning_curves(rates)


def field_support(positions: NDArray, config: SelectionConfig) -> NDArray:
    """Use asymmetric two-width support (rate above baseline >= exp(-2) of amplitude)."""
    displacement = np.asarray(positions)[None, :] - np.asarray(config.cell_centers)[:, None]
    return (
        displacement >= -config.field_width_multiplier * np.asarray(config.left_widths)[:, None]
    ) & (displacement <= config.field_width_multiplier * np.asarray(config.right_widths)[:, None])


def _shape_similarity(
    curves: NDArray[np.float64], template: NDArray[np.float64]
) -> NDArray[np.float64]:
    rmse = np.sqrt(np.mean((curves - template[None, :]) ** 2, axis=1))
    return np.clip(1.0 - rmse, 0.0, 1.0)


@jaxtyped(typechecker=None)
def score_tuning_curves(
    tuning_curves: Float[NDArray, "feature position_bin"],
    position_centers: Float[NDArray, " position_bin"],
    config: SelectionConfig,
) -> pd.DataFrame:
    """Score all features against all four exact, prespecified generative shapes."""

    curves = normalize_tuning_curves(tuning_curves)
    centers = np.asarray(position_centers, dtype=np.float64)
    if centers.ndim != 1 or curves.shape[1] != centers.size:
        raise ValueError("position_centers must match the tuning-curve position axis")

    peak_positions = centers[curves.argmax(axis=1)]
    result = pd.DataFrame(
        {"latent_idx": np.arange(curves.shape[0]), "peak_position": peak_positions}
    )
    supports = field_support(centers, config)
    for unit, template in enumerate(ground_truth_curves(centers, config)):
        target = f"cell_{unit}"
        score = _shape_similarity(curves, template)
        distance = np.abs(peak_positions - config.cell_centers[unit])
        outside = curves[:, ~supports[unit]].mean(axis=1)
        result[f"{target}_score"] = score
        result[f"{target}_peak_distance"] = distance
        result[f"{target}_outside_mean"] = outside
        result[f"{target}_hit"] = (
            (np.asarray(tuning_curves).max(axis=1) > 0)
            & (score >= config.min_shape_score)
            & (distance <= config.max_peak_distance)
            & (outside <= config.max_outside_mean)
        )
    return result


def best_features(scores: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """Return each cell's best passing feature, with no failed-feature fallback."""

    required = {"latent_idx"} | {
        f"cell_{unit}_{key}" for unit in range(4) for key in ("score", "hit")
    }
    if missing := required.difference(scores.columns):
        raise ValueError(f"scores is missing columns: {sorted(missing)}")
    results: Dict[str, Dict[str, object]] = {}
    for target in (f"cell_{unit}" for unit in range(4)):
        passing = scores.loc[scores[f"{target}_hit"]].sort_values(
            [f"{target}_score", "latent_idx"], ascending=[False, True]
        )
        if passing.empty:
            results[target] = {"latent_idx": None, "score": None, "hit": False}
            continue
        row = passing.iloc[0]
        results[target] = {
            "latent_idx": int(row["latent_idx"]),
            "score": float(row[f"{target}_score"]),
            "hit": bool(row[f"{target}_hit"]),
        }
    return results


def biological_traceback(
    activation_table: pd.DataFrame,
    zscores: NDArray[np.float32],
    positions: NDArray[np.float32],
    decoder: NDArray[np.float32],
    scores: pd.DataFrame,
    config: SelectionConfig,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Add support and biological-coactivity checks, without causal attribution.

    Units 0 through 3 are known place cells. Require both strongest coactivity
    and strongest positive unit-normalized decoder loading to identify the same
    expected unit, making passing assignments across cells necessarily distinct.
    """

    n_features, n_units = decoder.shape
    collapsed = (
        activation_table.loc[activation_table.activation_value > 0]
        .groupby(["latent_idx", "source_time_idx"])["activation_value"]
        .max()
        .reset_index()
    )
    latent = collapsed.latent_idx.to_numpy(dtype=np.int64)
    source = collapsed.source_time_idx.to_numpy(dtype=np.int64)
    values = collapsed.activation_value.to_numpy(dtype=np.float64)
    decoder = np.asarray(decoder, dtype=np.float64)
    decoder = decoder / np.maximum(np.linalg.norm(decoder, axis=1, keepdims=True), 1e-12)
    count = np.bincount(latent, minlength=n_features)
    sums = np.zeros((n_features, n_units), dtype=np.float64)
    weighted_sums = np.zeros_like(sums)
    np.add.at(sums, latent, zscores[source])
    np.add.at(weighted_sums, latent, zscores[source] * values[:, None])
    conditional = sums / np.maximum(count[:, None], 1)
    weighted = weighted_sums / np.maximum(
        np.bincount(latent, weights=values, minlength=n_features)[:, None], 1e-12
    )
    result = scores.copy()
    result["active_samples"] = count
    result["strongest_coactive_unit"] = conditional.argmax(axis=1)
    result["strongest_decoder_unit"] = decoder.argmax(axis=1)
    source_support = field_support(positions[source], config)
    field_counts = np.stack(
        [
            np.bincount(
                latent[support],
                minlength=n_features,
            )
            for support in source_support
        ],
        axis=1,
    )
    for unit in range(4):
        target = f"cell_{unit}"
        support = (count >= config.min_active_samples) & (
            field_counts[:, unit] >= config.min_active_per_field
        )
        biology = (
            (conditional[:, unit] >= config.min_conditional_zscore)
            & (decoder[:, unit] >= config.min_decoder_loading)
            & (conditional.argmax(axis=1) == unit)
            & (decoder.argmax(axis=1) == unit)
        )
        result[f"{target}_shape_hit"] = result[f"{target}_hit"]
        result[f"{target}_support_hit"] = support
        result[f"{target}_biology_hit"] = biology
        result[f"{target}_hit"] &= support & biology
    return result, {
        "conditional_zscore": conditional,
        "activation_weighted_zscore": weighted,
        "active_counts": count,
        "field_active_counts": field_counts,
        "decoder_weights": decoder,
    }
