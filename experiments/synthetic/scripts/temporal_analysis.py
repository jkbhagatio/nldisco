"""Motion and descriptive direction summaries for the paper experiments."""

import numpy as np
import pandas as pd


def window_motion(positions, starts, length, minimum=0.1, pair_centers=(0.2, 0.4)):
    """Keep strict raw motion and separately describe jitter-tolerant pair crossings."""
    trajectories = positions[starts[:, None] + np.arange(length)]
    steps = np.diff(trajectories, axis=1)
    displacement = trajectories[:, -1] - trajectories[:, 0]
    reversal = (steps.min(axis=1) < -1e-10) & (steps.max(axis=1) > 1e-10)
    low, high = pair_centers
    crossing = (trajectories.min(axis=1) <= low + 0.02) & (
        trajectories.max(axis=1) >= high - 0.02
    )
    excursion = np.where(
        displacement[:, None] >= 0,
        np.maximum.accumulate(trajectories, axis=1) - trajectories,
        trajectories - np.minimum.accumulate(trajectories, axis=1),
    ).max(axis=1)
    efficiency = np.abs(displacement) / (np.abs(steps).sum(axis=1) + 1e-12)
    purposeful = (
        crossing & (np.abs(displacement) >= high - low - 0.04)
        & (excursion <= 0.02) & (efficiency >= 0.8)
    )
    mean_position = trajectories.mean(axis=1)
    return pd.DataFrame(
        {
            "start": starts,
            "position": mean_position,
            "start_position": trajectories[:, 0],
            "end_position": trajectories[:, -1],
            "displacement": displacement,
            "direction": np.sign(displacement).astype(int),
            "reversal": reversal,
            "clean": (np.abs(displacement) >= minimum) & ~reversal,
            "first_pair_crossing": crossing,
            "strict_first_pair_transition": crossing & ~reversal,
            "purposeful_first_pair_transition": purposeful,
            "maximum_backtracking_m": excursion,
            "net_path_efficiency": efficiency,
            "ramp_clean": crossing & ~reversal & (mean_position >= low) & (mean_position <= high),
        }
    )


def matched_direction_metrics(
    activations, metadata, bins=10, minimum_bin_count=10, eligible_column="clean"
):
    """Match position occupancy across directions, including all inactive zeros.

    Each position bin receives weight proportional to the smaller direction
    count. Every feature uses exactly the same weights and eligible windows.
    """
    position_bin = np.clip((metadata.position.to_numpy() * bins).astype(int), 0, bins - 1)
    direction = metadata.direction.to_numpy()
    clean = metadata[eligible_column].to_numpy()
    means, weights = [], []
    for bin_id in range(bins):
        masks = [clean & (position_bin == bin_id) & (direction == sign) for sign in (-1, 1)]
        counts = [int(mask.sum()) for mask in masks]
        if min(counts) < minimum_bin_count:
            continue
        means.append(np.stack([activations[mask].mean(axis=0) for mask in masks]))
        weights.append(min(counts))
    if not weights:
        matched = np.zeros((2, activations.shape[1]))
    else:
        matched = np.average(np.stack(means), axis=0, weights=weights)
    left, right = matched
    return pd.DataFrame(
        {
            "latent_idx": np.arange(activations.shape[1]),
            "left_mean_activation": left,
            "right_mean_activation": right,
            "right_selectivity": (right - left) / (right + left + 1e-12),
            "left_active_windows": (activations[clean & (direction == -1)] > 0).sum(axis=0),
            "right_active_windows": (activations[clean & (direction == 1)] > 0).sum(axis=0),
            "matched_position_bins": len(weights),
            "matched_windows_per_direction": sum(weights),
        }
    )


def decoder_metrics(decoder, centers):
    """Measure known-place-cell enrichment and center-of-mass temporal motion."""
    positive = np.maximum(decoder, 0)
    energy = positive**2
    place = energy[:, : len(centers)]
    enrichment = place.mean(axis=(0, 1)) / (energy[:, len(centers) :].mean(axis=(0, 1)) + 1e-12)
    # Early/late weighted centers use all four known place units, without sorting peaks.
    half = max(1, decoder.shape[0] // 3)

    def center(segment):
        mass = segment.sum(axis=0)
        return (mass * centers[:, None]).sum(axis=0) / (mass.sum(axis=0) + 1e-12)

    span = center(place[-half:]) - center(place[:half])
    pair = place[:, :2]
    pair_mass = pair.sum(axis=0)
    pair_total = pair_mass.sum(axis=0)

    def pair_center(segment):
        mass = segment.sum(axis=0)
        return (mass * centers[:2, None]).sum(axis=0) / (mass.sum(axis=0) + 1e-12)

    pair_span = pair_center(pair[-half:]) - pair_center(pair[:half])
    return pd.DataFrame(
        {
            "latent_idx": np.arange(decoder.shape[-1]),
            "global_place_energy_enrichment": enrichment,
            "global_decoder_center_displacement": span,
            "decoder_center_displacement": pair_span,
            "place_energy_enrichment": pair.mean(axis=(0, 1)) / (
                energy[:, len(centers):].mean(axis=(0, 1)) + 1e-12
            ),
            "pair_min_unit_energy_fraction": pair_mass.min(axis=0) / (pair_total + 1e-12),
            "pair_known_place_energy_fraction": pair_total / (place.sum(axis=(0, 1)) + 1e-12),
            "place_energy_fraction": place.sum(axis=(0, 1)) / (energy.sum(axis=(0, 1)) + 1e-12),
        }
    )


def select_features(metrics):
    """Select one descriptive feature per direction from the supplied observations."""
    selected = {}
    eligible = metrics.left_active_windows + metrics.right_active_windows >= 20
    eligible &= metrics.matched_position_bins >= 2
    eligible &= metrics.pair_min_unit_energy_fraction >= 0.1
    eligible &= metrics.pair_known_place_energy_fraction >= 0.5
    for label, sign in (("right", 1), ("left", -1)):
        score = (sign * metrics.right_selectivity).clip(lower=0)
        score *= (sign * metrics.decoder_center_displacement).clip(lower=0)
        score *= np.log1p(metrics.place_energy_enrichment)
        score = score.where(eligible, -1)
        passed = eligible & (sign * metrics.right_selectivity >= 0.25)
        passed &= sign * metrics.decoder_center_displacement >= 0.01
        passed &= metrics.place_energy_enrichment >= 2
        metrics[f"{label}_selection_score"] = score
        metrics[f"{label}_passes"] = passed
        candidates = score.where(passed, -1) if passed.any() else score
        selected[label] = int(metrics.loc[candidates.idxmax(), "latent_idx"])
    return selected


def nonoverlapping_top_indices(activation, starts, length, count=16):
    """Greedily choose highest activating disjoint windows, without direction filtering."""
    selected = []
    for index in np.argsort(-activation, kind="stable"):
        if activation[index] <= 0:
            break
        if all(abs(int(starts[index]) - int(starts[other])) >= length for other in selected):
            selected.append(int(index))
            if len(selected) == count:
                break
    return np.asarray(selected, dtype=int)
