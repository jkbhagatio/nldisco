"""Full-recording descriptive latent exploration for the strict Aeon 18 h sweep.

All feature searches, cut points and reported scores use the same recording.
Scores are descriptive associations, not held-out decoding or causal effects.
"""

# ruff: noqa: F821 -- jaxtyping symbolic dimensions are not Python names.

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from scipy.ndimage import binary_closing, binary_opening, maximum_filter1d
from scipy.stats import rankdata


@dataclass
class Condition:
    """A behavior-defined condition with its own valid comparison population."""

    name: str
    family: str
    mask: np.ndarray
    valid: np.ndarray
    definition: str


@jaxtyped(typechecker=beartype)
def complete_support(valid: Bool[np.ndarray, "n"], width: int = 20) -> Bool[np.ndarray, "n"]:
    """Require all causal input-window bins to have valid behavior."""
    cumulative = np.r_[0, np.cumsum(valid, dtype=np.int64)]
    result = np.zeros(len(valid), dtype=bool)
    result[width - 1 :] = cumulative[width:] - cumulative[:-width] == width
    return result


@beartype
def centered_support(valid: np.ndarray, radius: int) -> np.ndarray:
    """Require valid observations on both sides, without wrapping recording edges."""
    causal = complete_support(valid, 2 * radius + 1)
    result = np.zeros(len(valid), dtype=bool)
    if radius:
        result[:-radius] = causal[radius:]
    else:
        result[:] = causal
    return result


@jaxtyped(typechecker=beartype)
def bout_starts(
    mask: Bool[np.ndarray, "n"], endpoints: Int[np.ndarray, "n"]
) -> Int[np.ndarray, "m"]:
    """Find contiguous condition events without bridging omitted neural bins."""
    previous = np.r_[False, mask[:-1] & (np.diff(endpoints) == 1)]
    return np.flatnonzero(mask & ~previous)


@beartype
def separated_bout_starts(
    mask: np.ndarray, endpoints: np.ndarray, gap_bins: int = 50
) -> np.ndarray:
    """Group behavior-positive bins into episodes separated by >=1 s or neural gaps."""
    positive = np.flatnonzero(mask)
    if not len(positive):
        return positive
    segments = np.cumsum(np.r_[False, np.diff(endpoints) != 1])
    start = np.r_[
        True, (np.diff(endpoints[positive]) >= gap_bins) | (np.diff(segments[positive]) != 0)
    ]
    return positive[start]


@beartype
def build_conditions(behavior: pd.DataFrame, endpoints: np.ndarray) -> list[Condition]:
    """Construct explicitly documented quantile, circular and temporal motifs."""
    conditions = []
    camera_valid = behavior.camera_valid.to_numpy(dtype=bool)
    if "tracking_suspect" in behavior:
        camera_valid = camera_valid & ~behavior.tracking_suspect.to_numpy(dtype=bool)
    wheel_valid = behavior.wheel_valid.to_numpy(dtype=bool)
    for field in (
        "x",
        "y",
        "area",
        "speed",
        "acceleration",
        "angular_velocity",
        "patch_distance",
        "radial_velocity",
        "wheel_speed",
        "wheel_acceleration",
    ):
        values = behavior[field].to_numpy(dtype=float)
        validity = wheel_valid if field.startswith("wheel") else camera_valid
        valid = complete_support(validity & np.isfinite(values))[endpoints]
        values = values[endpoints]
        if valid.sum() < 100:
            continue
        edges = np.unique(np.quantile(values[valid], [0, 0.2, 0.4, 0.6, 0.8, 1]))
        for i, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (
                valid
                & (values >= lower)
                & (values <= upper if i == len(edges) - 2 else values < upper)
            )
            conditions.append(
                Condition(
                    f"{field}_q{i + 1}",
                    field,
                    mask,
                    valid,
                    f"{lower:.9g} <= {field} {'<=' if i == len(edges) - 2 else '<'} {upper:.9g}; empirical full-recording quantiles",
                )
            )
    speed = behavior.speed.to_numpy(dtype=float)
    moving_threshold = float(np.nanquantile(speed[camera_valid & np.isfinite(speed)], 0.4))
    heading = behavior.heading.to_numpy(dtype=float)
    valid = complete_support(camera_valid & np.isfinite(heading) & (speed > moving_threshold))[
        endpoints
    ]
    angles = ((heading[endpoints] + np.pi) % (2 * np.pi)) - np.pi
    for i in range(8):
        lower, upper = -np.pi + i * np.pi / 4, -np.pi + (i + 1) * np.pi / 4
        conditions.append(
            Condition(
                f"heading_octant{i}",
                "heading",
                valid & (angles >= lower) & (angles < upper),
                valid,
                f"movement-derived heading in [{lower:.6g},{upper:.6g}) rad, speed > {moving_threshold:.6g}",
            )
        )
    distance = behavior.patch_distance.to_numpy(dtype=float)
    radial = behavior.radial_velocity.to_numpy(dtype=float)
    valid_full = camera_valid & np.isfinite(distance) & np.isfinite(radial) & np.isfinite(speed)
    near_threshold = float(np.nanquantile(distance[valid_full], 0.25))
    radial_threshold = float(np.nanquantile(np.abs(radial[valid_full]), 0.5))
    valid = complete_support(valid_full)[endpoints]
    for side, distance_mask in [
        ("occupancy_near", distance <= near_threshold),
        ("occupancy_far", distance > near_threshold),
    ]:
        for direction, radial_mask in [
            ("toward", radial < -radial_threshold),
            ("away", radial > radial_threshold),
        ]:
            mask = (
                valid
                & distance_mask[endpoints]
                & radial_mask[endpoints]
                & (speed[endpoints] > moving_threshold)
            )
            conditions.append(
                Condition(
                    f"patch_{side}_{direction}",
                    "patch_motion",
                    mask,
                    valid,
                    f"patch {side} uses distance Q25={near_threshold:.6g}; radial {direction} magnitude>{radial_threshold:.6g}; speed>{moving_threshold:.6g}",
                )
            )
    wheel = np.abs(behavior.wheel_speed.to_numpy(dtype=float))
    finite = wheel_valid & np.isfinite(wheel)
    patch_valid = complete_support(camera_valid & np.isfinite(distance))[endpoints]
    near = distance[endpoints] <= 100.0
    conditions.append(
        Condition(
            "patch_within100px",
            "patch_proximity",
            near & patch_valid,
            patch_valid,
            "CameraTop centroid <=100px from Patch2 center (891.75,851.5)",
        )
    )
    wheel_comparison = complete_support(finite)[endpoints]
    conditions.append(
        Condition(
            "wheel_active_gt0.05",
            "wheel_activity",
            wheel_comparison & (wheel[endpoints] > 0.05),
            wheel_comparison,
            "Wheel speed >0.05turn/s; fixed threshold above baseline encoder jitter",
        )
    )
    near_compare = (
        patch_valid
        & near
        & wheel_comparison
        & ((wheel[endpoints] > 0.05) | (wheel[endpoints] <= 0.02))
    )
    conditions.append(
        Condition(
            "nearpatch_wheelactive_vs_quiescent",
            "nearpatch_wheel",
            near_compare & (wheel[endpoints] > 0.05),
            near_compare,
            "Within100px: wheel>0.05turn/s versus wheel<=0.02turn/s; intermediate speeds omitted",
        )
    )
    patch_direction = np.arctan2(851.5 - behavior.y.to_numpy(), 891.75 - behavior.x.to_numpy())
    heading_error = (heading - patch_direction + np.pi) % (2 * np.pi) - np.pi
    if "heading_error_to_patch" in behavior:
        heading_error = behavior.heading_error_to_patch.to_numpy(dtype=float)
    alignment_valid = complete_support(camera_valid & np.isfinite(heading_error) & (speed > 20))[
        endpoints
    ]
    for name, mask in [
        ("toward", np.abs(heading_error) < np.pi / 4),
        ("away", np.abs(heading_error) >= 3 * np.pi / 4),
    ]:
        conditions.append(
            Condition(
                f"heading_aligned_{name}_patch",
                "patch_heading",
                alignment_valid & mask[endpoints],
                alignment_valid,
                f"Movement heading {name} patch; error{'<45' if name == 'toward' else '>=135'}degrees; speed>20px/s",
            )
        )
    area = behavior.area.to_numpy(dtype=float)
    derivative = np.full(len(area), np.nan)
    derivative[5:] = (area[5:] - area[:-5]) / 0.1
    derivative[~complete_support(camera_valid & np.isfinite(area), 6)] = np.nan
    derivative_definition = "100ms backward area difference/0.1s"
    if "area_derivative" in behavior:
        derivative = behavior.area_derivative.to_numpy(dtype=float)
        derivative_definition = (
            "shared-preprocessing causal100ms mean followed by100ms backward difference"
        )
    for family, values, eligible in [
        ("area_derivative", derivative, camera_valid),
        ("nearpatch_area", area, camera_valid & (distance <= 100)),
    ]:
        valid = complete_support(eligible & np.isfinite(values))[endpoints]
        if valid.sum() < 100:
            continue
        edges = np.unique(np.quantile(values[endpoints][valid], [0, 0.2, 0.4, 0.6, 0.8, 1]))
        for i, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (
                valid
                & (values[endpoints] >= lower)
                & (
                    values[endpoints] <= upper
                    if i == len(edges) - 2
                    else values[endpoints] < upper
                )
            )
            conditions.append(
                Condition(
                    f"{family}_q{i + 1}",
                    family,
                    mask,
                    valid,
                    f"{family} empirical within-eligible quantile: [{lower:.9g},{upper:.9g}]; derivative={derivative_definition}, nearpatch<=100px",
                )
            )
    positive = wheel[finite & (wheel > 1e-8)]
    if len(positive):
        threshold = 0.05
        raw = finite & (wheel > threshold)
        # Fill <0.2 s pauses; reject <0.4 s bouts. Never fill source gaps.
        moving = (
            binary_opening(binary_closing(raw, structure=np.ones(10)), structure=np.ones(20))
            & finite
        )
        transitions = np.diff(np.r_[False, moving, False].astype(np.int8))
        for name, indices in [
            ("onset", np.flatnonzero(transitions == 1)),
            ("offset", np.flatnonzero(transitions == -1)),
        ]:
            event = np.zeros(len(behavior), dtype=bool)
            indices = indices[indices < len(event)]
            event[indices[centered_support(finite, 50)[indices]]] = True
            mask_full = maximum_filter1d(event.astype(np.uint8), size=51) > 0
            valid = (complete_support(finite, 20) & centered_support(finite, 25))[endpoints]
            conditions.append(
                Condition(
                    f"wheel_{name}_pm0.5s",
                    "wheel_transition",
                    mask_full[endpoints] & valid,
                    valid,
                    f"+-0.5s of wheel bout {name}; |wheel_speed|>{threshold:.6g}; fill pauses<0.2s, minimum duration0.4s",
                )
            )
    return conditions


@jaxtyped(typechecker=beartype)
def binary_scores(
    active: Bool[np.ndarray, "n"], condition: Bool[np.ndarray, "n"], valid: Bool[np.ndarray, "n"]
) -> dict:
    """Compute paper selectivity and observed-prior precision with safe degeneracy."""
    positive, negative = condition & valid, ~condition & valid
    tp, fp = int((active & positive).sum()), int((active & negative).sum())
    np_, nn = int(positive.sum()), int(negative.sum())
    tpr, fpr = tp / np_ if np_ else np.nan, fp / nn if nn else np.nan
    return dict(
        n_condition=np_,
        n_outside=nn,
        active_condition=tp,
        active_outside=fp,
        tpr=tpr,
        fpr=fpr,
        selectivity=tpr / (tpr + fpr) if tpr + fpr > 0 else np.nan,
        precision=tp / (tp + fp) if tp + fp else np.nan,
    )


@jaxtyped(typechecker=beartype)
def continuous_auc(values: Float[np.ndarray, "n"], positive: Bool[np.ndarray, "n"]) -> float:
    """Exact Mann–Whitney AUROC, including tied native sparse zeros."""
    npos, nneg = int(positive.sum()), int((~positive).sum())
    if not npos or not nneg:
        return float("nan")
    ranks = rankdata(values)
    return float((ranks[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg))


@beartype
def screen(activations: np.ndarray, conditions: list[Condition]) -> pd.DataFrame:
    """Calculate exact native-threshold statistics for all latent/condition pairs."""
    rows = []
    totals = np.zeros((2 * len(conditions), activations.shape[1]), dtype=np.int64)
    for start in range(0, len(activations), 65536):
        stop = min(start + 65536, len(activations))
        masks = np.stack(
            [
                m
                for c in conditions
                for m in (
                    c.mask[start:stop] & c.valid[start:stop],
                    c.valid[start:stop] & ~c.mask[start:stop],
                )
            ]
        ).astype(np.float32)
        # Each dot product sums <=65536 exact 0/1 values, within float32's exact
        # integer range. Accumulate chunk totals in int64 across the recording.
        totals += (masks @ (activations[start:stop] > 0).astype(np.float32)).astype(np.int64)
    for ci, condition in enumerate(conditions):
        np_, nn = (
            int((condition.mask & condition.valid).sum()),
            int((condition.valid & ~condition.mask).sum()),
        )
        for latent in range(activations.shape[1]):
            tp, fp = map(int, totals[2 * ci : 2 * ci + 2, latent])
            tpr, fpr = tp / np_ if np_ else np.nan, fp / nn if nn else np.nan
            rows.append(
                dict(
                    latent=latent,
                    condition=condition.name,
                    family=condition.family,
                    n_condition=np_,
                    n_outside=nn,
                    active_condition=tp,
                    active_outside=fp,
                    tpr=tpr,
                    fpr=fpr,
                    selectivity=tpr / (tpr + fpr) if tpr + fpr > 0 else np.nan,
                    precision=tp / (tp + fp) if tp + fp else np.nan,
                    balanced_rate_difference=tpr - fpr,
                )
            )
    return pd.DataFrame(rows).sort_values("selectivity", ascending=False)


@beartype
def matched_control(
    active: np.ndarray, condition: Condition, behavior: pd.DataFrame, endpoints: np.ndarray
) -> dict:
    """Direct-standardize native activity inside/outside behavior within shared strata."""
    fields = ["x", "y"] + ([] if condition.family == "speed" else ["speed"])
    valid = condition.valid.copy()
    strata = np.zeros(len(active), dtype=np.int64)
    for field in fields:
        values = behavior[field].to_numpy(dtype=float)[endpoints]
        valid &= np.isfinite(values)
        edges = np.unique(np.nanquantile(values, [0, 0.25, 0.5, 0.75, 1]))
        strata = strata * 4 + np.clip(np.searchsorted(edges[1:-1], values, side="right"), 0, 3)
    pos, neg = valid & condition.mask, valid & ~condition.mask
    np_ = np.bincount(strata[pos], minlength=64)
    nn = np.bincount(strata[neg], minlength=64)
    tp = np.bincount(strata[pos & active], minlength=64)
    fp = np.bincount(strata[neg & active], minlength=64)
    overlap = (np_ >= 25) & (nn >= 25)
    if not overlap.any():
        return dict(
            matched_selectivity=np.nan, matched_condition_fraction=0.0, matched_fields=fields
        )
    weights = np_[overlap] / np_[overlap].sum()
    tpr = float(weights @ (tp[overlap] / np_[overlap]))
    fpr = float(weights @ (fp[overlap] / nn[overlap]))
    return dict(
        matched_selectivity=tpr / (tpr + fpr) if tpr + fpr > 0 else np.nan,
        matched_tpr=tpr,
        matched_fpr=fpr,
        matched_condition_fraction=float(np_[overlap].sum() / max(1, pos.sum())),
        matched_fields=fields,
    )


@beartype
def disjoint_peaks(
    values: np.ndarray, endpoints: np.ndarray, count: int = 12, separation: int = 500
) -> np.ndarray:
    """Select high-activation windows separated by at least ten seconds."""
    candidates = np.flatnonzero(values > 0)
    candidates = candidates[np.argsort(values[candidates])[::-1]]
    chosen = []
    for i in candidates:
        if all(abs(int(endpoints[i]) - int(endpoints[j])) >= separation for j in chosen):
            chosen.append(int(i))
            if len(chosen) == count:
                break
    return np.array(chosen, dtype=np.int64)


@beartype
def make_figures(data_path: Path, run_path: Path, selected: pd.DataFrame) -> None:
    """Recreate contextual figures from saved data and candidate metadata."""
    behavior = pd.read_parquet(data_path / "behavior.parquet")
    environment_path = data_path / "environment_context.parquet"
    if environment_path.exists():
        environment = pd.read_parquet(environment_path)
        if not np.array_equal(environment.bin_index, np.arange(len(behavior))):
            raise ValueError("Environment sidecar bin indices do not match behavior")
        behavior["maintenance"] = environment.maintenance.to_numpy(dtype=bool)
    endpoints = np.load(run_path / "endpoint_bins.npy")
    activations = np.load(run_path / "activations.npy", mmap_mode="r")
    counts = np.load(data_path / "counts.npy", mmap_mode="r")
    neural_valid = np.load(data_path / "neural_valid.npy", mmap_mode="r")
    decoder = np.load(run_path / "decoder_kernels.npy")
    pellet_path = data_path / "pellet_times.csv"
    pellet_seconds = (
        pd.read_csv(pellet_path).relative_seconds.to_numpy()
        if pellet_path.exists()
        else np.empty(0)
    )
    out = run_path / "analysis"
    out.mkdir(exist_ok=True)
    for row in selected.itertuples():
        latent = int(row.latent)
        values = np.asarray(activations[:, latent])
        peaks = disjoint_peaks(values, endpoints)
        peak_table = pd.DataFrame(
            {
                "endpoint_bin": endpoints[peaks],
                "seconds": endpoints[peaks] * 0.02,
                "activation": values[peaks],
            }
        )
        peak_table = pd.concat(
            [peak_table, behavior.iloc[endpoints[peaks]].reset_index(drop=True)], axis=1
        )
        if len(pellet_seconds):
            deltas = endpoints[peaks, None] * 0.02 - pellet_seconds[None, :]
            peak_table["seconds_after_nearest_pellet_trigger"] = deltas[
                np.arange(len(peaks)), np.argmin(np.abs(deltas), axis=1)
            ]
        peak_table.to_csv(out / f"latent{latent}_top_disjoint.csv", index=False)
        fig, axs = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
        fig.suptitle(f"Latent {latent}: {row.condition} | full-recording exploratory association")
        sample = np.arange(0, len(endpoints), 50)
        b = behavior.iloc[endpoints[sample]]
        camera_column = (
            "camera_analysis_valid" if "camera_analysis_valid" in behavior else "camera_valid"
        )
        eligible_scatter = b[camera_column].to_numpy(dtype=bool)
        axs[0, 0].scatter(
            b.x[eligible_scatter],
            b.y[eligible_scatter],
            c=values[sample][eligible_scatter],
            s=2,
            cmap="magma",
            rasterized=True,
        )
        axs[0, 0].set(
            title="1-second sampled spatial activation", xlabel="CameraTop x", ylabel="CameraTop y"
        )
        second_maximum = np.zeros(int(endpoints[-1] // 50) + 1)
        np.maximum.at(second_maximum, endpoints // 50, values)
        axs[0, 1].plot(np.arange(len(second_maximum)) / 3600, second_maximum, lw=0.3)
        if len(pellet_seconds):
            axs[0, 1].plot(
                pellet_seconds / 3600,
                np.zeros(len(pellet_seconds)),
                "|",
                color="red",
                ms=3,
                label="Pellet triggers",
            )
            axs[0, 1].legend(fontsize=7)
        axs[0, 1].set(title="1-second maximum activation / time", xlabel="Hours from strict start")
        axs[0, 2].imshow(decoder[latent], aspect="auto", origin="lower", cmap="RdBu_r")
        axs[0, 2].set(
            title="Decoder kernel",
            xlabel="Kernel index (see trainer convention)",
            ylabel="KS4 cluster index",
        )
        offsets = np.arange(-100, 101)
        windows = endpoints[peaks, None] + offsets
        good = (windows[:, 0] >= 0) & (windows[:, -1] < len(behavior))
        windows = windows[good]
        windows = windows[np.asarray(neural_valid[windows]).all(axis=1)]
        for ax, field in zip(axs[1], ("speed", "patch_distance", "wheel_speed")):
            if len(windows):
                traces = behavior[field].to_numpy(dtype=float)[windows]
                validity_field = "wheel_valid" if field.startswith("wheel") else camera_column
                traces[~behavior[validity_field].to_numpy(dtype=bool)[windows]] = np.nan
                ax.plot(offsets * 0.02, traces.T, alpha=0.25, lw=0.6)
                ax.plot(offsets * 0.02, np.nanmedian(traces, axis=0), color="black", lw=2)
            ax.axvline(0, color="red", lw=0.5)
            ax.set(title=f"Top separated episodes: {field}", xlabel="Seconds relative to peak")
        if len(windows):
            axs[2, 0].imshow(
                np.mean(counts[windows], axis=0).T,
                aspect="auto",
                origin="lower",
                extent=[-2, 2, 0, counts.shape[1]],
            )
            axs[2, 0].set(
                title="Mean neural counts around separated peaks",
                xlabel="Seconds",
                ylabel="KS4 cluster index",
            )
        for ax, field in zip(axs[2, 1:], ("area", "angular_velocity")):
            vals = behavior[field].to_numpy(dtype=float)[endpoints]
            valid = np.isfinite(vals) & behavior[camera_column].to_numpy(dtype=bool)[endpoints]
            if valid.any():
                edges = np.unique(np.quantile(vals[valid], np.linspace(0, 1, 21)))
                idx = np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, len(edges) - 2)
                n = np.bincount(idx[valid], minlength=len(edges) - 1)
                total = np.bincount(idx[valid], weights=values[valid], minlength=len(edges) - 1)
                ax.plot((edges[:-1] + edges[1:]) / 2, total / np.maximum(n, 1), ".-")
            ax.set(title=f"Mean native amplitude by {field}", xlabel=field)
        fig.savefig(out / f"latent{latent}_context.png", dpi=140)
        plt.close(fig)


@beartype
def analyze(data_path: Path, run_path: Path) -> None:
    """Screen all latents, audit shortlisted associations and save reproducible figures."""
    output = run_path / "analysis"
    output.mkdir(exist_ok=True)
    behavior = pd.read_parquet(data_path / "behavior.parquet")
    endpoints = np.load(run_path / "endpoint_bins.npy")
    activations = np.load(run_path / "activations.npy", mmap_mode="r")
    conditions = build_conditions(behavior, endpoints)
    lookup = {c.name: c for c in conditions}
    environment_path = data_path / "environment_context.parquet"
    maintenance = np.zeros(len(endpoints), dtype=bool)
    if environment_path.exists():
        environment = pd.read_parquet(environment_path)
        if not np.array_equal(environment.bin_index, np.arange(len(behavior))):
            raise ValueError("Environment sidecar bin indices do not match behavior")
        maintenance = environment.maintenance.to_numpy(dtype=bool)[endpoints]
    metrics_path = run_path / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    protocol = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Full-recording descriptive discovery and evaluation; no held-out decoding claims",
        "neural_population": "All 90 KS4 clusters: 23 good, 67 MUA; anatomical region unknown",
        "observation": "One causally valid 20-bin neural window, assigned once to its endpoint",
        "activity_rule": "Native calibrated sparse activation > 0",
        "tracking_artifacts": "Primary camera comparisons exclude tracking_suspect bins when provided, plus invalid camera support across the causal20bin window; source flag thresholds are documented in common data QC",
        "maintenance": "Primary full-recording comparisons retain maintenance; separately report native activation fraction during maintenance and Sel/AUROC sensitivity excluding maintenance from independent environment sidecar",
        "native_thresholds": metrics.get("thresholds", {}),
        "auroc_score": "Native calibrated sparse continuous amplitude, including tied zeros",
        "n_endpoints": len(endpoints),
        "dictionary_width": activations.shape[1],
        "searched_conditions": len(conditions),
        "searched_pairs": len(conditions) * activations.shape[1],
        "behavior_bouts": "Positive condition bins grouped until a >=1 s gap or neural invalid gap; not independent recordings",
        "top_episode_rule": "Twelve maximum-amplitude episodes, peaks separated by >=10 s",
        "matching": "Direct standardization across joint x/y/speed quartile strata, >=25 positive and negative bins; speed excluded when conditioning on speed",
        "candidate_selection": "Native Sel screen with >=500 bins per class, >=50 active-positive bins, TPR>=0.02; audit union of top60Sel, top4Sel per family, top30(TPR-FPR), top4(TPR-FPR) per family using exact native AUROC; five distinct-latent family-diverse foraging seeds plus up tofive broad seeds for manual inspection",
        "search_caveat": "Empirical cut points and candidate selection use the evaluation recording; multiple comparisons are exploratory",
    }
    (output / "analysis_protocol.json").write_text(json.dumps(protocol, indent=2))
    (output / "condition_definitions.json").write_text(
        json.dumps({c.name: c.definition for c in conditions}, indent=2)
    )
    scores = screen(activations, conditions)
    scores.to_csv(output / "all_native_selectivity.csv", index=False)
    supported = scores.query(
        "n_condition>=500 and n_outside>=500 and active_condition>=50 and tpr>=0.02"
    )
    balanced = supported.sort_values("balanced_rate_difference", ascending=False)
    shortlist = pd.concat(
        [
            supported.head(60),
            supported.groupby("family", sort=False).head(4),
            balanced.head(30),
            balanced.groupby("family", sort=False).head(4),
        ]
    ).drop_duplicates(["latent", "condition"])
    details = []
    for row in shortlist.itertuples():
        condition = lookup[row.condition]
        values = np.asarray(activations[:, row.latent])
        active = values > 0
        starts = separated_bout_starts(condition.mask & condition.valid, endpoints)
        event_id = np.searchsorted(starts, np.arange(len(endpoints)), side="right") - 1
        covered = np.unique(event_id[active & condition.mask & condition.valid])
        info = row._asdict()
        info.update(
            native_continuous_auroc=continuous_auc(
                values[condition.valid], condition.mask[condition.valid]
            ),
            condition_bouts=len(starts),
            covered_bouts=len(covered),
            contiguous_condition_bouts=len(
                bout_starts(condition.mask & condition.valid, endpoints)
            ),
            event_coverage=len(covered) / len(starts) if len(starts) else np.nan,
            **matched_control(active, condition, behavior, endpoints),
        )
        info["active_camera_invalid_fraction"] = float(
            (active & ~behavior.camera_valid.to_numpy(dtype=bool)[endpoints]).sum()
            / max(1, active.sum())
        )
        info["active_maintenance_fraction"] = float(
            (active & maintenance).sum() / max(1, active.sum())
        )
        outside_maintenance = condition.valid & ~maintenance
        no_maintenance = binary_scores(active, condition.mask, outside_maintenance)
        info["no_maintenance_selectivity"] = no_maintenance["selectivity"]
        info["no_maintenance_native_auroc"] = continuous_auc(
            values[outside_maintenance], condition.mask[outside_maintenance]
        )
        if "tracking_suspect" in behavior:
            info["active_tracking_suspect_fraction"] = float(
                (active & behavior.tracking_suspect.to_numpy(dtype=bool)[endpoints]).sum()
                / max(1, active.sum())
            )
        for block in range(6):
            valid = (
                condition.valid
                & (endpoints >= block * 540000)
                & (endpoints < (block + 1) * 540000)
            )
            block_scores = binary_scores(active, condition.mask, valid)
            info[f"block{block}_selectivity"] = block_scores["selectivity"]
            info[f"block{block}_active_condition"] = block_scores["active_condition"]
        details.append(info)
    detailed = pd.DataFrame(details)
    if detailed.empty:
        (output / "NO_SUPPORTED_CANDIDATES.json").write_text(
            json.dumps({"message": "No pairs passed support filters; do not force findings."})
        )
        return
    detailed = detailed.sort_values(["native_continuous_auroc", "selectivity"], ascending=False)
    detailed.to_csv(output / "candidate_audit.csv", index=False)
    foraging = detailed[
        detailed.family.isin(
            [
                "nearpatch_wheel",
                "wheel_transition",
                "patch_heading",
                "patch_proximity",
                "patch_motion",
                "nearpatch_area",
                "wheel_activity",
            ]
        )
    ]
    diverse = foraging.drop_duplicates("latent").drop_duplicates("family").head(5)
    selected = pd.concat([diverse, foraging, detailed]).drop_duplicates("latent").head(5)
    selected.to_csv(output / "selected_candidates.csv", index=False)
    broad = detailed.drop_duplicates("latent").drop_duplicates("family").head(5)
    exploration = pd.concat([selected, broad]).drop_duplicates("latent")
    exploration.to_csv(output / "exploration_candidates.csv", index=False)
    make_figures(data_path, run_path, exploration)
    shutil.copy2(__file__, output / "shared_analysis_snapshot.py")
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Strict 18 h Aeon descriptive latent exploration\n",
                    "All 90 KS4 clusters (23 good, 67 MUA). Full-data training, discovery and evaluation; multiple-search exploratory associations, no held-out decoding claims. Native calibrated sparse amplitudes are used for AUROC; activity means >0. Candidate selection requires independent scientific inspection.\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "from pathlib import Path\nimport importlib.util\nimport pandas as pd\n",
                    f"DATA=Path({str(data_path)!r}); RUN=Path({str(run_path)!r})\n",
                    "spec=importlib.util.spec_from_file_location('aeon_analysis', RUN/'analysis/shared_analysis_snapshot.py')\nimport sys\nm=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m)\n",
                    "selected=pd.read_csv(RUN/'analysis/exploration_candidates.csv')\nm.make_figures(DATA,RUN,selected)\nselected\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "from IPython.display import Image, display\n",
                    "for latent in selected.latent:\n    display(Image(filename=str(RUN/f'analysis/latent{latent}_context.png')))\n",
                ],
            },
        ],
    }
    (output / "analysis.ipynb").write_text(json.dumps(notebook, indent=2))
    print(
        json.dumps(
            {
                "run": str(run_path),
                "pairs_screened": len(scores),
                "pairs_audited": len(detailed),
                "candidates": selected[
                    ["latent", "condition", "selectivity", "native_continuous_auroc"]
                ].to_dict("records"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.data, args.run)
