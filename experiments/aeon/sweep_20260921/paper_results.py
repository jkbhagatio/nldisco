"""Recompute the four current Aeon feature scores and numerical plot inputs.

All analyses are descriptive on the full recording. No model is fitted, latent
threshold changed, or source spike/behavior data modified by this entry point.
"""

# ruff: noqa: F821 -- jaxtyping dimension strings are not Python names.

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from sklearn.metrics import roc_curve

from experiments.aeon.sweep_20260921.shared_analysis import (
    binary_scores,
    complete_support,
    continuous_auc,
    separated_bout_starts,
)

ROOT = Path("/ceph/aeon/aeon/nldisco/aeon/sweeps/20260921_18h_s20")
OUT = ROOT / "paper_revision_wheelcms_20260921"
WHEEL_RADIUS_CM = 4.
WHEEL_CM_PER_TURN = 2 * np.pi * WHEEL_RADIUS_CM
WHEEL_CUTOFF_CM_S = .75
WHEEL_CUTOFF = WHEEL_CUTOFF_CM_S / WHEEL_CM_PER_TURN
WHEEL_CENTERS = np.array([0., .002, .005, .01, .02, .03, .1, .25, .5, 1.])
WHEEL_TUNING_MAX = 1.75  # Retain the previous 1-turn/s bin; omit the old upper-tail dot.
WHEEL_METADATA = Path("/ceph/aeon/aeon/data/raw/AEON2/social-ephys0.1/2024-06-04T10-29-49/Metadata.yml")
WHEEL_RADIUS_SOURCE = "https://aeon.swc.ucl.ac.uk/user/aeon_modules/acquisition/foraging_patch/"
FEATURES = {
    "wheel": (192, 52, "Wheel activity"),
    "area": (256, 107, "Smaller segmented area near patch"),
    "speed": (256, 109, "Highest centroid-speed quintile"),
    "direction": (256, 10, "Fast directional movement"),
}




@jaxtyped(typechecker=beartype)
def binned_probability(
    values: Float[np.ndarray, "n"],
    active: Bool[np.ndarray, "n"],
    valid: Bool[np.ndarray, "n"],
    edges: Float[np.ndarray, "edge"],
    minimum_count: int = 100,
) -> dict:
    """Histogram activity probability; include the last edge, mask sparse cells."""
    total = np.histogram(values[valid], bins=edges)[0]
    positive = np.histogram(values[valid & active], bins=edges)[0]
    probability = np.divide(positive, total, out=np.full(len(total), np.nan), where=total >= minimum_count)
    return dict(edges=edges, total=total, active=positive, probability=probability)


@jaxtyped(typechecker=beartype)
def center_edges(centers: Float[np.ndarray, "n"], maximum: float) -> Float[np.ndarray, "edge"]:
    """Midpoint boundaries for prescribed representative speeds on a nonnegative axis."""
    if len(centers) < 2 or centers[0] != 0 or not np.all(np.diff(centers) > 0):
        raise ValueError("Centers must start at zero and be strictly increasing")
    return np.r_[0., (centers[1:] + centers[:-1]) / 2,
                 max(maximum + .001, centers[-1] + (centers[-1] - centers[-2]) / 2)]


@beartype
def wheel_tuning(values: np.ndarray, active: np.ndarray, valid: np.ndarray) -> dict:
    """Ten speed bins with cm/s coordinates; upper-tail omission affects only tuning."""
    edges = np.r_[0., (WHEEL_CENTERS[1:] + WHEEL_CENTERS[:-1]) / 2, WHEEL_TUNING_MAX]
    plotted = valid & (values < WHEEL_TUNING_MAX)
    result = binned_probability(values, active, plotted, edges, minimum_count=1)
    result.update(centers=WHEEL_CENTERS, centers_cm_s=WHEEL_CENTERS * WHEEL_CM_PER_TURN,
                  edges_cm_s=edges * WHEEL_CM_PER_TURN, low_support=result["total"] < 100,
                  omitted_upper_tail_windows=int((valid & (values >= WHEEL_TUNING_MAX)).sum()))
    return result


@beartype
def save_wheel_tuning(values: np.ndarray, active: np.ndarray, valid: np.ndarray, out: Path) -> None:
    """Save plot inputs and the radius provenance, without altering feature masks."""
    tuning = wheel_tuning(values, active, valid)
    np.savez(out / "tuning_wheel.npz", **tuning)
    metadata = json.loads(WHEEL_METADATA.read_text())
    signed_radius = float(metadata["Devices"]["Patch2"]["Radius"])
    if abs(signed_radius) != WHEEL_RADIUS_CM:
        raise ValueError("Configured radius differs from the plot conversion")
    (out / "wheel_tuning_protocol.json").write_text(json.dumps(dict(
        display_units="cm/s of wheel-rim motion, not mouse translational speed",
        radius_cm=WHEEL_RADIUS_CM, circumference_cm=WHEEL_CM_PER_TURN,
        signed_configured_radius=signed_radius, metadata=str(WHEEL_METADATA),
        metadata_sha256=hashlib.sha256(WHEEL_METADATA.read_bytes()).hexdigest(),
        radius_documentation=WHEEL_RADIUS_SOURCE,
        centers_turns_s=WHEEL_CENTERS.tolist(), centers_cm_s=tuning["centers_cm_s"].tolist(),
        edges_turns_s=tuning["edges"].tolist(), edges_cm_s=tuning["edges_cm_s"].tolist(),
        cutoff_turns_s=WHEEL_CUTOFF, cutoff_cm_s=WHEEL_CUTOFF_CM_S,
        omitted_upper_tail_windows=tuning["omitted_upper_tail_windows"],
        omission="Speeds >=1.75turns/s omitted only from tuning display; all retained in scores/ROC/traceback",
        revision="Replace 2.5turn/s dot with .25turn/s, use exact .75cm/s feature cutoff, remove '(native)' label",
    ), indent=2))


@beartype
def refresh_wheel_tuning(root: Path = ROOT, out: Path = OUT) -> None:
    """Rebin frozen wheel data while preserving all saved feature scores and profiles."""
    speed = pd.read_parquet(root / "data/behavior.parquet", columns=["wheel_speed"]).wheel_speed.to_numpy()
    with np.load(out / "feature_arrays.npz") as arrays:
        save_wheel_tuning(speed[arrays["endpoint_bins"]], arrays["values_wheel"] > 0,
                          arrays["valid_wheel"], out)
    protocol_path = out / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["tuning"] = ("Wheel: ten representative speeds with midpoint interior boundaries, "
                          "4cm radius conversion to cm/s; >=1.75turn/s omitted from tuning only. "
                          "See wheel_tuning_protocol.json. Other feature tuning is unchanged.")
    protocol_path.write_text(json.dumps(protocol, indent=2))


@beartype
def roc_operating_point(values: np.ndarray, labels: np.ndarray, target: float = .18) -> dict:
    """Nearest attainable empirical FPR; highest TPR breaks equal-distance ties.

    This visualization threshold never replaces native z>0 for Sel or traceback.
    """
    fpr, tpr, thresholds = roc_curve(labels, values, drop_intermediate=False)
    distance = np.abs(fpr - target)
    candidates = np.flatnonzero(distance == distance.min())
    index = int(candidates[np.argmax(tpr[candidates])])
    return dict(target_fpr=target, fpr=float(fpr[index]), tpr=float(tpr[index]),
                threshold=float(thresholds[index]), rule="z >= threshold",
                selection="Nearest attainable empirical FPR; highest TPR among ties",
                scope="Display-only operating point; Sel and source traceback retain z>0")


@beartype
def wheel_location_controls(
    behavior: pd.DataFrame, endpoints: np.ndarray, values: np.ndarray,
    condition: np.ndarray, valid: np.ndarray, out: Path,
) -> None:
    """Compare wheel states near the patch and within coarse 25px position cells."""
    camera = behavior.camera_analysis_valid.to_numpy(dtype=bool)
    distance = behavior.patch_distance.to_numpy(dtype=float)
    x, y = behavior.x.to_numpy(dtype=float), behavior.y.to_numpy(dtype=float)
    near = valid & complete_support(camera & np.isfinite(distance) & np.isfinite(x)
                                    & np.isfinite(y) & (distance <= 100))[endpoints]
    rows = [dict(comparison="within100px_patch", **score(values, condition, near))]
    positions = np.c_[np.floor(x[endpoints[near]] / 25), np.floor(y[endpoints[near]] / 25)]
    _, groups = np.unique(positions, axis=0, return_inverse=True)
    z, labels = values[near], condition[near]
    strata = []
    for group in np.unique(groups):
        selected = groups == group
        positive, negative = labels[selected].sum(), (~labels[selected]).sum()
        if positive >= 20 and negative >= 20:
            strata.append(dict(group=int(group), **score(z, labels, selected)))
    frame = pd.DataFrame(strata)
    if len(frame):
        weights = frame.n_condition.to_numpy(dtype=float)
        coverage = float(weights.sum() / labels.sum())
        weights /= weights.sum()
        tpr, fpr = float(weights @ frame.tpr), float(weights @ frame.fpr)
        rows.append(dict(comparison="within100px_patch_xy25px_standardized",
                         n_condition=int(frame.n_condition.sum()), n_outside=int(frame.n_outside.sum()),
                         tpr=tpr, fpr=fpr, selectivity=tpr / (tpr + fpr),
                         auroc=float(weights @ frame.auroc), positive_coverage=coverage,
                         n_strata=len(frame)))
    frame.to_csv(out / "wheel_location_strata.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "wheel_location_controls.csv", index=False)
    np.savez_compressed(out / "wheel_location_masks.npz", endpoint_bins=endpoints, near_patch_valid=near)
    (out / "wheel_location_protocol.json").write_text(json.dumps(dict(
        threshold_turns_s=WHEEL_CUTOFF, near_patch_radius_px=100, position_cell_width_px=25,
        support="All20bins camera-QC/position/wheel-valid and within100px of Patch2; endpoint wheel state",
        strata="At least20positive and20negative windows; positive-count-weighted rates and within-cell AUROC",
        limitation="Controls coarse location only, not body orientation, shape, limb motion, or causal effects",
        maintenance="Retained in primary comparisons"), indent=2))


@beartype
def score(values: np.ndarray, condition: np.ndarray, valid: np.ndarray) -> dict:
    """Native positive activity and positive-oriented exact AUROC including ties."""
    return {
        **binary_scores(values > 0, condition, valid),
        "auroc": continuous_auc(values[valid], condition[valid]),
    }


@beartype
def definitions(behavior: pd.DataFrame, endpoints: np.ndarray) -> tuple[dict, dict]:
    """Reproduce the three original full-recording comparison populations."""
    camera = behavior.camera_analysis_valid.to_numpy(dtype=bool)
    result, descriptions = {}, {}
    wheel = behavior.wheel_speed.to_numpy(dtype=float)
    valid = complete_support(behavior.wheel_valid.to_numpy() & np.isfinite(wheel))[endpoints]
    result["wheel"] = (wheel[endpoints] > WHEEL_CUTOFF, valid)
    descriptions["wheel"] = dict(
        definition=f"Endpoint wheel-rim speed >{WHEEL_CUTOFF_CM_S} cm/s ({WHEEL_CUTOFF} turns/s)",
        comparison="All neural windows with 20-bin finite wheel support; maintenance retained",
    )
    for name, field, eligible, quintile in [
        ("area", "area", camera & (behavior.patch_distance.to_numpy() <= 100), 0),
        ("speed", "speed", camera, 4),
    ]:
        source = behavior[field].to_numpy(dtype=float)
        valid = complete_support(eligible & np.isfinite(source))[endpoints]
        values = source[endpoints]
        edges = np.quantile(values[valid], np.linspace(0, 1, 6))
        mask = (values >= edges[quintile]) & (
            values <= edges[quintile + 1] if quintile == 4 else values < edges[quintile + 1]
        )
        result[name] = (mask, valid)
        descriptions[name] = dict(
            definition=f"{edges[quintile]:.9g} <= {field} "
            f"{'<=' if quintile == 4 else '<'} {edges[quintile + 1]:.9g}",
            comparison=("Camera-QC-valid complete 20-bin support" +
                        (" wholly within 100px of Patch2" if name == "area" else "") +
                        "; maintenance retained"),
            quantile_edges=edges.tolist(),
        )
    x, y, speed = (behavior[field].to_numpy(dtype=float) for field in ('x', 'y', 'speed'))
    heading = behavior.heading.to_numpy(dtype=float)[endpoints]
    valid = complete_support(camera & np.isfinite(x) & np.isfinite(y) & np.isfinite(speed))[endpoints]
    fast = speed[endpoints] >= 100
    error = (heading - np.arctan2(y[endpoints] - 561., x[endpoints] - 731.) + np.pi) % (2*np.pi) - np.pi
    result['direction'] = ((np.abs(error) <= np.pi/4) & fast, valid & fast)
    descriptions['direction'] = dict(definition='Speed >=100px/s, heading within45deg of radially outward from (731,561)',
                                     comparison='Other fast movement, with complete camera/finite position-speed support')
    return result, descriptions


@beartype
def compute(root: Path = ROOT, out: Path = OUT) -> None:
    """Save current feature scores, tuning, ROC curves, and behavior controls."""
    out.mkdir(parents=True, exist_ok=True)
    behavior = pd.read_parquet(root / "data/behavior.parquet")
    endpoints = np.load(root / "d192/k12_seed0/endpoint_bins.npy")
    np.testing.assert_array_equal(endpoints, np.load(root / "d256/k12_seed0/endpoint_bins.npy"))
    conditions, info = definitions(behavior, endpoints)
    neural_valid = np.load(root / "data/neural_valid.npy")
    np.testing.assert_equal(complete_support(neural_valid)[endpoints].all(), True)
    saved = {"endpoint_bins": endpoints}
    rows = []
    for name, (d, latent, title) in FEATURES.items():
        print(f"Scoring and tracing {name}: D{d}/K12/L{latent}", flush=True)
        values = np.array(np.load(root / f"d{d}/k12_seed0/activations.npy", mmap_mode="r")[:, latent])
        active = values > 0
        saved[f"values_{name}"] = values
        if name in conditions:
            condition, valid = conditions[name]
            saved[f"condition_{name}"] = condition
            saved[f"valid_{name}"] = valid
            metrics = score(values, condition, valid)
            rows.append(dict(feature=name, D=d, K=12, latent=latent, **metrics,
                             condition_episodes=len(separated_bout_starts(condition & valid, endpoints))))
            fpr, tpr, _ = roc_curve(condition[valid], values[valid], drop_intermediate=True)
            # Preserve every ROC corner; no approximate AUROC integration used for scoring.
            np.savez_compressed(out / f"roc_{name}.npz", fpr=fpr, tpr=tpr)
            if name == "wheel":
                marker = roc_operating_point(values[valid], condition[valid])
                (out / "wheel_roc_marker.json").write_text(json.dumps(marker, indent=2))
                wheel_location_controls(behavior, endpoints, values, condition, valid, out)
            if name in ("wheel", "area", "speed"):
                field = {"wheel": "wheel_speed", "area": "area", "speed": "speed"}[name]
                x = behavior[field].to_numpy(dtype=float)[endpoints]
                if name == "wheel":
                    save_wheel_tuning(x, active, valid, out)
                else:
                    tuning = binned_probability(x, active, valid, np.array(info[name]["quantile_edges"]))
                    np.savez(out / f"tuning_{name}.npz", **tuning)
            elif name == "direction":
                x = behavior.x.to_numpy()[endpoints] - 731.
                y = behavior.y.to_numpy()[endpoints] - 561.
                heading = behavior.heading.to_numpy()[endpoints]
                error = np.rad2deg((heading - np.arctan2(y, x) + np.pi) % (2*np.pi) - np.pi)
                np.savez(out / "tuning_direction.npz", **binned_probability(
                    error, active, valid, np.linspace(-180., 180., 25)))
    np.savez_compressed(out / "feature_arrays.npz", **saved)
    pd.DataFrame(rows).to_csv(out / "feature_scores.csv", index=False)
    (out / "feature_definitions.json").write_text(json.dumps(info, indent=2))
    protocol = dict(
        interval=["2024-06-06 23:00:00", "2024-06-07 17:00:00"], right_endpoint_exclusive=True,
        indices="zero-based latent indices; original raw KS4 cluster IDs",
        scoring="Native z>0 Sel=TPR/(TPR+FPR); exact continuous AUROC with ties, no sign flipping",
        standardization="Original full-neural-valid-bin mean and population SD; no smoothing",
        inference="Full-data post-discovery descriptive analysis, not held-out or causal inference",
        tuning="Wheel: ten user-requested representative speeds, arithmetic midpoint boundaries; "
        "cm/s from4cm radius; upper tail>=1.75turn/s omitted only from tuning. "
        "See wheel_tuning_protocol.json for bin and conversion provenance. "
        "Show low-support cells(<100 windows) as hollow dots without connecting line. "
        "Other tuning plots mask cells with<100 windows. Probability always uses native z>0.",
        revision="Wheel feature cutoff revised from .05 to .03turn/s, then to exact .75cm/s after plot inspection; "
        "ROC marker selected by requested FPR. Both are post-discovery display/definition revisions.",
        source_ready=json.loads((root / "data/READY.json").read_text()),
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


@beartype
def save_figure(fig: plt.Figure, out: Path, name: str) -> None:
    """Save matching vector and raster paper assets."""
    for extension in ("pdf", "png"):
        fig.savefig(out / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


@beartype
def render_main(out: Path = OUT) -> None:
    """Regenerate Figure 5 from saved tuning, ROC, and score inputs."""
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    scores = pd.read_csv(out / "feature_scores.csv").set_index("feature")
    blue = "#2166ac"
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.3), constrained_layout=True)
    tuning = np.load(out / "tuning_wheel.npz")
    centers = tuning["centers_cm_s"]
    supported = ~tuning["low_support"]
    axes[0].plot(centers, np.where(supported, tuning["probability"], np.nan), "o-", ms=3, color=blue)
    if (~supported).any():
        axes[0].scatter(centers[~supported], tuning["probability"][~supported], s=14,
                        facecolors="none", edgecolors=blue, zorder=3, clip_on=False)
    for index in np.flatnonzero(~supported):
        axes[0].annotate(f"n={tuning['total'][index]}", (centers[index], tuning["probability"][index]),
                         xytext=(-3, 8), textcoords="offset points", ha="right", fontsize=6)
    axes[0].axvline(WHEEL_CUTOFF_CM_S, color=".5", ls="--", lw=.8)
    axes[0].set(xscale="symlog", ylim=(0, 1.04), xlabel="Wheel speed (cm/s)",
                ylabel="P(latent active)", title="Wheel activity")
    axes[0].set_xscale("symlog", linthresh=.01 * WHEEL_CM_PER_TURN)
    axes[0].set_xlim(-.0005 * WHEEL_CM_PER_TURN, float(centers[-1]) * 1.12)
    axes[0].set_xticks([0, .25, 1, 5, 25], ["0", "0.25", "1", "5", "25"])
    roc = np.load(out / "roc_wheel.npz")
    axes[1].plot(roc["fpr"], roc["tpr"], color=blue)
    axes[1].plot([0, 1], [0, 1], color=".75", ls=":")
    row = scores.loc["wheel"]
    decode_path = (Path(__file__).resolve().parents[2] / "outputs"
                   / "decoder_balanced_accuracy_20260923" / "aeon" / "results.json")
    decode_score = json.loads(decode_path.read_text())["test_balanced_accuracy"]
    marker = json.loads((out / "wheel_roc_marker.json").read_text())
    axes[1].scatter(marker["fpr"], marker["tpr"], color=blue, s=20, zorder=3)
    axes[1].text(.98, .08,
                 f"Cov = {marker['tpr']:.3f}\nSpec = {1 - marker['fpr']:.3f}\n"
                 f"Sel = {marker['tpr'] / (marker['tpr'] + marker['fpr']):.3f}\n"
                 f"AUROC = {row.auroc:.3f}\nDecode = {decode_score:.3f}",
                 ha="right", transform=axes[1].transAxes)
    axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate",
                title="Latent ROC", xlim=(0, 1), ylim=(0, 1.04))
    for letter, ax in zip("ab", axes):
        ax.set_title(letter, loc="left", fontsize=10, fontweight="bold")
        ax.set_axisbelow(True)
        ax.grid(axis="both", which="major", color=".6", alpha=.18, lw=.5)
    save_figure(fig, out, "aeon_main")






@beartype
def write_manifest(out: Path = OUT) -> None:
    """Refresh hashes after notebook figure regeneration, excluding this manifest."""
    manifest = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in out.iterdir() if p.is_file() and p.suffix in (".csv", ".json", ".npz", ".pdf")
                and p.name != "artifact_hashes.json"}
    (out / "artifact_hashes.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='Fresh experiment directory for recomputed numerical inputs')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    compute(out=args.output)
    write_manifest(args.output)
