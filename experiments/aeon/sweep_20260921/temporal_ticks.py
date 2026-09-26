"""Native endpoint states for the two wheel-versus-patch temporal panels."""

# ruff: noqa: F821 -- jaxtyping symbolic dimensions are not Python names.

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from beartype import beartype
from jaxtyping import Bool, Int, jaxtyped

from experiments.aeon.sweep_20260921.shared_analysis import complete_support

ROOT = Path("/ceph/aeon/aeon/nldisco/aeon/sweeps/20260921_18h_s20")
SOURCE = ROOT / "paper_revision_wheelcms_20260921"
OUT = Path(__file__).resolve().parents[1] / "outputs/paper"
START = pd.Timestamp("2024-06-06 23:00:00")
TRACKS = (
    ("patch", "At patch", "Centroid within 100 px of Patch2", "#737980"),
    ("wheel", "Wheel active", "Wheel-rim speed >0.75 cm/s", "#d98c38"),
    ("latent", "Latent active", "D192 / K12 · latent 52 · z > 0", "#2166ac"),
)


@jaxtyped(typechecker=beartype)
def full_track(
    endpoints: Int[np.ndarray, "window"], condition: Bool[np.ndarray, "window"],
    valid: Bool[np.ndarray, "window"], n_bins: int,
) -> tuple[Bool[np.ndarray, "time"], Bool[np.ndarray, "time"]]:
    """Scatter endpoint states without treating unobserved windows as inactive."""
    if len(endpoints) and (endpoints[0] < 0 or endpoints[-1] >= n_bins
                           or np.any(np.diff(endpoints) <= 0)):
        raise ValueError("Endpoints must be unique, increasing, and within the recording")
    active, observed = np.zeros(n_bins, dtype=bool), np.zeros(n_bins, dtype=bool)
    active[endpoints] = condition & valid
    observed[endpoints] = valid
    return active, observed


@beartype
def prepare(out: Path) -> dict:
    """Save exact boolean states on the recorded bin-center time axis."""
    behavior = pd.read_parquet(ROOT / "data/behavior.parquet",
                               columns=["time", "patch_distance", "camera_analysis_valid"])
    with np.load(SOURCE / "feature_arrays.npz") as arrays:
        endpoints = arrays["endpoint_bins"]
        camera = behavior.camera_analysis_valid.to_numpy(dtype=bool)
        distance = behavior.patch_distance.to_numpy(dtype=float)
        patch_valid = complete_support(camera & np.isfinite(distance))[endpoints]
        inputs = {
            "patch": (distance[endpoints] <= 100, patch_valid),
            "wheel": (arrays["condition_wheel"], arrays["valid_wheel"]),
            "latent": (arrays["values_wheel"] > 0, np.isfinite(arrays["values_wheel"])),
        }
        states = {"hours": (behavior.time - START).dt.total_seconds().to_numpy() / 3600}
        for name, (condition, valid) in inputs.items():
            states[name], states[f"valid_{name}"] = full_track(endpoints, condition, valid, len(behavior))
    np.savez_compressed(out / "tick_states.npz", **states)
    summary = pd.DataFrame([
        dict(track=name, active_windows=int(states[name].sum()),
             valid_windows=int(states[f"valid_{name}"].sum()),
             invalid_windows=int((~states[f"valid_{name}"]).sum()))
        for name, *_ in TRACKS
    ])
    summary.to_csv(out / "support.csv", index=False)
    protocol = dict(
        interval=[str(START), "2024-06-07 17:00:00"], end_exclusive=True,
        clock="Acquisition-clock timestamps; no timezone conversion", bin_seconds=.02, input_bins=20,
        ticks="One tick at each active window endpoint, not bout onsets; no downsampling or rate smoothing",
        patch="Endpoint centroid within100px of Patch2; all20camera bins QC-valid/finite; neural-supported endpoints",
        wheel="Saved exact >0.75cm/s condition and complete20bin wheel-valid support; all speeds retained",
        latent="Saved native z>0, D192/K12/latent52; all neural-supported windows, independently of behavior validity",
        unavailable="Light gray ticks distinguish missing/invalid support from known inactive baseline",
        source=str(SOURCE), recording_maintenance="Retained, matching primary analysis",
        scope="Two descriptive temporal excerpts; no temporal-stability claim",
        feature_scores_sha256=hashlib.sha256((SOURCE / "feature_scores.csv").read_bytes()).hexdigest(),
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    print(summary.to_string(index=False), flush=True)
    return states


