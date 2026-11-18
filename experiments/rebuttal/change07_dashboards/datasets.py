"""Saved activations and behavior of the Section 4.2 and 4.3 models, as dashboard datasets.

Nothing is retrained or rescored: both loaders only read frozen exports. Churchland
behavior is git-ignored, so it is read from the main checkout unless a path is given.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from beartype import beartype

from experiments.rebuttal.change07_dashboards.dashboard import (
    Dataset,
    Map,
    Variable,
    as_flags,
    segment_ids,
    trailing_slope,
)

AEON = Path("/ceph/aeon/aeon/nldisco/aeon/sweeps/20260921_18h_s20")
CHURCHLAND = Path("/ceph/aeon/aeon/nldisco/churchland/sweeps/20260914_s10_endpoint_v1")
CHURCHLAND_RUN = CHURCHLAND / "width192_checks/nldisco_seeds_1_to_5/d192_k16_l2_seed1"
CHURCHLAND_DATA = Path("/nfs/nhome/live/jbhagat/nldisco/experiments/churchland/data/nitschke_20090812")
CM_PER_TURN = 8 * np.pi  # 4 cm wheel radius
ARENA_CENTER = (731., 561.)  # px, as in the paper's outward-direction feature


@beartype
def load_aeon(sweep: Path = AEON, run: str = "d192/k12_seed0") -> Dataset:
    """Aeon 18 h recording: wheel, centroid and patch variables at each window's last bin."""
    acts = np.load(sweep / run / "activations.npy")
    bins = np.load(sweep / run / "endpoint_bins.npy")
    args = json.loads((sweep / run / "config.json").read_text())["args"]
    frame = pd.read_parquet(sweep / "data/behavior.parquet", columns=[
        "x", "y", "area", "speed", "wheel_speed", "patch_distance", "heading_error_to_patch",
        "heading", "camera_analysis_valid", "wheel_valid"]).iloc[bins]
    camera = frame.camera_analysis_valid.to_numpy()
    column = lambda name: np.where(camera, frame[name].to_numpy(), np.nan)  # noqa: E731
    wheel = np.where(frame.wheel_valid, frame.wheel_speed * CM_PER_TURN, np.nan)
    x, y, speed, distance = column("x"), column("y"), column("speed"), column("patch_distance")
    area = column("area")
    patch = (891.75, 851.5)  # behavior_summary.json: patch_center_px
    # Movement heading relative to the outward radial direction from the arena center.
    outward = np.degrees(column("heading") - np.arctan2(y - ARENA_CENTER[1], x - ARENA_CENTER[0]))
    outward = np.mod(outward + 180, 360) - 180
    pellets = pd.read_csv(sweep / "data/pellet_times.csv").relative_seconds.to_numpy()
    pellet_rows = np.searchsorted(bins, np.floor(pellets / .02).astype(np.int64))
    hours = bins * .02 / 3600
    return Dataset(
        title=f"Mouse foraging (Aeon), TW-SED D={args['d']}, k={args['k']}",
        activations=acts, bin_seconds=.02, segments=segment_ids(bins),
        variables=(
            Variable("Wheel speed (cm/s)", wheel, np.array(
                [0, .1, .2, .3, .5, .75, 1.5, 3, 6, 12, 25, 80]), log=True),
            Variable("Centroid speed (px/s)", speed,
                     np.r_[0, np.geomspace(.25, 1000, 13)], log=True),
            Variable("Distance to patch (px)", distance, np.linspace(0, 840, 15)),
            Variable(r"Segmented area (px$^2$)", area, np.linspace(100, 1000, 13)),
            Variable("Heading rel. to\npatch (deg)",
                     np.degrees(column("heading_error_to_patch")), np.linspace(-180, 180, 13)),
            Variable("Heading rel. to\noutward (deg)", outward, np.linspace(-180, 180, 13)),
        ),
        maps=(
            Map("Position in arena", x, y, np.arange(200, 1251, 25.), np.arange(40, 1101, 25.),
                "x (px)", "y (px)", marker=(*patch, "patch"), invert_y=True),
            Map("Position near patch", x, y, np.arange(patch[0] - 150, patch[0] + 151, 10.),
                np.arange(patch[1] - 150, patch[1] + 151, 10.), "x (px)", "",
                marker=(*patch, "patch"), invert_y=True),
        ),
        timeline=Variable("Time in recording (h)", hours, np.linspace(0, 18, 217)),
        traces={"Wheel\n(cm/s)": wheel, "Speed\n(px/s)": speed, "To patch\n(px)": distance,
                "Area\n" r"(px$^2$)": area},
        events={"pellet trigger": as_flags(len(bins), pellet_rows)},
        episode_labels=np.char.add("hour ", np.char.mod("%.2f", hours)),
        half_width=500,
    )


@beartype
def load_churchland(run: Path = CHURCHLAND_RUN, data: Path = CHURCHLAND_DATA,
                    index: Path = CHURCHLAND / "shared/index.npz") -> Dataset:
    """Macaque reaching: kinematics, target and movement timing on development trials."""
    with np.load(index) as file:
        anchors = file["anchor_index"]
    with np.load(data / "metadata.npz") as file:
        meta = {name: file[name] for name in file.files}
    rows = np.flatnonzero(meta["split"][anchors] != 2)  # decoder-test trials stay unopened
    anchors = anchors[rows]
    acts = np.load(run / "activations_level_192.npy")[rows]
    trial = meta["trial_id"][anchors]
    speed, vx, vy = (meta[name][anchors] for name in ("speed", "vel_x", "vel_y"))
    slope = trailing_slope(meta["speed"][anchors[:, None] + np.arange(-4, 1)], .05)
    direction = np.where(speed >= 50, np.mod(np.degrees(np.arctan2(vy, vx)), 360), np.nan)
    with (data / "trials.csv").open() as stream:
        table = {int(row["trial_id"]): row for row in csv.DictReader(stream)}
    onset, end = (np.array([float(table[int(t)][name]) for t in trial])
                  for name in ("movement_onset", "movement_end"))
    since_onset, since_end = meta["timestamps"][anchors] - onset, meta["timestamps"][anchors] - end
    targets = np.column_stack([meta["target_x"][anchors], meta["target_y"][anchors]])
    points, codes = np.unique(targets, axis=0, return_inverse=True)
    labels = np.array([f"trial {t}\ntarget ({a:.0f}, {b:.0f})"
                       for t, (a, b) in zip(trial, targets)])
    edges = np.arange(-800, 801, 50.)
    return Dataset(
        title="Macaque reaching (Churchland), TW-SED D=192, k=16",
        activations=acts, bin_seconds=.05, segments=segment_ids(anchors, trial),
        variables=(
            Variable("Hand speed (mm/s)", speed, np.linspace(0, 1000, 21)),
            Variable("Speed slope over\n" r"last 250 ms (mm/s$^2$)", slope, np.linspace(-3000, 3000, 25)),
            Variable("Velocity direction (deg)", direction, np.arange(0, 361, 15.)),
            Variable(r"$v_x$ (mm/s)", vx, edges),
            Variable(r"$v_y$ (mm/s)", vy, edges),
        ),
        maps=(
            Map("Hand velocity", vx, vy, edges, edges, r"$v_x$ (mm/s)", r"$v_y$ (mm/s)"),
            Map("Target location", codes.ravel().astype(np.float64), np.zeros(len(rows)), edges,
                edges, "x (mm)", "y (mm)", points=points),
        ),
        timeline=Variable("Time from movement onset (s)", since_onset,
                          np.arange(-1.525, 2.03, .05)),
        traces={"Speed\n(mm/s)": speed, r"$v_x$" "\n(mm/s)": vx, r"$v_y$" "\n(mm/s)": vy},
        events={"movement onset": np.abs(since_onset) <= .025,
                "movement end": np.abs(since_end) <= .025},
        episode_labels=labels, half_width=40, min_count=10,
    )


LOADERS = {"aeon": load_aeon, "churchland": load_churchland}
