"""Part A: features prespecified from behavior variables alone.

This file was written and committed before any model was trained on or inspected
for session B. No definition refers to a latent of any method. Thresholds are
either fixed physical values or quantiles of the discovery part's behavior.
`PART_A_FEATURES.md` gives the same definitions in words.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from beartype import beartype

from experiments.rebuttal.change03_discovery.data import MOVING, WHEEL_ACTIVE

NEAR_PATCH = 100.0  # px
NEST_RADIUS = 100.0  # px
NEST_GRID = 25.0  # px
STATIONARY = 5.0  # px/s
FAST = 100.0  # px/s
ACCELERATION = 200.0  # px/s^2
WHEEL_IDLE = 0.02  # turns/s
TRANSITION = 0.5  # s after a wheel bout starts or ends
POST_PELLET = 2.0  # s
PRE_PELLET = 1.0  # s


@dataclass
class Feature:
    """A behavior-defined condition and the windows it is compared within."""

    name: str
    family: str
    label: str
    definition: str
    population: str
    mask: np.ndarray
    valid: np.ndarray


@beartype
def modal_position(x: np.ndarray, y: np.ndarray, grid: float = NEST_GRID) -> tuple[float, float]:
    """Centre of the most occupied `grid`-px cell."""
    cells, counts = np.unique(np.c_[np.floor(x / grid), np.floor(y / grid)], axis=0,
                              return_counts=True)
    cx, cy = cells[np.argmax(counts)]
    return float((cx + 0.5) * grid), float((cy + 0.5) * grid)


@beartype
def build_features(variables: dict, support: dict, discovery: np.ndarray) -> list[Feature]:
    """Build the 23 prespecified features over all windows.

    Args:
        variables: endpoint variables from `data.endpoint_variables`.
        support: 20-bin support masks from `data.endpoint_variables`.
        discovery: windows of the discovery part; quantile thresholds and the modal
            position use only these windows.
    """
    v = variables
    camera, wheel = support["camera"], support["wheel"]
    both = camera & wheel
    features: list[Feature] = []

    def add(name, family, label, definition, population, mask, valid):
        features.append(Feature(name, family, label, definition, population,
                                np.asarray(mask & valid), np.asarray(valid)))

    # Place.
    reference = discovery & camera
    nest_x, nest_y = modal_position(v["x"][reference], v["y"][reference])
    nest_distance = np.hypot(v["x"] - nest_x, v["y"] - nest_y)
    near_patch = v["patch_distance"] <= NEAR_PATCH
    at_nest = nest_distance <= NEST_RADIUS
    tracked = "windows with valid tracking"
    add("near_patch", "place", "near the patch", f"patch distance <= {NEAR_PATCH:g} px",
        tracked, near_patch, camera)
    add("at_modal_location", "place", "at the most occupied location",
        f"within {NEST_RADIUS:g} px of the most occupied {NEST_GRID:g}-px cell of the "
        f"discovery part ({nest_x:g}, {nest_y:g})", tracked, at_nest, camera)
    add("open_arena", "place", "elsewhere in the arena",
        "neither near the patch nor at the most occupied location", tracked,
        ~near_patch & ~at_nest, camera)
    # Speed.
    add("stationary", "speed", "stationary", f"speed < {STATIONARY:g} px/s", tracked,
        v["speed"] < STATIONARY, camera)
    add("moving", "speed", "moving", f"speed >= {MOVING:g} px/s", tracked,
        v["speed"] >= MOVING, camera)
    add("fast", "speed", "fast locomotion", f"speed >= {FAST:g} px/s", tracked,
        v["speed"] >= FAST, camera)
    # Acceleration.
    finite_acceleration = camera & np.isfinite(v["acceleration"])
    add("accelerating", "acceleration", "accelerating",
        f"acceleration >= {ACCELERATION:g} px/s^2", tracked,
        v["acceleration"] >= ACCELERATION, finite_acceleration)
    add("decelerating", "acceleration", "decelerating",
        f"acceleration <= -{ACCELERATION:g} px/s^2", tracked,
        v["acceleration"] <= -ACCELERATION, finite_acceleration)
    # Heading, compared among moving windows only.
    moving = camera & (v["speed"] >= MOVING) & np.isfinite(v["heading"])
    moving_text = f"windows with valid tracking and speed >= {MOVING:g} px/s"
    heading = v["heading"]
    for name, label, mask in [
        ("heading_right", "moving right (+x)", np.abs(heading) < 45),
        ("heading_down", "moving down (+y)", (heading >= 45) & (heading < 135)),
        ("heading_left", "moving left (-x)", np.abs(heading) >= 135),
        ("heading_up", "moving up (-y)", (heading >= -135) & (heading < -45)),
    ]:
        add(name, "heading", label, "movement direction within 45 deg of "
            + label.split("(")[1].rstrip(")"), moving_text, mask, moving)
    # Movement relative to the patch, compared among moving windows only.
    error = np.abs(v["heading_error_to_patch"])
    moving_patch = moving & np.isfinite(error)
    add("toward_patch", "patch-relative movement", "moving toward the patch",
        "movement direction within 45 deg of the direction to the patch", moving_text,
        error < 45, moving_patch)
    add("away_from_patch", "patch-relative movement", "moving away from the patch",
        "movement direction at least 135 deg from the direction to the patch",
        moving_text, error >= 135, moving_patch)
    # Body area.
    low, high = np.quantile(v["area"][reference], [0.2, 0.8])
    add("small_area", "body area", "small body area",
        f"area <= lowest-quintile edge of the discovery part ({low:g} px^2)", tracked,
        v["area"] <= low, camera)
    add("large_area", "body area", "large body area",
        f"area >= highest-quintile edge of the discovery part ({high:g} px^2)", tracked,
        v["area"] >= high, camera)
    # Wheel.
    wheeled = "windows with valid wheel samples"
    active = v["wheel_speed"] > WHEEL_ACTIVE
    add("wheel_active", "wheel", "wheel running", f"wheel speed > {WHEEL_ACTIVE:g} turns/s",
        wheeled, active, wheel)
    median = float(np.median(v["wheel_speed"][discovery & wheel & active])) \
        if (discovery & wheel & active).any() else float("nan")
    add("wheel_fast", "wheel", "fast wheel running",
        f"wheel speed >= median wheel speed of wheel-running windows in the discovery "
        f"part ({median:.3g} turns/s)", "wheel-running windows",
        v["wheel_speed"] >= median, wheel & active)
    add("wheel_onset", "wheel", "wheel-bout onset",
        f"0 to {TRANSITION:g} s after a wheel bout starts", wheeled,
        v["_time_since_wheel_onset"] < TRANSITION, wheel)
    add("wheel_offset", "wheel", "wheel-bout offset",
        f"0 to {TRANSITION:g} s after a wheel bout ends", wheeled,
        v["_time_since_wheel_offset"] < TRANSITION, wheel)
    add("patch_wheel_idle", "wheel", "at the patch, wheel idle",
        f"patch distance <= {NEAR_PATCH:g} px and wheel speed <= {WHEEL_IDLE:g} turns/s",
        "windows with valid tracking and wheel samples",
        near_patch & (v["wheel_speed"] <= WHEEL_IDLE), both)
    # Food delivery.
    everything = support["all"]
    add("post_pellet", "food delivery", "after a pellet trigger",
        f"0 to {POST_PELLET:g} s after a pellet trigger", "all windows",
        v["time_since_pellet"] < POST_PELLET, everything)
    add("pre_pellet", "food delivery", "before a pellet trigger",
        f"0 to {PRE_PELLET:g} s before a pellet trigger", "all windows",
        (v["time_to_pellet"] > 0) & (v["time_to_pellet"] <= PRE_PELLET), everything)
    return features
