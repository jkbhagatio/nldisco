"""Prepare trial-contained Churchland counts and behavioral labels.

Run with ``uv run --no-sync python experiments/churchland/scripts/prepare.py``.
The processed source supplies its audited clock repair and channel exclusions;
kinematics and label partitions are rebuilt without its previous label split.
"""

# Ruff interprets jaxtyping's symbolic dimension strings as forward references.
# ruff: noqa: F821

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "experiments/churchland/data/nitschke_20090812"
BIN_WIDTH = 0.05


@beartype
def checksum(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading a source file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@jaxtyped(typechecker=beartype)
def trial_grid(
    starts: Float[np.ndarray, "trial"], ends: Float[np.ndarray, "trial"]
) -> Tuple[Float[np.ndarray, "row"], Int[np.ndarray, "row"]]:
    """Return left edges and trial IDs for every complete 50 ms trial bin."""
    sizes = np.floor((ends - starts) / BIN_WIDTH + 1e-8).astype(np.int64)
    if np.any(sizes < 1) or np.any(starts[1:] < ends[:-1]):
        raise ValueError("Trials must be ordered, disjoint, and at least one bin long.")
    trial_id = np.repeat(np.arange(len(starts)), sizes)
    offsets = np.arange(sizes.sum()) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    return starts[trial_id] + offsets * BIN_WIDTH, trial_id


@jaxtyped(typechecker=beartype)
def add_spikes(
    counts: Float[np.ndarray, "row neuron"],
    left: Float[np.ndarray, "row"],
    timestamps: Float[np.ndarray, "spike"],
    units: Int[np.ndarray, "spike"],
) -> int:
    """Accumulate spikes in half-open bins; gaps and right endpoints stay out."""
    row = np.searchsorted(left, timestamps, side="right") - 1
    candidate = np.maximum(row, 0)
    valid = (row >= 0) & (timestamps < left[candidate] + BIN_WIDTH)
    np.add.at(counts, (row[valid], units[valid]), 1)
    return int(valid.sum())


@jaxtyped(typechecker=beartype)
def split_trials(conditions: Int[np.ndarray, "trial"], seed: int) -> Int[np.ndarray, "trial"]:
    """Stratify whole trials into reproducible 80/10/10 label partitions."""
    trial_id = np.arange(len(conditions))
    development, test = train_test_split(
        trial_id, test_size=0.1, random_state=seed, stratify=conditions
    )
    train, validation = train_test_split(
        development,
        test_size=1 / 9,
        random_state=seed,
        stratify=conditions[development],
    )
    split = np.zeros(len(conditions), dtype=np.int8)
    split[validation] = 1
    split[test] = 2
    return split


@jaxtyped(typechecker=beartype)
def bin_behavior(
    times: Float[np.ndarray, "sample"],
    positions: Float[np.ndarray, "sample xy"],
    left: Float[np.ndarray, "row"],
) -> Tuple[Float[np.ndarray, "row"], Float[np.ndarray, "row xy"]]:
    """Differentiate within one trial and average sample speed/velocity per bin.

    A bin is invalid if its behavioral sample coverage is incomplete or contains
    missing values. Mean instantaneous speed is distinct from norm(mean velocity).
    No smoothing, interpolation, or derivative crosses a trial boundary.
    """
    speed = np.full(len(left), np.nan)
    velocity = np.full((len(left), 2), np.nan)
    if len(times) < 2:
        return speed, velocity
    if np.any(np.diff(times) <= 0):
        raise ValueError("Behavioral timestamps must increase within a trial.")
    vel = np.gradient(positions, times, axis=0, edge_order=1)
    magnitude = np.linalg.norm(vel, axis=1)
    lo = np.searchsorted(times, left, side="left")
    hi = np.searchsorted(times, left + BIN_WIDTH, side="left")
    for i, (start, stop) in enumerate(zip(lo, hi)):
        t = times[start:stop]
        v = vel[start:stop]
        if (
            len(t) >= 49
            and t[0] - left[i] <= 0.0011
            and left[i] + BIN_WIDTH - t[-1] <= 0.0011
            and np.max(np.diff(t)) <= 0.0011
            and np.isfinite(v).all()
        ):
            speed[i] = magnitude[start:stop].mean()
            velocity[i] = v.mean(axis=0)
    return speed, velocity


@beartype
def self_check() -> None:
    """Exercise empty bins, boundaries, trial gaps, speed units, and splitting."""
    left, ids = trial_grid(np.array([0.0, 1.0]), np.array([0.15, 1.11]))
    np.testing.assert_array_equal(ids, [0, 0, 0, 1, 1])
    counts = np.zeros((5, 2), dtype=np.float32)
    add_spikes(
        counts,
        left,
        np.array([-0.1, 0.0, 0.05, 0.149, 0.2, 0.8, 1.0, 1.1]),
        np.zeros(8, dtype=np.int64),
    )
    np.testing.assert_array_equal(counts[:, 0], [1, 1, 1, 1, 0])
    assert counts[:, 1].sum() == 0
    t = np.arange(151) * 0.001
    speed, vel = bin_behavior(t, np.column_stack((3 * t, 4 * t)), left[:3])
    np.testing.assert_allclose(speed, 5)
    np.testing.assert_allclose(vel, np.tile([3, 4], (3, 1)))
    bad = np.column_stack((3 * t, 4 * t))
    bad[75] = np.nan
    invalid, _ = bin_behavior(t, bad, left[:3])
    assert np.isnan(invalid[1]) and np.isfinite(invalid[[0, 2]]).all()
    conditions = np.repeat(np.arange(4), 100)
    splits = split_trials(conditions, 42)
    np.testing.assert_array_equal(splits, split_trials(conditions, 42))
    np.testing.assert_array_equal(np.bincount(splits), [320, 40, 40])


@beartype
def validate(output: Path) -> Dict:
    """Validate saved shapes, row alignment, flags, split integrity and counts."""
    counts = np.load(output / "counts.npy", mmap_mode="r")
    with np.load(output / "metadata.npz") as source:
        metadata = dict(source)
    with (output / "trials.csv").open() as stream:
        trials = list(csv.DictReader(stream))
    assert counts.dtype == np.float32 and counts.shape[0] == len(metadata["timestamps"])
    assert all(len(values) == len(counts) for values in metadata.values())
    assert np.isfinite(counts).all() and (counts >= 0).all()
    assert np.equal(counts, np.floor(counts)).all()
    ids = metadata["trial_id"]
    np.testing.assert_array_equal(np.unique(ids), np.arange(len(trials)))
    assert np.all(np.diff(metadata["timestamps"]) > 0)
    for i, trial in enumerate(trials):
        row = ids == i
        assert row.sum() == int(trial["n_bins"])
        assert np.unique(metadata["split"][row]).tolist() == [int(trial["split"])]
        np.testing.assert_allclose(np.diff(metadata["timestamps"][row]), BIN_WIDTH)
        assert metadata["timestamps"][row][0] >= float(trial["start"])
        assert metadata["timestamps"][row][-1] <= float(trial["end"])
        np.testing.assert_allclose(metadata["target_x"][row], float(trial["target_x"]))
        np.testing.assert_allclose(metadata["target_y"][row], float(trial["target_y"]))
    assert metadata["movement"].dtype == bool
    assert np.issubdtype(metadata["trial_id"].dtype, np.integer)
    assert np.issubdtype(metadata["split"].dtype, np.integer)
    finite = np.isfinite(metadata["speed"])
    assert np.all(metadata["speed"][finite] >= -1e-9)
    assert np.all(
        metadata["speed"][finite] + 1e-6
        >= np.hypot(metadata["vel_x"][finite], metadata["vel_y"][finite])
    )
    return {
        "shape": list(counts.shape),
        "trials": len(trials),
        "spikes": int(counts.sum(dtype=np.float64)),
        "zero_count_rows": int(np.count_nonzero(counts.sum(axis=1) == 0)),
        "invalid_behavior_rows": int((~finite).sum()),
        "trial_splits": np.bincount([int(t["split"]) for t in trials]).tolist(),
        "row_splits": np.bincount(metadata["split"]).tolist(),
    }


@beartype
def prepare(raw: Path, processed: Path, output: Path, seed: int) -> None:
    """Audit sources, prepare counts/labels, and save validated provenance."""
    self_check()
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Prepared data already exists: {output}; use --validate-only")
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(raw, "r") as nwb, h5py.File(processed, "r") as source:
        rt, pt = nwb["intervals/trials"], source["trials"]
        fields = [
            "start",
            "end",
            "is_valid",
            "correct_reach",
            "discard_trial",
            "task_success",
            "novel_maze",
            "move_begins_time",
            "move_ends_time",
            "maze_condition",
            "hit_target_position",
            "go_cue_time",
            "target_on_time",
        ]
        trials = {key: pt[key][:] for key in fields}
        mapping = {
            "start": "start_time",
            "end": "stop_time",
            "target_on_time": "target_presentation_time",
        }
        offset = trials["start"] - rt["start_time"][:]
        time_fields = {
            "start",
            "end",
            "move_begins_time",
            "move_ends_time",
            "go_cue_time",
            "target_on_time",
        }
        for key in fields:
            if key == "is_valid":
                continue
            expected = rt[mapping.get(key, key)][:]
            if key in time_fields:
                expected = expected + offset
            np.testing.assert_allclose(trials[key], expected, equal_nan=True, atol=1e-8)
        np.testing.assert_array_equal(
            trials["is_valid"],
            (trials["discard_trial"] == 0) & (trials["task_success"] == 1),
        )
        keep = np.ones(len(offset), dtype=bool)
        exclusions = {}
        predicates = {
            "invalid": trials["is_valid"],
            "incorrect_reach": trials["correct_reach"] == 1,
            "novel_maze": trials["novel_maze"] == 0,
            "invalid_interval": np.isfinite(trials["start"])
            & np.isfinite(trials["end"])
            & (trials["end"] - trials["start"] >= BIN_WIDTH),
            "invalid_movement_times": np.isfinite(trials["move_begins_time"])
            & np.isfinite(trials["move_ends_time"])
            & (trials["move_begins_time"] >= trials["start"])
            & (trials["move_ends_time"] <= trials["end"])
            & (trials["move_ends_time"] > trials["move_begins_time"]),
        }
        for reason, valid in predicates.items():
            exclusions[reason] = int(np.count_nonzero(keep & ~valid))
            keep &= valid
        original = rt["id"][:][keep]
        trials = {key: value[keep] for key, value in trials.items()}
        left, trial_id = trial_grid(trials["start"], trials["end"])
        sizes = np.bincount(trial_id)
        split = split_trials(trials["maze_condition"], seed)
        neuron_ids = [x.decode() for x in source["units/id"][:]]
        counts = np.zeros((len(left), len(neuron_ids)), dtype=np.float32)
        print(f"Preparing {len(original)} trials, {counts.shape} counts", flush=True)
        total_spikes = len(source["spikes/timestamps"])
        for start in range(0, total_spikes, 2_000_000):
            stop = min(start + 2_000_000, total_spikes)
            add_spikes(
                counts,
                left,
                source["spikes/timestamps"][start:stop],
                source["spikes/unit_index"][start:stop],
            )
            if start % 10_000_000 == 0:
                print(f"Counted {stop:,}/{total_spikes:,} source spikes", flush=True)
        times = source["hand/timestamps"][:]
        positions = source["hand/pos_2d"][:]
        speed = np.full(len(left), np.nan)
        velocity = np.full((len(left), 2), np.nan)
        row_offsets = np.r_[0, np.cumsum(sizes)]
        for i, (start, stop) in enumerate(zip(trials["start"], trials["end"])):
            lo, hi = np.searchsorted(times, [start - 1e-9, stop], side="left")
            rows = slice(row_offsets[i], row_offsets[i + 1])
            speed[rows], velocity[rows] = bin_behavior(times[lo:hi], positions[lo:hi], left[rows])
        centers = left + BIN_WIDTH / 2
        metadata = {
            "timestamps": centers,
            "trial_id": trial_id,
            "speed": speed,
            "vel_x": velocity[:, 0],
            "vel_y": velocity[:, 1],
            "target_x": trials["hit_target_position"][trial_id, 0].astype(float),
            "target_y": trials["hit_target_position"][trial_id, 1].astype(float),
            "movement": (centers >= trials["move_begins_time"][trial_id])
            & (centers < trials["move_ends_time"][trial_id]),
            "split": split[trial_id],
        }
        np.save(output / "counts.npy", counts)
        np.savez_compressed(output / "metadata.npz", **metadata)
        rows = []
        for i in range(len(original)):
            rows.append(
                {
                    "trial_id": i,
                    "original_trial_id": int(original[i]),
                    "start": trials["start"][i],
                    "end": trials["end"][i],
                    "movement_onset": trials["move_begins_time"][i],
                    "movement_end": trials["move_ends_time"][i],
                    "target_x": int(trials["hit_target_position"][i, 0]),
                    "target_y": int(trials["hit_target_position"][i, 1]),
                    "maze_condition": int(trials["maze_condition"][i]),
                    "split": int(split[i]),
                    "n_bins": int(sizes[i]),
                    "terminal_remainder_seconds": trials["end"][i]
                    - trials["start"][i]
                    - sizes[i] * BIN_WIDTH,
                }
            )
        with (output / "trials.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        raw_unit = nwb["processing/behavior/Position/Hand/data"].attrs["unit"]
        raw_conversion = nwb["processing/behavior/Position/Hand/data"].attrs["conversion"]
        unit_metadata = []
        electrodes = nwb["general/extracellular_ephys/electrodes"]
        for index, identifier in enumerate(neuron_ids):
            raw_index = int(identifier.split("elec")[-1])
            unit_metadata.append(
                {
                    "column": index,
                    "id": f"nitschke_20090812/{identifier}",
                    "raw_unit_row": raw_index,
                    "raw_unit_id": int(nwb["units/id"][raw_index]),
                    "electrode_group": electrodes["group_name"][raw_index].decode(),
                    "location": electrodes["location"][raw_index].decode(),
                    "recording_type": "Utah array threshold crossings (not spike-sorted single cells)",
                }
            )
        manifest = {
            "schema_version": 1,
            "session": "nitschke_20090812",
            "bin_width_seconds": BIN_WIDTH,
            "timestamp_definition": "Bin center, seconds on processed repaired session clock",
            "binning": "Trial-start anchored complete half-open 50 ms bins, zeros retained",
            "terminal_remainder_policy": "Exclude only final incomplete bin (<50 ms per trial)",
            "total_terminal_remainder_seconds": sum(r["terminal_remainder_seconds"] for r in rows),
            "feature_period": "Whole retained trial: all complete bins, including hold/reach/return",
            "inclusion": "is_valid & correct_reach & ~novel_maze & valid interval/movement times",
            "source_trials": len(offset),
            "excluded_trials_sequential": exclusions,
            "clock_repair_offsets_seconds": np.unique(np.round(offset, 9)).tolist(),
            "clock_repair": "Brainsets blockwise repair verified against raw NWB trial time fields",
            "neurons": unit_metadata,
            "neuron_ids": [u["id"] for u in unit_metadata],
            "channel_exclusion": "Processed source excludes raw channel 1 for malformed spike timestamp blocks; no model-dependent selection",
            "units": {
                "speed": "source_position_units/s",
                "velocity": "source_position_units/s",
                "target": "source_position_units",
                "timestamps": "s",
                "counts": "spikes/bin",
            },
            "unit_audit": {
                "nwb_hand_declared_unit": str(raw_unit),
                "nwb_hand_conversion": None
                if not np.isfinite(raw_conversion)
                else float(raw_conversion),
                "status": "Physical position scale unresolved: NWB declares meters with NaN conversion; retain native numeric scale, do not label m/s",
                "velocity_computation": "Trial-local numerical position derivative in seconds; mean instantaneous speed within bin",
                "behavior_smoothing": "None added; positions are supplied by original source",
                "conversion_source": "https://github.com/catalystneuro/shenoy-lab-to-nwb/blob/319acb472372c9e4359ebacf13c8df3652a822b1/maze_task_unsorted/shenoymatdatainterface.py",
                "conversion_evidence": "Converter copies native HAND X/Y positions and explicitly passes conversion=np.nan; no physical calibration is supplied",
                "missing_behavior": "Hand-position samples end before nominal trial stop; insufficiently covered bins remain NaN while neural counts and target labels are retained",
            },
            "label_split": {
                "seed": seed,
                "fractions": [0.8, 0.1, 0.1],
                "stratification": "maze_condition",
                "procedure": "sklearn train_test_split 10% test, then 1/9 remaining validation; ceil rounding",
                "codes": {"0": "train", "1": "validation", "2": "test"},
            },
            "data_use": "All retained neural rows for unsupervised representation learning; split applies only to labels (transductive/offline)",
            "sources": {},
        }
    manifest["validation"] = validate(output)
    print(json.dumps(manifest["validation"]), flush=True)
    for name, path in {"nwb": raw, "processed_hdf5": processed}.items():
        print(f"Hashing {path.name}", flush=True)
        manifest["sources"][name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": checksum(path),
        }
    manifest["sources"]["dandi"] = "https://dandiarchive.org/dandiset/000070"
    manifest["artifacts"] = {
        name: {"sha256": checksum(output / name)}
        for name in ["counts.npy", "metadata.npz", "trials.csv"]
    }
    manifest["preparation_script_sha256"] = checksum(Path(__file__))
    with (output / "manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Prepared and validated {output}", flush=True)


@beartype
def main() -> None:
    """Run source preparation or executable validation checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "data/raw/sub-Nitschke_ses-20090812_behavior+ecephys.nwb",
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default=ROOT / "data/processed/nitschke_20090812_center_out_reaching.h5",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("Preparation self-checks passed")
    elif args.validate_only:
        self_check()
        print(json.dumps(validate(args.output), indent=2))
    else:
        prepare(args.raw, args.processed, args.output, args.split_seed)


if __name__ == "__main__":
    main()
