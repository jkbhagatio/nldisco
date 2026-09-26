"""Churchland MC Maze data loading and preprocessing for one experiment."""

import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import brainsets_pipelines.churchland_shenoy_neural_2012.prepare_data as prep
import h5py
import numpy as np
import pandas as pd
import requests
from dandi.dandiapi import DandiAPIClient
from plotly import colors as pc
from plotly import express as px
from temporaldata import Data
from tqdm import tqdm


def download_with_progress(url, dest: Path, chunk_size=1024 * 1024):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with (
        open(dest, "wb") as f,
        tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
    ):
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def download_and_preprocess(raw_dir, processed_dir, subject_name: str, num_files: int):
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    subj = subject_name.strip().lower()
    if subj not in {"jenkins", "nitschke"}:
        raise ValueError("subject_name must be 'Jenkins' or 'Nitschke'")

    # Date-ordered lists (ascending) per subject
    jenkins_assets = [
        "sub-Jenkins/sub-Jenkins_ses-20090912_behavior+ecephys.nwb",
        "sub-Jenkins/sub-Jenkins_ses-20090916_behavior+ecephys.nwb",
        "sub-Jenkins/sub-Jenkins_ses-20090918_behavior+ecephys.nwb",
        "sub-Jenkins/sub-Jenkins_ses-20090923_behavior+ecephys.nwb",
    ]
    nitschke_assets = [
        "sub-Nitschke/sub-Nitschke_ses-20090812_behavior+ecephys.nwb",
        "sub-Nitschke/sub-Nitschke_ses-20090819_behavior+ecephys.nwb",
        "sub-Nitschke/sub-Nitschke_ses-20090910_behavior+ecephys.nwb",
    ]

    if subj == "jenkins":
        cap = 4
        asset_paths = jenkins_assets[: min(num_files, cap)]
    else:
        cap = 3
        asset_paths = nitschke_assets[: min(num_files, cap)]

    # Download selected assets
    local_paths = []
    with DandiAPIClient() as client:
        ds = client.get_dandiset("000070", "draft")
        for ap in asset_paths:
            asset = ds.get_asset_by_path(ap)
            local_path = raw_dir / Path(ap).name
            if local_path.exists():
                print(f"Already exists: {local_path}")
            else:
                url = asset.download_url  # direct link to file
                download_with_progress(url, local_path)
                print(f"Downloaded: {local_path}")
            local_paths.append(local_path)

    # Preprocess with brainsets pipeline
    for nwb_path in tqdm(local_paths, desc="Preprocessing NWB files", unit="file"):
        date = nwb_path.stem.split("_")[1].replace("ses-", "")
        out_file = processed_dir / f"{subj}_{date}_center_out_reaching.h5"

        if out_file.exists():
            print(f"Already preprocessed: {out_file}")
            continue

        _argv = sys.argv[:]
        try:
            sys.argv = [
                "prepare_data",
                "--input_file",
                str(nwb_path),
                "--output_dir",
                str(processed_dir),
            ]
            prep.main()
        finally:
            sys.argv = _argv


def clean_session_data(session: Data) -> Data:
    """Clean session data by filtering trials and spikes based on quality criteria."""
    # Filter out extraneous trials
    good_trials = (
        (session.trials.trial_type > 0)
        & (session.trials.is_valid == 1)
        & (session.trials.discard_trial == 0)
        & (session.trials.novel_maze == 0)
        & (session.trials.trial_version < 3)
    )
    session.trials = session.trials.select_by_mask(good_trials)

    # Keep only successful trials
    success = session.trials.task_success == 1
    session.trials = session.trials.select_by_mask(success)

    # Keep trials that are long enough
    post_move = 0.8  # must have at least this many ms after movement onset
    long_enough = session.trials.end - session.trials.move_begins_time >= post_move
    session.trials = session.trials.select_by_mask(long_enough)

    # Keep consistent reaches
    consistent = session.trials.correct_reach == 1
    session.trials = session.trials.select_by_mask(consistent)

    # Ensure primary conditions are monotonic and start at 1
    primary_conditions = np.unique(session.trials.maze_condition)
    if min(primary_conditions) != 1 or len(np.unique(np.diff(primary_conditions))) != 1:
        raise ValueError("Primary conditions are not monotonic or do not start from 1")

    # Restrict spikes/hand/eye to cleaned trials
    session.spikes = session.spikes.select_by_interval(session.trials)
    session.hand = session.hand.select_by_interval(session.trials)
    session.eye = session.eye.select_by_interval(session.trials)

    # Convert session recording date to timestamp
    session.session.recording_date = datetime.strptime(
        session.session.recording_date, "%Y-%m-%d %H:%M:%S"
    ).timestamp()

    return session


def analyze_maze_conditions(session: Data) -> Tuple[pd.DataFrame, object]:
    """Analyze maze conditions and return summary dataframe and plot."""
    unique_conditions = np.unique(session.trials.maze_condition)

    condition_summary = []
    for condition in unique_conditions:
        condition_mask = session.trials.maze_condition == condition

        barriers = np.unique(session.trials.maze_num_barriers[condition_mask])
        targets = np.unique(session.trials.maze_num_targets[condition_mask])
        hit_position = np.unique(session.trials.hit_target_position[condition_mask], axis=0)
        if len(hit_position) > 1:
            raise ValueError(f"Condition {condition} has multiple hit positions: {hit_position}")
        hit_position = hit_position[0]

        num_trials = int(condition_mask.sum())
        condition_summary.append(
            {
                "Maze Condition": condition,
                "Trials": num_trials,
                "Barriers": barriers,
                "Targets": targets,
                "Hit Position": hit_position,
                "Hit Position Angles": str(
                    np.degrees(np.arctan2(hit_position[1], hit_position[0]))
                ),
            }
        )

    summary_df = pd.DataFrame(condition_summary)

    # Deduplicate hit positions for plotting
    summary_df_temp = summary_df.copy()
    summary_df_temp["Hit Position Tuple"] = summary_df_temp["Hit Position"].apply(tuple)
    plot_df = summary_df_temp.drop_duplicates(subset=["Hit Position Tuple"], keep="first").drop(
        "Hit Position Tuple", axis=1
    )

    plot_data = pd.DataFrame(
        {
            "Hit Position X": plot_df["Hit Position"].apply(lambda x: x[0]),
            "Hit Position Y": plot_df["Hit Position"].apply(lambda x: x[1]),
            "Maze Condition": plot_df["Maze Condition"].astype(str),
        }
    )

    # Generate unique colors
    n_conditions = len(plot_data["Maze Condition"].unique())
    colors = pc.sample_colorscale(
        "viridis", [i / max(n_conditions - 1, 1) for i in range(n_conditions)]
    )

    fig = px.scatter(
        plot_data,
        x="Hit Position X",
        y="Hit Position Y",
        color="Maze Condition",
        labels={
            "Hit Position X": "Hit Position X",
            "Hit Position Y": "Hit Position Y",
            "color": "Maze Condition",
        },
        title="Hit Position by Maze Condition",
        color_discrete_sequence=colors,
        hover_data=["Maze Condition"],
    )
    fig.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[-150, 150]),
        yaxis=dict(constrain="domain", range=[-150, 150]),
        width=600,
        height=600,
    )

    return summary_df, fig


def fix_maze_conditions_consistency(sessions: List[Data]) -> List[Data]:
    """Fix maze condition numbering to be consistent across all sessions."""

    def get_maze_signature(session: Data, condition: int):
        condition_mask = session.trials.maze_condition == condition
        barriers = tuple(np.unique(session.trials.maze_num_barriers[condition_mask]))
        targets = tuple(np.unique(session.trials.maze_num_targets[condition_mask]))
        hit_position = np.unique(session.trials.hit_target_position[condition_mask], axis=0)
        if len(barriers) > 1 or len(targets) > 1 or len(hit_position) > 1:
            raise ValueError(
                f"Condition {condition} has >1 unique values: "
                f"barriers={barriers}, targets={targets}, hit_position={hit_position}."
            )
        return (barriers, targets, tuple(tuple(hit_position[0])))

    def get_group_signature(session: Data, group_conditions: List[int]):
        return tuple(get_maze_signature(session, cond) for cond in sorted(group_conditions))

    # Keep only sessions with multiples of 3 conditions
    valid_sessions: List[Data] = []
    for session in sessions:
        num_conditions = len(np.unique(session.trials.maze_condition))
        if num_conditions % 3 == 0:
            valid_sessions.append(session)
    if not valid_sessions:
        raise ValueError("No valid sessions remaining after filtering")

    # Global registry of unique group signatures -> group index (0-based)
    group_sig_to_idx = {}
    next_group_idx = 0

    processed_sessions: List[Data] = []
    for session in valid_sessions:
        conditions = sorted(np.unique(session.trials.maze_condition))
        session_groups = [conditions[i : i + 3] for i in range(0, len(conditions), 3)]

        condition_mapping = {}
        for group in session_groups:
            group_sig = get_group_signature(session, group)
            if group_sig not in group_sig_to_idx:
                group_sig_to_idx[group_sig] = next_group_idx
                next_group_idx += 1
            base = group_sig_to_idx[group_sig] * 3 + 1
            for i, old in enumerate(sorted(group)):
                condition_mapping[old] = base + i

        session.trials.maze_condition = np.array(
            [condition_mapping[old] for old in session.trials.maze_condition]
        )
        processed_sessions.append(session)

    return processed_sessions


def load_sessions(
    data_path: Path,
    subject_name: str,
) -> List[Data]:
    """Load, clean, and harmonise MC Maze sessions from HDF5 files."""
    subject_name = subject_name.lower()

    # Allowed filenames per subject (case-insensitive match)
    j_allowed = [
        "jenkins_20090912_center_out_reaching.h5",
        "jenkins_20090916_center_out_reaching.h5",
        "jenkins_20090918_center_out_reaching.h5",
        "jenkins_20090923_center_out_reaching.h5",
    ]
    n_allowed = [
        "nitschke_20090812_center_out_reaching.h5",
        "nitschke_20090819_center_out_reaching.h5",
        "nitschke_20090910_center_out_reaching.h5",
    ]

    # Detect files present in the directory
    h5_files = [p.name for p in Path(data_path).iterdir() if p.suffix.lower() == ".h5"]

    # Filter strictly to the allowed filenames (present on disk), preserving the intended order
    j_files = [fn for fn in j_allowed if fn.lower() in {f.lower() for f in h5_files}]
    n_files = [fn for fn in n_allowed if fn.lower() in {f.lower() for f in h5_files}]

    if subject_name == "jenkins":
        subject_files = j_files
    elif subject_name == "nitschke":
        subject_files = n_files
    else:
        raise ValueError(
            f"Unsupported subject '{subject_name}'. Expected 'jenkins' or 'nitschke'."
        )

    if not subject_files:
        raise FileNotFoundError(
            f"No allowed files found for subject {subject_name} in {data_path}"
        )

    sessions: List[Data] = []
    total = len(subject_files)
    for i, fname in enumerate(subject_files, start=1):
        file_path = Path(data_path) / fname
        print(f"\nLoading file {i}/{total}: {fname}")

        with h5py.File(str(file_path), "r") as f:
            session = Data.from_hdf5(f)

            session.spikes.materialize()
            session.trials.materialize()
            session.hand.materialize()
            session.eye.materialize()
            session.session.materialize()
            session.units.materialize()

            print(f"Session ID: {session.session.id}")
            print(f"Session subject id: {session.subject.id}")
            print(f"Session subject sex: {session.subject.sex}")
            print(f"Session subject species: {session.subject.species}")
            print(f"Session recording date: {session.session.recording_date}")
            print(f"Original number of trials: {len(session.trials.start)}")

            try:
                print("Cleaning data...")
                session = clean_session_data(session)
                print(f"Final number of trials after cleaning: {len(session.trials.start)}")

                sessions.append(session)
            except Exception as e:
                print(f"Error processing session {session.session.id}: {e}")
                continue

    print(f"\nSuccessfully loaded and cleaned {len(sessions)} sessions for subject {subject_name}")

    sessions = fix_maze_conditions_consistency(sessions)

    print("\nList of maze conditions present in each session:")
    for session in sessions:
        unique_conditions = sorted(np.unique(session.trials.maze_condition))
        print(f"Session {session.session.id}: {unique_conditions}")

    return sessions


def bin_spike_data(
    sessions: List[Data],
    bin_size: float,
) -> pd.DataFrame:
    """Bin spike data across multiple sessions into a single dataframe."""
    # Check unit consistency across sessions
    unit_ids = np.unique(sessions[0].spikes.unit_index)
    for session in sessions:
        unique_units = np.unique(session.spikes.unit_index)
        if not np.array_equal(unique_units, unit_ids):
            raise ValueError("Sessions do not have the same unit IDs. Cannot combine spike data.")

    # Global bin alignment start
    global_start = min(
        session.session.recording_date + session.trials.start.min() for session in sessions
    )
    global_start = np.floor(global_start / bin_size) * bin_size

    # Decimal precision for timestamps
    n_decimals = int(-np.log10(bin_size)) + 1 if bin_size < 1 else 0

    session_spikes_dfs: List[pd.DataFrame] = []
    for session in sessions:
        # Absolute spike timestamps
        abs_timestamps = session.spikes.timestamps + session.session.recording_date
        unit_ids_this_session = session.spikes.unit_index
        df_spikes = pd.DataFrame(
            {"timestamp": abs_timestamps, "unit": unit_ids_this_session}
        ).sort_values("timestamp")

        # Absolute trial times
        trial_starts = session.trials.start + session.session.recording_date
        trial_ends = session.trials.end + session.session.recording_date
        df_trials = pd.DataFrame(
            {"trial_start": trial_starts, "trial_end": trial_ends}
        ).sort_values("trial_start")

        # Assign spikes to trials
        df_merged = pd.merge_asof(
            df_spikes,
            df_trials[["trial_start", "trial_end"]],
            left_on="timestamp",
            right_on="trial_start",
            direction="backward",
        )
        df_merged = df_merged[df_merged["timestamp"] < df_merged["trial_end"]]

        # Compute bin index
        df_merged["bin"] = ((df_merged["timestamp"] - global_start) / bin_size).astype(int)

        # Group and pivot
        df_counts = (
            df_merged.groupby(["bin", "unit"], observed=True).size().reset_index(name="count")
        )
        session_spikes_df = df_counts.pivot_table(
            index="bin", columns="unit", values="count", fill_value=0
        )

        # Consistent timestamps
        session_timestamps = np.round(
            global_start + session_spikes_df.index * bin_size, n_decimals
        )
        session_spikes_df.index = pd.Index(session_timestamps, name="timestamp")
        session_spikes_df.columns.name = None

        session_spikes_dfs.append(session_spikes_df)

    # Concatenate across sessions
    spikes_df = pd.concat(session_spikes_dfs)
    spikes_df = spikes_df[~spikes_df.index.duplicated()]
    spikes_df.sort_index(inplace=True)

    return spikes_df


def retrieve_metadata(sessions: List[Data]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build continuous metadata (hand, eye, events, trial mapping) across sessions."""
    hand_data = defaultdict(list)
    eye_data = defaultdict(list)
    trial_data = defaultdict(list)

    for i, session in enumerate(sessions):
        recording_date = session.session.recording_date

        # Hand (absolute timestamps)
        timestamps = session.hand.timestamps + recording_date
        acc = session.hand.acc_2d
        pos = session.hand.pos_2d
        vel = session.hand.vel_2d

        hand_data["timestamp"].append(timestamps)
        hand_data["acc_x"].append(acc[:, 0])
        hand_data["acc_y"].append(acc[:, 1])
        hand_data["pos_x"].append(pos[:, 0])
        hand_data["pos_y"].append(pos[:, 1])
        hand_data["vel_x"].append(vel[:, 0])
        hand_data["vel_y"].append(vel[:, 1])
        hand_data["session"].append(np.full(len(timestamps), i))

        # Eye (absolute timestamps)
        eye_timestamps = session.eye.timestamps + recording_date
        eye_pos = session.eye.pos
        eye_data["timestamp"].append(eye_timestamps)
        eye_data["pos_x"].append(eye_pos[:, 0])
        eye_data["pos_y"].append(eye_pos[:, 1])

        # Trials (absolute timestamps for all key events)
        trial_data["start"].append(session.trials.start + recording_date)
        trial_data["end"].append(session.trials.end + recording_date)
        trial_data["target_on_time"].append(session.trials.target_on_time + recording_date)
        trial_data["go_cue_time"].append(session.trials.go_cue_time + recording_date)
        trial_data["move_begins_time"].append(session.trials.move_begins_time + recording_date)
        trial_data["move_ends_time"].append(session.trials.move_ends_time + recording_date)
        trial_data["maze_condition"].append(session.trials.maze_condition)
        trial_data["barriers"].append(session.trials.maze_num_barriers)
        trial_data["targets"].append(session.trials.maze_num_targets)
        trial_data["hit_position_x"].append([p[0] for p in session.trials.hit_target_position])
        trial_data["hit_position_y"].append([p[1] for p in session.trials.hit_target_position])
        trial_data["hit_position_angle"].append(
            [np.degrees(np.arctan2(p[1], p[0])) for p in session.trials.hit_target_position]
        )

    # Concatenate arrays → unified per-stream tables
    combined_hand = {k: np.concatenate(v) for k, v in hand_data.items()}
    combined_eye = {k: np.concatenate(v) for k, v in eye_data.items()}
    combined_trials = {k: np.concatenate(v) for k, v in trial_data.items()}

    combined_hand_df = pd.DataFrame(combined_hand).set_index("timestamp")
    combined_eye_df = pd.DataFrame(combined_eye).set_index("timestamp")
    combined_trials_df = pd.DataFrame(combined_trials)

    # Hit target position as tuples
    combined_trials_df["hit_target_position"] = list(
        zip(combined_trials_df["hit_position_x"], combined_trials_df["hit_position_y"])
    )

    # Master timestamp index (hand + eye + all event times)
    all_event_ts = np.concatenate(
        [
            combined_trials_df["target_on_time"].values,
            combined_trials_df["go_cue_time"].values,
            combined_trials_df["move_begins_time"].values,
            combined_trials_df["move_ends_time"].values,
        ]
    )
    all_ts = np.unique(
        np.concatenate([combined_hand_df.index.values, combined_eye_df.index.values, all_event_ts])
    )
    metadata = pd.DataFrame(index=pd.Index(all_ts, name="timestamp"))

    # Join kinematics
    metadata = metadata.join(combined_hand_df, how="left")
    metadata = metadata.join(combined_eye_df, how="left", rsuffix="_eye")

    # Event column (mark exact event timestamps)
    event_map = {
        "target_on_time": "target_on",
        "go_cue_time": "go_cue",
        "move_begins_time": "move_begins",
        "move_ends_time": "move_ends",
    }
    event_col = pd.Series(index=metadata.index, dtype="object")
    for col, label in event_map.items():
        event_times = combined_trials_df[col].values
        mask = np.isin(metadata.index.values, event_times)
        # Use .iloc with boolean mask aligned to metadata index positions
        event_col.iloc[mask] = label
    metadata["event"] = event_col

    # Efficient trial assignment for each timestamp via binary search
    trial_sort_idx = np.argsort(combined_trials_df["start"].values)
    starts = combined_trials_df["start"].values[trial_sort_idx]
    ends = combined_trials_df["end"].values[trial_sort_idx]
    ts_vals = metadata.index.values

    start_pos = np.searchsorted(starts, ts_vals, side="right") - 1
    valid_mask = (start_pos >= 0) & (start_pos < len(starts))
    valid_pos = start_pos[valid_mask]
    valid_ts = ts_vals[valid_mask]
    within_end = valid_ts <= ends[valid_pos]

    final_mask = np.zeros(len(ts_vals), dtype=bool)
    final_mask[valid_mask] = within_end

    trial_indices = np.full(len(ts_vals), np.nan)
    trial_indices[final_mask] = trial_sort_idx[valid_pos[within_end]]

    metadata["trial_idx"] = pd.Series(trial_indices, index=metadata.index, dtype="float64")

    # Map trial-level fields onto per-timestamp rows
    for col in [
        "maze_condition",
        "barriers",
        "targets",
        "hit_position_x",
        "hit_position_y",
        "hit_position_angle",
    ]:
        metadata[col] = metadata["trial_idx"].astype("Int64").map(combined_trials_df[col])

    # Derived kinematic scalars
    dx = metadata["pos_x"].diff()
    dy = metadata["pos_y"].diff()
    metadata["movement_angle"] = np.degrees(np.arctan2(dy, dx))
    metadata["vel_magnitude"] = np.sqrt(
        metadata["vel_x"].values ** 2 + metadata["vel_y"].values ** 2
    )
    metadata["accel_magnitude"] = np.sqrt(
        metadata["acc_x"].values ** 2 + metadata["acc_y"].values ** 2
    )

    return metadata, combined_trials_df


def bin_metadata(
    metadata: pd.DataFrame,
    combined_trials_df: pd.DataFrame,
    bin_size: float,
    spikes_df_index: pd.Index,
) -> pd.DataFrame:
    """Bin continuous metadata to the same time grid as spike counts."""
    ts = spikes_df_index.values
    metadata_binned = pd.DataFrame(index=pd.Index(ts, name="timestamp"))

    # Assign each metadata timestamp to a bin: half-open [t, t+bin)
    bin_ids = np.searchsorted(ts, metadata.index.values, side="right") - 1
    bin_ids = np.clip(bin_ids, 0, len(ts) - 1)

    # Distribute events forward by offsetting bin index in-order (keeps simple ordering)
    events = metadata["event"].values
    event_mask = pd.notna(events)
    if event_mask.any():
        event_bin_ids = bin_ids[event_mask]
        valid_events = events[event_mask].astype(str)
        event_agg = np.full(len(ts), None, dtype=object)

        evt_idx = np.arange(len(valid_events))
        target_bins = event_bin_ids + evt_idx  # simple spreading; clipped below
        valid = target_bins < len(ts)
        event_agg[target_bins[valid]] = valid_events[valid]
    else:
        event_agg = np.full(len(ts), None, dtype=object)

    # Nearest-neighbour reindex for continuous columns
    meta_ts = metadata.index.values
    pos = np.searchsorted(meta_ts, ts, side="left")
    pos = np.clip(pos, 0, len(meta_ts) - 1)

    left_pos = np.maximum(pos - 1, 0)
    left_dist = np.abs(ts - meta_ts[left_pos])
    right_dist = np.abs(ts - meta_ts[pos])
    nearest = np.where(left_dist < right_dist, left_pos, pos)

    for col in metadata.columns:
        if col not in ("event", "trial_idx"):
            metadata_binned[col] = metadata[col].iloc[nearest].values

    metadata_binned["event"] = event_agg

    # Override trial_idx by exact interval overlap of bins to prevent NN bleed
    starts = combined_trials_df["start"].to_numpy()
    ends = combined_trials_df["end"].to_numpy()
    order = np.argsort(starts)
    s, e = starts[order], ends[order]

    left = ts
    right = ts + bin_size  # bins are [left, right)
    cand = np.searchsorted(s, right, side="right") - 1
    valid = (cand >= 0) & (left < e[cand]) & (right > s[cand])

    trial_idx_binned = np.full(len(ts), np.nan)
    trial_idx_binned[valid] = order[cand[valid]]
    metadata_binned["trial_idx"] = pd.Series(
        trial_idx_binned, index=metadata_binned.index, dtype="float64"
    )

    # Mark first/last bin of each trial
    first = metadata_binned.groupby("trial_idx", sort=False).head(1).index
    last = metadata_binned.groupby("trial_idx", sort=False).tail(1).index
    metadata_binned.loc[first, "event"] = "start"
    metadata_binned.loc[last.difference(first), "event"] = "end"

    # Fill “A -> B” transitions inside trials
    metadata_binned["event"] = metadata_binned["event"].astype(object)
    prev_ev = metadata_binned.groupby("trial_idx", sort=False)["event"].ffill()
    next_ev = metadata_binned.groupby("trial_idx", sort=False)["event"].bfill()
    between = metadata_binned["event"].isna() & prev_ev.notna() & next_ev.notna()
    metadata_binned.loc[between, "event"] = prev_ev[between] + " -> " + next_ev[between]

    allowed = {
        "start",
        "start -> target_on",
        "target_on",
        "target_on -> go_cue",
        "go_cue",
        "go_cue -> move_begins",
        "move_begins",
        "move_begins -> move_ends",
        "move_ends",
        "move_ends -> end",
        "end",
    }
    metadata_binned["event"] = metadata_binned["event"].where(
        metadata_binned["event"].isin(allowed)
    )

    return metadata_binned
