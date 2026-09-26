"""Regression checks for current and legacy Churchland preprocessing files."""

# Optional dependencies must be checked before importing the data helpers.
# ruff: noqa: E402

import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("brainsets")
h5py = pytest.importorskip("h5py")

from brainsets import serialize_fn_map
from brainsets.descriptions import SessionDescription
from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries

from experiments.churchland import data
from experiments.churchland.scripts import prepare


def make_session(date="20090812", legacy=False):
    """Make three valid conditions and one trial excluded by its quality flag."""
    starts = np.arange(4, dtype=float) * 2
    trials = Interval(
        start=starts,
        end=starts + 1.5,
        move_begins_time=starts + 0.2,
        trial_type=np.ones(4),
        is_valid=np.array([1, 1, 1, 0]),
        discard_trial=np.zeros(4),
        novel_maze=np.zeros(4),
        trial_version=np.ones(4),
        task_success=np.ones(4),
        correct_reach=np.ones(4),
        maze_condition=np.array([1, 2, 3, 1]),
        maze_num_barriers=np.zeros(4),
        maze_num_targets=np.ones(4),
        hit_target_position=np.array([[1, 0], [0, 1], [-1, 0], [1, 0]]),
    )
    recording_date = datetime.strptime(date, "%Y%m%d")
    metadata = (
        Data(id=f"nitschke_{date}_center_out_reaching", recording_date=str(recording_date))
        if legacy
        else SessionDescription(id=f"nitschke_{date}_maze", recording_date=recording_date)
    )
    timestamps = starts + 0.3
    return Data(
        session=metadata,
        trials=trials,
        spikes=IrregularTimeSeries(
            timestamps=timestamps, unit_index=np.zeros(4, dtype=int), domain="auto"
        ),
        hand=IrregularTimeSeries(timestamps=timestamps, pos_2d=np.ones((4, 2)), domain="auto"),
        eye=IrregularTimeSeries(timestamps=timestamps, pos=np.zeros((4, 2)), domain="auto"),
        cursor=IrregularTimeSeries(timestamps=timestamps, pos=np.ones((4, 2)), domain="auto"),
        units=ArrayDict(id=np.array(["unit0"])),
        train_domain=Interval(start=starts, end=starts + 1.5),
        domain="auto",
    )


def write_session(path, date="20090812", legacy=False):
    with h5py.File(path, "w") as file:
        make_session(date, legacy).to_hdf5(file, serialize_fn_map=serialize_fn_map)


@pytest.mark.parametrize("legacy", [False, True])
def test_load_formats_and_materialize_before_closing(tmp_path, legacy):
    suffix = "center_out_reaching" if legacy else "maze"
    write_session(tmp_path / f"NITSCHKE_20090812_{suffix.upper()}.H5", legacy=legacy)
    (session,) = data.load_sessions(tmp_path, " Nitschke ")
    np.testing.assert_array_equal(session.trials.maze_condition, [1, 2, 3])
    np.testing.assert_array_equal(session.spikes.timestamps, [0.3, 2.3, 4.3])
    np.testing.assert_array_equal(session.hand.pos_2d, np.ones((3, 2)))
    np.testing.assert_array_equal(session.eye.pos, np.zeros((3, 2)))
    np.testing.assert_array_equal(session.units.id, ["unit0"])
    # These fields were left lazy by the old loader and failed after file close.
    np.testing.assert_array_equal(session.cursor.pos, np.ones((4, 2)))
    np.testing.assert_array_equal(session.train_domain.start, [0, 2, 4, 6])
    assert session.session.recording_date == datetime(2009, 8, 12).timestamp()


def test_current_format_wins_without_duplicate_sessions_and_dates_are_sorted(tmp_path):
    write_session(tmp_path / "nitschke_20090819_maze.h5", date="20090819")
    write_session(tmp_path / "nitschke_20090812_center_out_reaching.h5", legacy=True)
    write_session(tmp_path / "nitschke_20090812_maze.h5")
    (tmp_path / "nitschke_19990101_maze.h5").write_text("unrelated file")
    assert [s.session.id for s in data.load_sessions(tmp_path, "Nitschke")] == [
        "nitschke_20090812_maze",
        "nitschke_20090819_maze",
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("trial_type", 0),
        ("is_valid", 0),
        ("discard_trial", 1),
        ("novel_maze", 1),
        ("trial_version", 3),
        ("task_success", 0),
        ("correct_reach", 0),
        ("move_begins_time", 7.0),
    ],
)
def test_trial_quality_filters_still_restrict_spikes_and_behavior(field, value):
    session = make_session()
    session.trials.is_valid[-1] = 1
    getattr(session.trials, field)[-1] = value
    result = data.clean_session_data(session)
    assert len(result.trials) == len(result.spikes) == len(result.hand) == len(result.eye) == 3


@pytest.mark.parametrize("as_string", [False, True])
def test_recording_date_preserves_timezone(as_string):
    session = make_session()
    date = datetime(2009, 8, 12, tzinfo=timezone.utc)
    session.session.recording_date = date.isoformat() if as_string else date
    assert data.clean_session_data(session).session.recording_date == date.timestamp()


def test_invalid_session_is_reported_with_filename_and_cause(tmp_path):
    path = tmp_path / "nitschke_20090812_maze.h5"
    path.write_text("not HDF5")
    with pytest.raises(ValueError, match=path.name) as caught:
        data.load_sessions(tmp_path, "Nitschke")
    assert isinstance(caught.value.__cause__, OSError)


def test_missing_subject_data_and_invalid_subject(tmp_path):
    with pytest.raises(FileNotFoundError, match="No allowed files"):
        data.load_sessions(tmp_path, "Nitschke")
    with pytest.raises(ValueError, match="subject_name"):
        data.load_sessions(tmp_path, "unknown")


@pytest.mark.parametrize("legacy", [False, True])
def test_cached_nwb_uses_pipeline_without_network_or_cli_mutation(tmp_path, monkeypatch, legacy):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    source = raw / "sub-Nitschke_ses-20090812_behavior+ecephys.nwb"
    source.touch()
    network = MagicMock(side_effect=AssertionError("unexpected network access"))
    monkeypatch.setattr(data, "DandiAPIClient", network)
    calls = []

    def process(pipeline, path):
        assert path == source
        assert pipeline.raw_dir == raw
        assert not pipeline.args.reprocess and not pipeline.args.redownload
        calls.append(path)
        suffix = "center_out_reaching" if legacy else "maze"
        write_session(pipeline.processed_dir / f"nitschke_20090812_{suffix}.h5", legacy=legacy)

    monkeypatch.setattr(data.Pipeline, "process", process)
    monkeypatch.setattr(sys, "argv", ["host-program", "--unrelated-option"])
    data.download_and_preprocess(raw, processed, "Nitschke", 1)
    # Cached outputs also work when the raw directory has no recording.
    data.download_and_preprocess(tmp_path / "empty", processed, "Nitschke", 1)
    assert calls == [source]
    assert sys.argv == ["host-program", "--unrelated-option"]
    assert not network.called
    assert len(data.load_sessions(processed, "Nitschke")) == 1


def test_download_missing_recording_from_expected_dandi_asset(tmp_path, monkeypatch):
    client = MagicMock()
    asset = client.get_dandiset.return_value.get_asset_by_path.return_value
    asset.download_url = "https://example.invalid/recording.nwb"
    factory = MagicMock()
    factory.return_value.__enter__.return_value = client
    monkeypatch.setattr(data, "DandiAPIClient", factory)
    download = MagicMock(side_effect=lambda url, path: path.touch())
    monkeypatch.setattr(data, "download_with_progress", download)

    def process(pipeline, path):
        assert path.is_file()
        (pipeline.processed_dir / "jenkins_20090912_maze.h5").touch()

    monkeypatch.setattr(data.Pipeline, "process", process)
    data.download_and_preprocess(tmp_path / "raw", tmp_path / "processed", "Jenkins", 1)
    client.get_dandiset.assert_called_once_with("000070", "draft")
    client.get_dandiset.return_value.get_asset_by_path.assert_called_once_with(
        "sub-Jenkins/sub-Jenkins_ses-20090912_behavior+ecephys.nwb"
    )
    download.assert_called_once_with(
        asset.download_url, tmp_path / "raw/sub-Jenkins_ses-20090912_behavior+ecephys.nwb"
    )


def test_pipeline_missing_output_is_not_silently_accepted(tmp_path, monkeypatch):
    (tmp_path / "sub-Nitschke_ses-20090812_behavior+ecephys.nwb").touch()
    monkeypatch.setattr(data.Pipeline, "process", lambda self, path: None)
    with pytest.raises(FileNotFoundError, match="expected file"):
        data.download_and_preprocess(tmp_path, tmp_path / "processed", "Nitschke", 1)
    with pytest.raises(ValueError, match="nonnegative"):
        data.download_and_preprocess(tmp_path, tmp_path, "Nitschke", -1)


@pytest.mark.parametrize("current", [False, True])
def test_preparation_cli_selects_current_or_legacy_source(tmp_path, monkeypatch, current):
    folder = tmp_path / "data/processed"
    folder.mkdir(parents=True)
    legacy = folder / "nitschke_20090812_center_out_reaching.h5"
    legacy.touch()
    modern = folder / "nitschke_20090812_maze.h5"
    if current:
        modern.touch()
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["prepare", "--output", str(tmp_path / "output")])
    run = MagicMock()
    monkeypatch.setattr(prepare, "prepare", run)
    prepare.main()
    assert run.call_args.args[1] == (modern if current else legacy)
