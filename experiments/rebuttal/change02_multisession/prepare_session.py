"""Prepare a further Aeon session in the same format as the Section 4.3 data.

This is ``prepare_sweep_18h.py`` (saved with the session A data under
``/ceph/aeon/aeon/nldisco/aeon/sweeps/20260921_18h_s20/data/``) with the sorting
block, interval and output directory made arguments. Every processing step is
unchanged: the audited amplifier concatenation, the ONIX->HARP regressions with
the acquisition-source +1 second convention, masking of unsupported time and the
behavior binning. Two things differ, both forced by the environment:

* The ingestion database was unreachable, so the per-hour clock regressions are
  refitted from the raw ``HarpSync`` messages instead of being downloaded. For
  session A this reproduces all 20 saved coefficients and intercepts exactly
  (see ``check_refit_against_saved``).
* Checks that hard-coded session A's counts (90 clusters, 19 source files) are
  replaced by recording the observed values.

Run with the ingestion interpreter (``swc-aeon``, ``scikit-learn==1.7.2``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import pickle
import sys

SESSION_A_DATA = Path('/ceph/aeon/aeon/nldisco/aeon/sweeps/20260921_18h_s20/data')
sys.path.insert(0, str(SESSION_A_DATA))  # unchanged helpers saved with session A

from sanity_helpers import BEHAVIOR_EPOCH, EPHYS_EPOCH  # noqa: E402
from validate_single_chunk import (  # noqa: E402
    apply_models, residual_summary, recording_sources, read_acquisition_source,
)

HERE = SESSION_A_DATA
BLOCKS = Path('/ceph/aeon/aeon/dj_store/ephys-processed/social-ephys0.1-aeon3/'
              'BAA-1104292/insertion_1/ephys_blocks')
BIN_NS = 20_000_000
# Set by ``configure`` from the command line before any stage runs.
OUT = SORT_ROOT = START = END = NBINS = HARP_START = BLOCK_START = BLOCK_END = None
MIN_WINDOW_FRACTION = .99


def configure(block: str, start: str, end: str, out: str) -> None:
    """Set the module-level interval, sorting block and output directory."""
    global OUT, SORT_ROOT, START, END, NBINS, HARP_START, BLOCK_START, BLOCK_END
    OUT = Path(out)
    SORT_ROOT = BLOCKS / block / '0-95'
    START, END = pd.Timestamp(start), pd.Timestamp(end)
    BLOCK_START, BLOCK_END = (pd.Timestamp(part.replace('T', ' ').replace('-', ':', 4).replace(':', '-', 2))
                              for part in block.split('_'))
    assert BLOCK_START <= START < END <= BLOCK_END, 'Interval must lie inside the sorting block'
    NBINS = (END.value - START.value) // BIN_NS
    HARP_START = (START - pd.Timestamp('1904-01-01')).total_seconds()


def load_stream(epoch, device, stream_reader, start, end, required=True):
    """``sanity_helpers.load_stream`` with timestamps kept timezone-naive.

    Newer ``swc-aeon`` releases return UTC-aware datetimes; session A was prepared
    with naive HARP datetimes, so the timezone label is dropped (same instants).
    """
    from swc.aeon.io import api

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    files, frames = [], []
    for hour in pd.date_range(start.floor('h') - pd.Timedelta(hours=1), end.floor('h'), freq='h'):
        stamp = hour.strftime('%Y-%m-%dT%H-%M-%S')
        pattern = stream_reader.pattern.replace('*', stamp) + '.' + stream_reader.extension
        for path in sorted((Path(epoch) / device).glob(pattern)):
            frame = stream_reader.read(path)
            index = pd.DatetimeIndex(api.to_datetime(frame.index))
            frame.index = index.tz_localize(None) if index.tz is not None else index
            frame.index.name = 'time'
            frame = frame.loc[(frame.index >= start) & (frame.index < end)]
            if len(frame):
                frames.append(frame)
            files.append(str(path))
    result = pd.concat(frames) if frames else pd.DataFrame(
        columns=stream_reader.columns, index=pd.DatetimeIndex([], name='time'))
    if required and result.empty:
        raise ValueError(f'No samples for {device}/{stream_reader.pattern} in requested window')
    return result, files


def dump_json(path: Path, value: object) -> None:
    """Write a new analysis artifact with explicit JSON formatting."""
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def digest(path: Path) -> str:
    """Hash a source in bounded pieces."""
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def valid_window_starts(valid: np.ndarray, length: int = 20) -> np.ndarray:
    """Return windows whose entire support is observed; do not bridge gaps."""
    invalid = np.r_[0, np.cumsum(~valid, dtype=np.int64)]
    return np.flatnonzero(invalid[length:] == invalid[:-length])


def audit_raw_clock(clock: np.ndarray) -> tuple[dict, np.ndarray, list[tuple[int,int]]]:
    """Scan every clock word, conservatively excluding both sides of bad steps.

    Ordinary measured steps are roughly8317ticks. Isolated anomalous runs may
    be excluded only if their external neighbors retain ordinary elapsed time.
    A reset, dropout, or sustained discontinuity still stops preparation.
    """
    bad_steps, minimum, maximum = [], None, None
    for start in range(0,len(clock)-1,4_000_000):
        delta=np.diff(np.asarray(clock[start:start+4_000_001],dtype=np.int64))
        lo,hi=int(delta.min()),int(delta.max())
        minimum=lo if minimum is None else min(minimum,lo)
        maximum=hi if maximum is None else max(maximum,hi)
        bad=np.flatnonzero((delta<=8000)|(delta>=9000))
        if len(bad):
            bad_steps.extend((bad+start).tolist())
    step=np.asarray(bad_steps,dtype=np.int64)
    excluded=np.unique(np.r_[step,step+1])
    brackets=[]
    if len(excluded):
        groups=np.split(excluded,np.flatnonzero(np.diff(excluded)>1)+1)
        for group in groups:
            left,right=int(group[0]-1),int(group[-1]+1)
            assert left>=0 and right<len(clock) and len(group)<10
            elapsed=int(clock[right])-int(clock[left])
            assert 8000*(right-left)<elapsed<9000*(right-left), 'Clock discontinuity is not an isolated word anomaly'
            brackets.append((left,right))
    return dict(samples=len(clock),min_tick_step=minimum,max_tick_step=maximum,
                abnormal_transition_count=len(step),excluded_neighbor_samples=excluded.tolist(),
                excluded_sample_clock_values=[int(clock[i]) for i in excluded]),excluded,brackets


def fetch_sync_models() -> tuple[list[dict], dict]:
    """Refit the per-hour ONIX->HARP regressions from the raw HarpSync messages.

    The pipeline stores one ``LinearRegression`` per hourly sync file. Refitting
    from the same messages reproduces the stored session A models exactly.
    """
    import joblib
    from sklearn.linear_model import LinearRegression

    models = []
    first = BLOCK_START.floor('h') - pd.Timedelta(hours=1)
    for hour in pd.date_range(first, BLOCK_END.floor('h'), freq='h'):
        stamp = hour.strftime('%Y-%m-%dT%H-%M-%S')
        path = EPHYS_EPOCH / 'NeuropixelsV2Beta' / f'NeuropixelsV2Beta_HarpSync_{stamp}.csv'
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        x = frame['Value.Clock'].to_numpy(np.uint64)
        y = frame['Value.HarpTime'].to_numpy(float)
        model = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
        with tempfile.TemporaryDirectory(prefix='aeon-session-models-') as temporary:
            source = Path(temporary) / 'model.joblib'
            joblib.dump(model, source)
            source_hash = digest(source)
            destination = OUT / 'sync_models' / f'{source_hash}.joblib'
            if not destination.exists():
                shutil.copy2(source, destination)
        models.append(dict(start=int(x[0]), end=int(x[-1]), harp_start=str(hour), object=model,
                           sha256=source_hash, file=str(destination), sync_file=str(path)))
    models.sort(key=lambda row: row['start'])
    assert models and all(a['end'] < b['start'] for a, b in zip(models, models[1:]))
    return models, dict(warnings=[], model_count=len(models),
                        model_source='refitted from raw HarpSync messages (database unreachable); '
                                     'reproduces the stored session A models exactly')


def check_refit_against_saved() -> dict:
    """Compare refitted regressions with the session A models saved from the database."""
    from sklearn.linear_model import LinearRegression

    saved = json.loads((SESSION_A_DATA / 'sync_models.json').read_text())
    identical = 0
    for row in saved:
        stamp = pd.Timestamp(row['harp_start']).strftime('%Y-%m-%dT%H-%M-%S')
        frame = pd.read_csv(EPHYS_EPOCH / 'NeuropixelsV2Beta' / f'NeuropixelsV2Beta_HarpSync_{stamp}.csv')
        x = frame['Value.Clock'].to_numpy(np.uint64)
        model = LinearRegression().fit(x.reshape(-1, 1), frame['Value.HarpTime'].to_numpy(float).reshape(-1, 1))
        identical += int(int(x[0]) == row['start'] and int(x[-1]) == row['end']
                         and np.ravel(model.coef_).tolist() == row['coefficient']
                         and np.ravel(model.intercept_).tolist() == row['intercept'])
    return dict(session_a_models=len(saved), refit_identical=identical)


def audit_block_sync(models: list[dict]) -> tuple[list[dict], pd.DataFrame, list[tuple]]:
    """Audit available anchors and explicitly identify missing-heartbeat spans."""
    summaries, residuals, gaps = [], [], []
    for index, model in enumerate(models):
        stamp = pd.Timestamp(model['harp_start']).strftime('%Y-%m-%dT%H-%M-%S')
        path = EPHYS_EPOCH/'NeuropixelsV2Beta'/f'NeuropixelsV2Beta_HarpSync_{stamp}.csv'
        frame = pd.read_csv(path)
        assert frame.notna().all().all()
        x = frame['Value.Clock'].to_numpy(np.uint64)
        payload = frame['Value.HarpTime'].to_numpy(float)
        acquired = frame['Seconds'].to_numpy(float)
        assert int(x[0]) == model['start'] and int(x[-1]) == model['end']
        assert np.all(np.diff(x.astype(np.int64)) > 0)
        spacing = np.diff(acquired)
        assert np.all(spacing >= 1) and np.all(spacing == np.round(spacing))
        assert np.all(acquired-payload == 1)
        predicted = model['object'].predict(x.reshape(-1,1)).ravel()+1
        elapsed = (x.astype(np.int64)-int(x[0])).astype(float)
        target = acquired-acquired[0]
        train = (target.astype(np.int64)//30)%2 == 0
        design = np.c_[elapsed,np.ones(len(x))]
        coef = np.linalg.lstsq(design[train],target[train],rcond=None)[0]
        heldout = (design@coef-target)[~train]
        missing = np.flatnonzero(spacing > 1)
        for row in missing:
            gaps.append((float(acquired[row]-HARP_START),float(acquired[row+1]-HARP_START)))
        summaries.append(dict(model=index,file=str(path),raw_messages=len(frame),
                              missing_heartbeats=int(np.sum(spacing-1)),
                              saved_plus_one_vs_acquisition=residual_summary(predicted-acquired),
                              heldout_vs_acquisition=residual_summary(heldout)))
        residuals.append(pd.DataFrame(dict(model=index,onix_clock=x,
                                          acquisition_seconds=acquired,
                                          corrected_residual_us=(predicted-acquired)*1e6,
                                          heldout=~train,
                                          heldout_refit_residual_us=(design@coef-target)*1e6)))
    return summaries,pd.concat(residuals,ignore_index=True),gaps


def prepare_neural() -> dict:
    """Map all original KS4 spikes, crop, bin, and mask incomplete support."""
    (OUT / 'sync_models').mkdir(parents=True, exist_ok=True)
    conversion = read_acquisition_source('workflows/ephys/Extensions/ConvertToHarpTimestamp.cs')
    assert 'source.Select(value => (double)(value + 1))' in conversion
    models, metadata = fetch_sync_models()
    summaries, residuals, sync_gaps = audit_block_sync(models)
    residuals.to_parquet(OUT / 'sync_residuals.parquet', index=False)
    max_fit = max(r['saved_plus_one_vs_acquisition']['max_abs_us'] for r in summaries)
    max_holdout = max(r['heldout_vs_acquisition']['max_abs_us'] for r in summaries)
    # Prospective gate for this 20-ms analysis, not a revision of the earlier
    # failed 100-us validation criterion or a physical latency certification.
    assert max_fit < 1000 and max_holdout < 1000, (max_fit, max_holdout)
    dump_json(OUT / 'sync_validation.json', dict(
        summaries=summaries, max_saved_fit_residual_us=max_fit,
        max_heldout_residual_us=max_holdout, analysis_gate_us=1000,
        missing_heartbeat_brackets_relative_seconds=sync_gaps,
        physical_latency_independently_verified=False,
        previous_100us_criterion_not_reclassified=True))
    dump_json(OUT / 'sync_models.json', [
        {k: v for k, v in row.items() if k != 'object'} |
        dict(coefficient=np.asarray(row['object'].coef_).ravel().tolist(),
             intercept=np.asarray(row['object'].intercept_).ravel().tolist()) for row in models])
    sorter = SORT_ROOT / 'kilosort4_1' / 'spike_sorting' / 'sorter_output'
    labels = pd.read_csv(sorter / 'cluster_KSLabel.tsv', sep='\t').sort_values('cluster_id')
    ids = labels.cluster_id.to_numpy(np.int64)
    assert set(labels.KSLabel) <= {'mua', 'good'}
    spike_samples = np.load(sorter / 'spike_times.npy', mmap_mode='r').reshape(-1)
    clusters = np.load(sorter / 'spike_clusters.npy', mmap_mode='r').reshape(-1)
    assert len(spike_samples) == len(clusters)
    for left in range(0, len(spike_samples) - 1, 2_000_000):
        assert np.all(np.diff(spike_samples[left:left + 2_000_001].astype(np.int64)) >= 0)
    recording = SORT_ROOT / 'recording' / 'si_recording.pkl'
    with recording.open('rb') as handle:
        sources = recording_sources(pickle.load(handle))  # trusted pipeline metadata
    refit_check = check_refit_against_saved()
    assert refit_check['refit_identical'] == refit_check['session_a_models']
    counts = np.lib.format.open_memmap(OUT / 'counts.npy', mode='w+', dtype=np.uint16,
                                       shape=(NBINS, len(ids)))
    counts[:] = 0
    neural_valid = np.zeros(NBINS, dtype=bool)
    manifest, coverage, offset, total_mapped, total_cropped = [], [], 0, 0, 0
    clock_anomaly_gaps=[]
    unsupported_spikes = 0
    previous_clock = None
    reconstruction_dir = OUT / 'reconstructed_spikes'
    reconstruction_dir.mkdir(exist_ok=True)
    for number, (relative, kwargs) in enumerate(sources):
        amp_path = (SORT_ROOT / 'kilosort4_1' / relative).resolve()
        clock_path = amp_path.with_name(amp_path.name.replace('AmplifierData', 'Clock'))
        samples = amp_path.stat().st_size // (384 * 2)
        assert amp_path.stat().st_size == samples * 384 * 2
        assert clock_path.stat().st_size == samples * 8 and kwargs['sampling_frequency'] == 30000
        clock = np.memmap(clock_path, mode='r', dtype=np.uint64)
        clock_summary, excluded_samples, anomaly_brackets = audit_raw_clock(clock)
        for left,right in anomaly_brackets:
            seconds,_=apply_models(np.asarray([clock[left],clock[right]]),models)
            assert np.isfinite(seconds).all()
            a,b=seconds+1-HARP_START
            clock_anomaly_gaps.append((float(a),float(b)))
        if previous_clock is not None:
            assert 8000 < int(clock[0]) - previous_clock < 9000
        previous_clock = int(clock[-1])
        lo, hi = np.searchsorted(spike_samples, [offset, offset + samples])
        local = spike_samples[lo:hi].astype(np.int64) - offset
        original_ids = np.asarray(clusters[lo:hi], dtype=np.int64)
        columns = np.searchsorted(ids, original_ids)
        assert np.all(columns < len(ids)) and np.array_equal(ids[columns], original_ids)
        mapped, assignment = apply_models(np.asarray(clock[local]), models)
        clock_rejected=np.isin(local,excluded_samples) if len(excluded_samples) else np.zeros(len(local),dtype=bool)
        mapped[clock_rejected]=np.nan
        assignment[clock_rejected]=-1
        acquired = mapped + 1.0
        relative_s = acquired - HARP_START
        supported = np.isfinite(relative_s)
        in_range = supported & (relative_s >= 0) & (relative_s < (END - START).total_seconds())
        bins = np.floor(relative_s[in_range] / .02).astype(np.int64)
        np.add.at(counts, (bins, columns[in_range]), 1)
        total_mapped += int(supported.sum())
        total_cropped += int(in_range.sum())
        unsupported_spikes += int((~supported).sum())
        # Preserve all original spike identities, including unsupported NaNs,
        # for replay without contacting the DB or rereading multi-GB clocks.
        np.savez(reconstruction_dir / f'chunk_{number:02d}.npz',
                 original_sample=spike_samples[lo:hi], cluster_id=original_ids.astype(np.int16),
                 relative_seconds=relative_s, model_id=assignment)
        for model_index, model in enumerate(models):
            low_tick, high_tick = max(int(clock[0]), model['start']), min(int(clock[-1]), model['end'])
            if low_tick > high_tick:
                continue
            a, b = model['object'].predict(np.asarray([[low_tick], [high_tick]], dtype=np.uint64)).ravel() + 1 - HARP_START
            # Include one sample's duration, but never fill a saved-model seam.
            b += 1 / 30000
            coverage.append((float(a), float(b), number, model_index))
        manifest.append(dict(amplifier=str(amp_path), clock=str(clock_path),
                             sample_offset=offset, samples=samples, original_spikes=int(hi-lo),
                             mapped_spikes=int(supported.sum()), strict_interval_spikes=int(in_range.sum()),
                             unsupported_spikes=int((~supported).sum()), clock_qc=clock_summary))
        offset += samples
        print(f'Neural chunk {number+1}/{len(sources)}: {hi-lo:,} spikes, {in_range.sum():,} in strict interval', flush=True)
    assert spike_samples[-1] < offset
    assert (SORT_ROOT / 'recording' / 'recording.dat').stat().st_size == offset * 96 * 2
    # Union contiguous raw-file coverage first, avoiding artificial hourly gaps.
    merged = []
    for a, b, _, _ in sorted(coverage):
        if merged and a <= merged[-1][1] + 1e-5:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    for a, b in merged:
        first = max(0, int(np.ceil(a / .02)))
        stop = min(NBINS, int(np.floor(b / .02)))
        if stop > first:
            neural_valid[first:stop] = True
    for a, b in sync_gaps+clock_anomaly_gaps:
        # Mask bins touching the whole bracketing interval, despite a smooth
        # saved regression. Do not infer a missing spike or a neural silence.
        first=max(0,int(np.floor(a/.02)))
        stop=min(NBINS,int(np.ceil(b/.02)))
        neural_valid[first:stop]=False
    starts = valid_window_starts(neural_valid)
    assert len(starts) > MIN_WINDOW_FRACTION * (NBINS - 19)
    assert int(counts.sum(dtype=np.uint64)) == total_cropped
    assert counts.max() < 65535
    counts.flush()
    np.save(OUT / 'cluster_ids.npy', ids)
    np.save(OUT / 'neural_valid.npy', neural_valid)
    np.save(OUT / 'valid_starts.npy', starts)
    labels['spikes_in_strict_interval'] = counts.sum(axis=0, dtype=np.uint64)
    labels['spikes_in_valid_bins'] = counts[neural_valid].sum(axis=0, dtype=np.uint64)
    labels['mean_rate_hz_valid'] = labels.spikes_in_valid_bins / (neural_valid.sum() * .02)
    assert (labels.spikes_in_valid_bins > 0).all()
    labels.to_csv(OUT / 'cluster_metadata.csv', index=False)
    dump_json(OUT / 'recording_manifest.json', manifest)
    pd.DataFrame(coverage, columns=['start_relative_s', 'end_relative_s', 'file_index', 'model_index']).to_csv(OUT / 'supported_intervals.csv', index=False)
    metadata.update(total_sorter_spikes=len(spike_samples), mapped_source_spikes=total_mapped,
                    refit_check=refit_check, source_files=len(sources),
                    n_clusters=len(ids), ks_labels=labels.KSLabel.value_counts().to_dict(),
                    clock_anomaly_brackets_relative_seconds=clock_anomaly_gaps,
                    clock_anomaly_policy='exclude both endpoints of abnormal steps; no raw-clock repairs',
                    source_spikes_outside_model_support=unsupported_spikes,
                    strict_interval_spikes=total_cropped, valid_bins=int(neural_valid.sum()),
                    invalid_bins=int((~neural_valid).sum()), valid_windows=len(starts),
                    source_sha256={str(p): digest(p) for p in [recording, sorter/'cluster_KSLabel.tsv',
                                                              sorter/'spike_times.npy', sorter/'spike_clusters.npy']})
    dump_json(OUT / 'neural_summary.json', metadata)
    return metadata


def trailing_mean(values: np.ndarray, width: int) -> np.ndarray:
    """Causal descriptive smoothing requiring every sample to be valid."""
    return pd.Series(values).rolling(width, min_periods=width).mean().to_numpy()


def backward_difference(values: np.ndarray, lag: int) -> np.ndarray:
    """Causal difference requiring all intermediate values to be finite."""
    result = np.full(len(values), np.nan)
    result[lag:] = (values[lag:] - values[:-lag]) / (lag * .02)
    continuous = pd.Series(np.isfinite(values).astype(int)).rolling(lag+1, min_periods=lag+1).sum().to_numpy() == lag+1
    result[~continuous] = np.nan
    return result


def prepare_behavior() -> dict:
    """Bin observed behavior; preserve missingness and explicit units."""
    from swc.aeon.io import reader
    arrays = {name: np.full(NBINS, np.nan) for name in ['x','y','area','ellipse_angle','major','minor',
                                                       'wheel_signed_speed']}
    camera_counts = np.zeros(NBINS, dtype=np.int16)
    wheel_counts = np.zeros(NBINS, dtype=np.int16)
    source_files, triggers, detections = [], [], []
    for hour_start in pd.date_range(START, END, freq='h', inclusive='left'):
        hour_end = hour_start + pd.Timedelta(hours=1)
        position, files = load_stream(BEHAVIOR_EPOCH, 'CameraTop', reader.Position('CameraTop_200_*'), hour_start, hour_end)
        source_files.extend(files)
        times = position.index.asi8
        assert np.all(np.diff(times) > 0)
        bins = ((times - START.value) // BIN_NS).astype(np.int64)
        valid = np.isfinite(position[['x','y','area']].to_numpy()).all(axis=1) & (position.area.to_numpy() > 0)
        b = bins[valid]
        unique, inverse, n = np.unique(b, return_inverse=True, return_counts=True)
        camera_counts[unique] = n
        for name in ['x','y','area','major','minor']:
            arrays[name][unique] = np.bincount(inverse, weights=position[name].to_numpy()[valid]) / n
        arrays['ellipse_angle'][unique] = np.bincount(inverse, weights=position['angle'].to_numpy()[valid]) / n
        wheel, files = load_stream(BEHAVIOR_EPOCH, 'Patch2', reader.Encoder('Patch2_90_*'),
                                   hour_start - pd.Timedelta(milliseconds=5), hour_end)
        source_files.extend(files)
        angle = wheel.angle.to_numpy(np.int64)
        assert np.all((angle >= 0) & (angle < 16384))
        dt = np.diff(wheel.index.asi8) / 1e9
        assert np.all(dt > 0)
        delta = ((np.diff(angle) + 8192) % 16384 - 8192) / 16384.0
        b = ((wheel.index.asi8[1:] - START.value) // BIN_NS).astype(np.int64)
        valid = (dt <= .004) & (b >= (hour_start.value-START.value)//BIN_NS) & (b < (hour_end.value-START.value)//BIN_NS)
        unique, inverse, n = np.unique(b[valid], return_inverse=True, return_counts=True)
        wheel_counts[unique] = n
        velocity = np.bincount(inverse, weights=delta[valid]) / .02
        velocity[n < 8] = np.nan
        arrays['wheel_signed_speed'][unique] = velocity
        for pattern, mask, label, output in [('Patch2_35_*',1,'TriggerPellet',triggers),
                                             ('Patch2_32_*',0x22,'PelletDetected',detections)]:
            events, files = load_stream(BEHAVIOR_EPOCH, 'Patch2', reader.BitmaskEvent(pattern,mask,label), hour_start,hour_end,required=False)
            source_files.extend(files)
            output.extend(events.index.tolist())
        print(f'Behavior hour {hour_start}: {len(position):,} camera frames, {len(wheel):,} wheel samples', flush=True)
    metadata = json.loads((BEHAVIOR_EPOCH/'Metadata.yml').read_text())
    region = next(value['Patch2Region']['ArrayOfPoint'] for value in metadata.values()
                  if isinstance(value,dict) and 'Patch2Region' in value)
    polygon = np.asarray([[float(p['X']),float(p['Y'])] for p in region])
    center = polygon.mean(axis=0)
    frame = pd.DataFrame(arrays)
    frame['time'] = START + pd.to_timedelta(np.arange(NBINS)*BIN_NS+BIN_NS//2,unit='ns')
    frame['bin_index'] = np.arange(NBINS)
    frame['camera_valid'] = (camera_counts > 0) & np.isfinite(frame[['x','y','area']]).all(axis=1)
    frame['camera_sample_count'] = camera_counts
    frame['wheel_valid'] = wheel_counts >= 8
    frame['wheel_sample_count'] = wheel_counts
    # Camera sampled at50Hz. Causal100ms averaging then100ms differences reduce
    # frame-to-frame centroid jitter. Unsmooth coordinates remain available.
    x, y = trailing_mean(arrays['x'],5), trailing_mean(arrays['y'],5)
    vx, vy = backward_difference(x,5), backward_difference(y,5)
    speed = np.hypot(vx,vy)
    frame['vx'], frame['vy'], frame['speed'] = vx,vy,speed
    frame['acceleration'] = backward_difference(speed,5)
    heading = np.arctan2(vy,vx)
    heading[(speed < 20) | ~np.isfinite(speed)] = np.nan
    frame['heading'] = heading
    turn = np.full(NBINS,np.nan)
    delta_heading = np.angle(np.exp(1j*(heading[5:]-heading[:-5]))) / .1
    turn[5:] = delta_heading
    heading_valid = pd.Series(np.isfinite(heading).astype(int)).rolling(6,min_periods=6).sum().to_numpy() == 6
    turn[~heading_valid] = np.nan
    frame['angular_velocity'] = turn
    distance = np.hypot(arrays['x']-center[0],arrays['y']-center[1])
    frame['patch_distance'] = distance
    frame['radial_velocity'] = backward_difference(trailing_mean(distance,5),5)
    toward = np.arctan2(center[1]-y,center[0]-x)
    frame['heading_error_to_patch'] = np.angle(np.exp(1j*(heading-toward)))
    frame['area_derivative'] = backward_difference(trailing_mean(arrays['area'],5),5)
    frame['wheel_speed_raw'] = np.abs(arrays['wheel_signed_speed'])
    frame['wheel_speed'] = trailing_mean(np.abs(arrays['wheel_signed_speed']),5)
    frame['wheel_acceleration'] = backward_difference(frame.wheel_speed.to_numpy(),5)
    frame.to_parquet(OUT/'behavior.parquet',index=False)
    event_summary = {}
    for name, values in [('pellet_times',triggers),('pellet_detection_times',detections)]:
        times = pd.DatetimeIndex(values)
        event_frame = pd.DataFrame({'time':times,'relative_seconds':(times-START).total_seconds()})
        event_frame.to_csv(OUT/f'{name}.csv',index=False)
        event_summary[name] = len(values)
    summary = dict(**event_summary, patch_center_px=center.tolist(), patch_polygon_px=polygon.tolist(),
                   camera_valid_fraction=float(frame.camera_valid.mean()),
                   wheel_valid_fraction=float(frame.wheel_valid.mean()),
                   speed_valid_fraction=float(np.isfinite(speed).mean()),
                   camera_bins_with_multiple_frames=int((camera_counts>1).sum()),
                   movement_heading_min_speed_px_s=20,
                   derivative_convention='causal100ms mean then100ms backward difference; no cross-NaN interpolation',
                   area_units='pixels_squared (segmented image area)',
                   wheel_speed_units='turns/s; absolute net14bit encoder increment per20ms then causal100ms mean',
                   heading_units='radians, atan2(vy,vx) in image coordinates (y down); not anatomical head direction',
                   radial_velocity_sign='negative=toward patch',
                   source_files=sorted(set(source_files)))
    dump_json(OUT/'behavior_summary.json',summary)
    return summary


def add_behavior_qc() -> None:
    """Flag gross tracking artifacts without erasing raw behavioral measurements."""
    from scipy.ndimage import maximum_filter1d
    frame=pd.read_parquet(OUT/'behavior.parquet')
    steps=np.r_[np.nan,np.hypot(np.diff(frame.x),np.diff(frame.y))]
    jump=steps>100
    implausible_area=frame.area.to_numpy()>5000
    direct=jump|implausible_area
    expanded=maximum_filter1d(direct.astype(np.uint8),size=41,mode='constant')>0
    frame['tracking_suspect']=expanded
    frame['camera_analysis_valid']=frame.camera_valid & ~expanded
    frame.to_parquet(OUT/'behavior.parquet',index=False)
    summary=json.loads((OUT/'behavior_summary.json').read_text())
    summary.update(tracking_qc=dict(
        criterion='>100px centroid step between adjacent20ms bins or area>5000px²; ±400ms exclusion',
        interpretation='conservative gross-artifact screen, not a calibrated speed/posture boundary',
        jump_bins=int(jump.sum()),large_area_bins=int(implausible_area.sum()),
        expanded_suspect_bins=int(expanded.sum()),
        camera_analysis_valid_fraction=float(frame.camera_analysis_valid.mean())))
    dump_json(OUT/'behavior_summary.json',summary)


def main() -> None:
    """Run named stages; readiness requires an explicit final QC stage."""
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['neural','behavior','behavior_qc','ready'])
    parser.add_argument('--block',required=True,help='sorting block directory name')
    parser.add_argument('--start',required=True)
    parser.add_argument('--end',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--min-wheel-valid',type=float,default=.99)
    parser.add_argument('--min-window-fraction',type=float,default=.99)
    args=parser.parse_args()
    configure(args.block,args.start,args.end,args.out)
    global MIN_WINDOW_FRACTION
    MIN_WINDOW_FRACTION=args.min_window_fraction
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='neural':
        prepare_neural()
    elif args.stage=='behavior':
        prepare_behavior()
    elif args.stage=='behavior_qc':
        add_behavior_qc()
    else:
        neural=json.loads((OUT/'neural_summary.json').read_text())
        behavior=json.loads((OUT/'behavior_summary.json').read_text())
        assert behavior['camera_valid_fraction']>.9 and behavior['wheel_valid_fraction']>args.min_wheel_valid
        assert neural['valid_windows']>args.min_window_fraction*(NBINS-19)
        summary=dict(interval_start=str(START),interval_end=str(END),interval='half-open',
                     sorting_block=str(SORT_ROOT),
                     clock='HARP acquisition datetimes; timezone not independently established',
                     bin_seconds=.02,seq_len=20,n_clusters=neural['n_clusters'],n_bins=NBINS,
                     acquisition_offset_seconds=1,neural=neural,behavior=behavior,
                     region_assignments='unknown',
                     prepared_like='/ceph/aeon/aeon/nldisco/aeon/sweeps/20260921_18h_s20/data (session A)')
        dump_json(OUT/'metadata.json',summary)
        for source_name in ['sanity_helpers.py','validate_single_chunk.py']:
            shutil.copy2(HERE/source_name,OUT/source_name)
        dump_json(OUT/'data_hashes.json',{name:digest(OUT/name) for name in [
            'counts.npy','cluster_ids.npy','neural_valid.npy','valid_starts.npy',
            'behavior.parquet','cluster_metadata.csv','pellet_times.csv',
            'sync_models.json','recording_manifest.json']})
        (OUT/'README.md').write_text(
            f'# Aeon session prepared in the session A format\n\n'
            f'Interval: {START} inclusive to {END} exclusive, recorded acquisition HARP '
            f'convention. Sorting block: {SORT_ROOT}.\n\n'
            f'`counts.npy` is [{NBINS:,} bins,{neural["n_clusters"]} raw KS4 clusters], 20 ms uint16 '
            f'counts ({neural["ks_labels"]}). `cluster_ids.npy` maps columns to original cluster '
            'identifiers. `neural_valid.npy` excludes any partially unsupported bin and '
            '`valid_starts.npy` admits only 20-bin windows with complete neural support. '
            '`behavior.parquet` has one row per bin with the same columns and definitions as '
            'session A (see behavior_summary.json). Prepared by prepare_session.py, the session A '
            'script with the block and interval as arguments; clock regressions were refitted from '
            'the raw HarpSync messages, which reproduces the stored session A models exactly.\n')
        dump_json(OUT/'PREPARED.json',dict(created_utc=str(pd.Timestamp.now(tz='UTC')),
                                         metadata_sha256=digest(OUT/'metadata.json')))
    shutil.copy2(Path(__file__),OUT/'prepare_session.py')


if __name__=='__main__':
    main()
