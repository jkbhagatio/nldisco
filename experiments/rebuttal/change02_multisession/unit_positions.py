"""Probe positions of Kilosort clusters, used to align units across sessions.

Each recording block is sorted separately, so cluster identities differ between
sessions. Both sessions use the same probe insertion, so a cluster's position on
the probe (mean Kilosort spike position) gives a behavior-free correspondence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from beartype import beartype

BLOCKS = Path("/ceph/aeon/aeon/dj_store/ephys-processed/social-ephys0.1-aeon3/"
              "BAA-1104292/insertion_1/ephys_blocks")


@beartype
def cluster_positions(block: str, cluster_ids: np.ndarray) -> pd.DataFrame:
    """Return mean spike position, peak channel and waveform width for each cluster.

    Args:
        block: Sorting block directory name.
        cluster_ids: Cluster identifiers in the column order of ``counts.npy``.

    Returns:
        One row per cluster, in the given order, with probe coordinates in um.
    """
    sorter = BLOCKS / block / "0-95" / "kilosort4_1" / "spike_sorting" / "sorter_output"
    clusters = np.load(sorter / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    positions = np.load(sorter / "spike_positions.npy", mmap_mode="r")
    templates = np.load(sorter / "spike_templates.npy", mmap_mode="r").reshape(-1)
    size = int(max(clusters.max(), cluster_ids.max())) + 1
    n = np.zeros(size)
    total = np.zeros((size, 2))
    votes = np.zeros((size, int(templates.max()) + 1), dtype=np.int64)
    for start in range(0, len(clusters), 8_000_000):
        c = np.asarray(clusters[start:start + 8_000_000], dtype=np.int64)
        p = np.asarray(positions[start:start + 8_000_000], dtype=np.float64)
        n += np.bincount(c, minlength=size)
        for axis in range(2):
            total[:, axis] += np.bincount(c, weights=p[:, axis], minlength=size)
        np.add.at(votes, (c, np.asarray(templates[start:start + 8_000_000], dtype=np.int64)), 1)
    shapes = np.load(sorter / "templates.npy")  # [template, sample, channel], whitened
    channel_xy = np.load(sorter / "channel_positions.npy")
    main = votes[cluster_ids].argmax(axis=1)
    peak = np.abs(shapes[main]).max(axis=1).argmax(axis=1)
    wave = shapes[main, :, peak]
    labels = pd.read_csv(sorter / "cluster_KSLabel.tsv", sep="\t").set_index("cluster_id")
    return pd.DataFrame(dict(
        cluster_id=cluster_ids, ks_label=labels.KSLabel.reindex(cluster_ids).to_numpy(),
        n_spikes=n[cluster_ids].astype(np.int64),
        x_um=total[cluster_ids, 0] / n[cluster_ids], y_um=total[cluster_ids, 1] / n[cluster_ids],
        peak_channel=peak, peak_channel_x_um=channel_xy[peak, 0], peak_channel_y_um=channel_xy[peak, 1],
        trough_to_peak_samples=wave.argmax(axis=1) - wave.argmin(axis=1),
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", required=True)
    parser.add_argument("--cluster-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cluster_positions(args.block, np.load(args.cluster_ids)).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
