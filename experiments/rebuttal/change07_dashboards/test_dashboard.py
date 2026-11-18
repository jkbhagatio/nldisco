"""Fast checks of the dashboard statistics and rendering on small synthetic data."""

import matplotlib.pyplot as plt
import numpy as np

from experiments.rebuttal.change07_dashboards import dashboard as db


def toy(n: int = 4000, seed: int = 0) -> db.Dataset:
    """Latent 0 is active only when x > 0.5; latent 1 is random; latent 2 is dead."""
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    x[:10] = np.nan
    acts = np.zeros((n, 3), np.float32)
    acts[:, 0] = np.where(x > .5, 1 + x, 0)
    acts[:, 1] = rng.uniform(0, 1, n) * (rng.uniform(0, 1, n) < .2)
    edges = np.linspace(0, 1, 5)
    return db.Dataset(
        title="Toy", activations=acts, bin_seconds=.1,
        segments=db.segment_ids(np.r_[np.arange(n // 2), np.arange(n // 2) + n]),
        variables=(db.Variable("x", x, edges), db.Variable("y", y, edges, log=True)),
        maps=(db.Map("xy", x, y, edges, edges, "x", "y", marker=(.5, .5, "mid")),
              db.Map("points", (y > .5).astype(float), y, edges, edges,
                     points=np.array([[0., 0.], [1., 1.]]))),
        timeline=db.Variable("time", np.arange(n) * .1, np.linspace(0, n * .1, 9)),
        traces={"x": x, "y": y}, events={"start": db.as_flags(n, np.array([0, n // 2, -1]))},
        episode_labels=np.arange(n).astype(str), half_width=20, n_episodes=4, min_count=5)


def test_digitize_handles_edges_and_invalid_values():
    edges = np.array([0., 1., 2.])
    values = np.array([-.1, 0., .99, 1., 2., 2.1, np.nan])
    np.testing.assert_array_equal(db.digitize(values, edges), [-1, 0, 0, 1, 1, -1, -1])


def test_binned_activity_matches_a_direct_loop():
    rng = np.random.default_rng(1)
    acts = (rng.uniform(0, 1, (500, 4)) * (rng.uniform(0, 1, (500, 4)) < .3)).astype(np.float32)
    bins = rng.integers(-1, 4, 500)
    counts, active, mean = db.binned_activity(acts, bins, 5)
    for b in range(4):
        rows = bins == b
        assert counts[b] == rows.sum()
        np.testing.assert_allclose(active[b], (acts[rows] > 0).mean(0), rtol=1e-5)
        np.testing.assert_allclose(mean[b], acts[rows].mean(0), rtol=1e-5)
    assert counts[4] == 0 and np.isnan(active[4]).all()


def test_top_episodes_are_ordered_and_disjoint():
    z = np.zeros(1000)
    z[[100, 105, 400, 900]] = [5, 4, 3, 2]
    np.testing.assert_array_equal(db.top_episodes(z, 5, 10), [100, 400, 900])
    assert len(db.top_episodes(np.zeros(50), 3, 5)) == 0


def test_episode_bounds_stay_inside_segments():
    segments = db.segment_ids(np.array([0, 1, 2, 3, 10, 11, 12, 13]))
    np.testing.assert_array_equal(segments, [0, 0, 0, 0, 1, 1, 1, 1])
    bounds = db.episode_bounds(np.array([1, 4, 7]), segments, 2)
    np.testing.assert_array_equal(bounds, [[0, 4], [4, 7], [5, 8]])
    grouped = db.segment_ids(np.arange(4), np.array([0, 0, 1, 1]))
    np.testing.assert_array_equal(grouped, [0, 0, 1, 1])


def test_trailing_slope_recovers_a_line():
    values = 3. * np.arange(5) * .05 + np.array([[1.], [2.]])
    np.testing.assert_allclose(db.trailing_slope(values, .05), [3., 3.])


def test_dashboard_recovers_the_planted_tuning_and_writes_every_active_view(tmp_path):
    board = db.Dashboard(toy())
    np.testing.assert_array_equal(board.active_latents, [0, 1])
    _, active, _ = board.tuning[0]  # latent 0 against x, in four bins
    np.testing.assert_allclose(active[:, 0], [0, 0, 1, 1])
    assert np.all(np.abs(active[:, 1] - .2) < .08)
    fig = board.figure(0, letters=True)
    assert len(fig.axes) > 10
    plt.close(fig)
    index = db.write_dashboard(board, tmp_path, dpi=40)
    assert sorted(p.name for p in (tmp_path / "views").iterdir()) == [
        "latent_0000.png", "latent_0001.png"]
    assert "latents = [0, 1]" in index.read_text()
