"""Within-window behavioral tuning for the two illustrative NLDisco latents."""

import numpy as np
from beartype import beartype


@beartype
def activity_grid(x: np.ndarray, y: np.ndarray, active: np.ndarray,
                  trials: np.ndarray, xedges: np.ndarray, yedges: np.ndarray) -> tuple:
    """Pool windows per cell, requiring 20 windows and 10 distinct trials."""
    counts = np.histogram2d(y, x, bins=(yedges, xedges))[0]
    hits = np.histogram2d(y, x, bins=(yedges, xedges), weights=active.astype(float))[0]
    nx, ny = len(xedges) - 1, len(yedges) - 1
    ix = np.searchsorted(xedges, x, side='right') - 1
    iy = np.searchsorted(yedges, y, side='right') - 1
    ix[x == xedges[-1]], iy[y == yedges[-1]] = nx - 1, ny - 1
    valid = np.isfinite(x) & np.isfinite(y) & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    pairs = np.unique(np.column_stack((trials[valid], iy[valid] * nx + ix[valid])), axis=0)
    support = np.bincount(pairs[:, 1].astype(int), minlength=nx * ny).reshape(ny, nx)
    rate = np.divide(hits, counts, out=np.full_like(counts, np.nan),
                     where=(counts >= 20) & (support >= 10))
    return rate, counts, support


@beartype
def slope_history(speed: np.ndarray) -> tuple:
    """Trailing five-bin slopes ending at -250,...,0 ms, inside ten-bin input."""
    if speed.ndim != 2 or speed.shape[1] != 10:
        raise ValueError('Expected one ten-bin speed history per endpoint.')
    times = np.arange(5) * .05
    times -= times.mean()
    slopes = np.column_stack([speed[:, j-4:j+1] @ times / (times @ times)
                              for j in range(4, 10)])
    return (np.arange(4, 10) - 9) * .05, slopes


def draw_heatmap(ax, name, speed, vx, vy, active, trials):
    """Draw the approved interpretation panel, with all targets pooled."""
    import matplotlib.pyplot as plt

    if name == 'recent_braking':
        lag, slopes = slope_history(speed)
        limit = float(np.ceil(np.quantile(np.abs(slopes), .995) / 500) * 500)
        yedges = np.linspace(-limit, limit, 41)
        xedges = np.r_[lag - .025, lag[-1] + .025]
        columns = [activity_grid(np.full(len(trials), t), slopes[:, i], active, trials,
                                 np.array([t-.025, t+.025]), yedges)
                   for i, t in enumerate(lag)]
        rate, counts, support = [np.concatenate([r[i] for r in columns], axis=1)
                                 for i in range(3)]
        ax.set(xlim=(-.2, 0), xlabel='Time before endpoint (s)',
               ylabel=r'Acceleration (native units/s$^2$)')
        ax.set_xticks([-.2, -.15, -.1, -.05, 0], ['−0.20', '−0.15', '−0.10', '−0.05', '0.00'])
    elif name == 'fast_target_specific':
        direction = np.mod(np.degrees(np.arctan2(vy, vx)), 360)
        direction = np.where((vx != 0) | (vy != 0), direction, np.nan)
        limit = float(np.ceil(np.quantile(speed[:, -1], .995) / 100) * 100)
        xedges, yedges = np.arange(0, 361, 15), np.arange(0, limit + 1, 100)
        rate, counts, support = activity_grid(direction, speed[:, -1], active,
                                              trials, xedges, yedges)
        ax.set(xlim=(0, 360), xlabel='Endpoint velocity direction',
               ylabel='Speed (native units/s)')
        ax.set_xticks([0, 90, 180, 270, 360],
                       ['0°\nRight', '90°\nUp', '180°\nLeft', '270°\nDown', '360°\nRight'])
    else:
        raise ValueError(f'No interpretation heatmap for {name}.')
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#eeeeee')
    im = ax.pcolormesh(xedges, yedges, rate, cmap=cmap, vmin=0, vmax=1, shading='flat')
    bar = ax.figure.colorbar(im, ax=ax, fraction=.035, pad=.025, ticks=[0, .5, 1])
    bar.set_label('P(latent active)', fontsize=6, labelpad=3)
    bar.ax.tick_params(labelsize=6)
    return dict(xedges=xedges.tolist(), yedges=yedges.tolist(), rate=rate.tolist(),
                window_counts=counts.tolist(), trial_counts=support.tolist(),
                target_restriction=None, minimum_windows=20, minimum_trials=10,
                activity='native z > 0 at endpoint', xlimits=list(ax.get_xlim()))
