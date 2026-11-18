"""Build a latent dashboard, and optionally export static views for the paper.

Run from the repository root, for example::

    python -m experiments.rebuttal.change07_dashboards.build --dataset aeon \
        --out /ceph/aeon/aeon/nldisco/rebuttal_20261118/change07_dashboards/dashboards/aeon \
        --export 52 --figures paper/iclr_paper/figures/rebuttal/change07_dashboards

Open `<out>/index.html` and step through latents with the arrow keys.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.rebuttal.change07_dashboards.dashboard import Dashboard, write_dashboard
from experiments.rebuttal.change07_dashboards.datasets import LOADERS


def main() -> None:
    """Parse arguments, precompute all latents' statistics, and write the requested outputs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(LOADERS), required=True)
    parser.add_argument("--run", help="Aeon only: run directory in the sweep, e.g. d256/k12_seed0.")
    parser.add_argument("--out", type=Path, help="Directory for index.html and views/.")
    parser.add_argument("--latents", type=int, nargs="*", help="Default: every active latent.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--export", type=int, nargs="*", default=[],
                        help="Latents to save as static PDF and PNG views in --figures.")
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    data = LOADERS[args.dataset](**({"run": args.run} if args.run else {}))
    board = Dashboard(data)
    stem = f"dashboard_{args.dataset}" + (f"_{args.run.split('/')[0]}" if args.run else "")
    for latent in args.export:
        args.figures.mkdir(parents=True, exist_ok=True)
        fig = board.figure(latent, letters=True)
        for extension in ("pdf", "png"):
            fig.savefig(args.figures / f"{stem}_latent{latent}.{extension}", dpi=300)
        plt.close(fig)
    if args.out is not None:
        print(write_dashboard(board, args.out, args.latents, workers=args.workers))


if __name__ == "__main__":
    main()
