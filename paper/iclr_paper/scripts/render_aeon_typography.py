"""Restyle Aeon paper figures from frozen figure inputs, without recomputing scores."""

from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

from result_typography import format_plot_label, format_tex

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from matplotlib.figure import Figure
from matplotlib.text import Text
from experiments.aeon.sweep_20260921 import paper_results


def main() -> None:
    """Generate paper-local figures with unchanged numerical arrays and scores."""
    scratch = Path(tempfile.mkdtemp(prefix='aeon-paper-typography-'))
    destination = ROOT / 'paper/iclr_paper/figures'
    for name in ('feature_scores.csv', 'traceback.npz', 'tuning_wheel.npz',
                 'tuning_area.npz', 'tuning_speed.npz', 'tuning_direction.npz',
                 'roc_wheel.npz', 'roc_direction.npz'):
        shutil.copyfile(paper_results.OUT / name, scratch / name)
    original_savefig = Figure.savefig

    def styled_savefig(figure, *args, **kwargs):
        for text in figure.findobj(match=Text):
            text.set_text(format_plot_label(text.get_text()))
        return original_savefig(figure, *args, **kwargs)

    with patch.object(Figure, 'savefig', styled_savefig):
        paper_results.render(scratch)
    for name in ('aeon_main', 'aeon_additional_features', 'aeon_direction',
                 'aeon_traceback', 'aeon_wheel_context_traceback'):
        for extension in ('pdf', 'png'):
            shutil.copyfile(scratch / f'{name}.{extension}',
                            destination / f'{name}.{extension}')
    (destination / 'aeon_feature_metrics.tex').write_text(
        format_tex((scratch / 'aeon_feature_metrics.tex').read_text()))
    print(f'Updated Aeon paper typography from saved inputs in {scratch}')


if __name__ == '__main__':
    main()
