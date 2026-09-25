"""Render paper-only Churchland assets with verified millimetre labels.

Run from the repository root:
    uv run --no-sync python paper/iclr_paper/scripts/render_churchland_units.py

Numeric positions already represent mm; derivatives already use seconds.
No data, feature definitions, scores, or experiment outputs are modified.
The existing caption-disclosed visualization mock is preserved separately.
"""

from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

from result_typography import format_plot_label

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from matplotlib.figure import Figure
from matplotlib.text import Text

from experiments.churchland.sweeps.s10_endpoint_v1 import comparison_features as shared
from experiments.churchland.sweeps.s10_endpoint_v1 import matched192_figures as figures
from experiments.churchland.sweeps.s10_endpoint_v1 import nice_cutoffs_revision as revision


def main() -> None:
    source = ROOT / 'experiments/churchland/outputs/s10_endpoint_revision_v4_nice_cutoffs'
    destination = ROOT / 'paper/iclr_paper/figures'
    scratch = Path(tempfile.mkdtemp(prefix='churchland-paper-mm-'))
    for name in ('frozen_spec.json', 'all_candidates.json'):
        shutil.copyfile(source / name, scratch / name)

    original_savefig = Figure.savefig

    def save_with_units(figure, *args, **kwargs):
        for text in figure.findobj(match=Text):
            label = text.get_text().replace('native units', 'mm')
            label = label.replace('Target (125, −18)', 'Target (125, −18) mm')
            # Figures are saved as PDF and PNG: do not append the unit twice.
            label = label.replace(' mm mm', ' mm')
            text.set_text(format_plot_label(label))
        return original_savefig(figure, *args, **kwargs)

    with patch.object(shared, 'load_development_features', revision.features), \
            patch.object(Figure, 'savefig', save_with_units):
        figures.render(scratch, scratch / 'mock', mock_banner=False)

    for extension in ('pdf', 'png'):
        for source_name, paper_name in (
            ('main_features', 'figure4'),
            ('mock/main_features_mock', 'figure4_mock'),
        ):
            shutil.copyfile(scratch / f'{source_name}.{extension}',
                            destination / f'{paper_name}.{extension}')
    # Appendix figures and the consolidated table use the current selection rule.
    # Retain Table 1 as the source of its displayed values.
    from paper.iclr_paper.scripts.render_churchland_appendix import main as render_appendix

    render_appendix()
    print(f'Updated paper figures in {destination}; rendering provenance: {scratch}')


if __name__ == '__main__':
    main()
