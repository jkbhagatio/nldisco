"""Regenerate Figure 3 and S4–S11 from the saved synthetic experiment."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from beartype import beartype
from .scripts import paper, appendix, attribution_traceback as attribution, pca_supplement

INPUTS = Path(__file__).resolve().parent / 'outputs/paper'
OUTPUT = Path(__file__).resolve().parent / 'outputs/reproduced'
FIGURES = ('figure3','supplement_rate_design','supplement_temporal_comparison',
           'supplement_temporal_features','supplement_ramp_conditioned','supplement_unit_traceback',
           'supplement_temporal_coactivity','supplement_pca_spatial','supplement_pca_direction')


@beartype
def load() -> dict:
    """Load verified runs and keep all regenerated outputs in their own directory."""
    OUTPUT.mkdir(parents=True,exist_ok=True)
    paper.PAPER=OUTPUT
    pca_supplement.OUTPUT=OUTPUT
    return paper.load_results()


@beartype
def render_tracebacks(results: dict, recompute: bool = False) -> None:
    """Render encoder attribution; optionally recompute it from the checkpoints."""
    raw=np.load(paper.DATA/'spike_matrix.npy').T.astype(np.float32)
    if recompute:
        spatial=attribution.spatial_attribution(raw,np.load(paper.DATA/'positions.npy'))
        temporal=None
    else:
        with np.load(INPUTS/'appendix_unit_attribution.npz') as saved:spatial=dict(saved)
        with np.load(INPUTS/'appendix_temporal_attribution.npz') as saved:temporal=dict(saved)
    plt.close(attribution.spatial_figure(results,spatial))
    fig,temporal=attribution.temporal_figure(results,raw,temporal)
    plt.close(fig)
    for old,new in (('supplement_unit_traceback_attribution','supplement_unit_traceback'),
                    ('supplement_temporal_coactivity_attribution','supplement_temporal_coactivity')):
        for ext in ('pdf','png'):(OUTPUT/f'{old}.{ext}').rename(OUTPUT/f'{new}.{ext}')
    np.savez_compressed(OUTPUT/'appendix_unit_attribution.npz',**spatial)
    np.savez_compressed(OUTPUT/'appendix_temporal_attribution.npz',**temporal)


@beartype
def main() -> None:
    """Replot every synthetic data figure without training models."""
    results=load();paper.write_report(results)
    for draw in (paper.main_figure,appendix.place_field_figure,paper.transformer_figure,
                 appendix.temporal_features_figure,appendix.clean_crossings_figure):plt.close(draw(results))
    appendix.direction_table(results)
    render_tracebacks(results)
    pca_supplement.main()
    (OUTPUT/'reproduction.json').write_text(json.dumps(dict(figures=FIGURES,
        sources='Saved runs, activations, selected features, gradient-times-input arrays and frozen PCA fit',
        training=False),indent=2)+'\n')


if __name__=='__main__':
    main()
