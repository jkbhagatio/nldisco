# README figure exports

These PNGs are display exports of the current [ICLR paper](../../paper/iclr_paper/full_paper.pdf),
not independently redrawn results. Captions in the project README are shortened for that context.

| Asset | Source |
| --- | --- |
| `figure-1.png` | `paper/iclr_paper/figures/nldisco_pipeline.pdf` |
| `figure-2.png` | `paper/iclr_paper/figures/sed_arch.pdf` |
| `figure-s1.png` | `paper/iclr_paper/figures/interpretable_latents_vs_latent_space.svg` and `.pdf` |
| `table-s1.png` | Page 18 of the current `paper/iclr_paper/full_paper.pdf`, including its caption and notes |

Figure S3 uses `paper/iclr_paper/figures/task_overview.png` directly.
Figure S1's panel-c title was corrected in the SVG to “NLDisco latents” and its
standalone PDF regenerated. Plot data and legends were retained. The compiled full-paper
PDF is not rebuilt by these documentation edits.

From the repository root, export figures with Poppler:

```bash
pdftoppm -scale-to 2000 -png -singlefile paper/iclr_paper/figures/nldisco_pipeline.pdf docs/assets/figure-1
pdftoppm -scale-to 2400 -png -singlefile paper/iclr_paper/figures/sed_arch.pdf docs/assets/figure-2
pdftoppm -scale-to 2000 -png -singlefile paper/iclr_paper/figures/interpretable_latents_vs_latent_space.pdf docs/assets/figure-s1
pdftoppm -f 18 -l 18 -r 200 -x 306 -y 429 -W 1090 -H 1400 -hide-annotations -png -singlefile paper/iclr_paper/full_paper.pdf docs/assets/table-s1
```

Recheck the table's page and crop if the paper layout changes. Hiding annotations removes
PDF link boxes while preserving the table text, values, caption, and footnotes.
