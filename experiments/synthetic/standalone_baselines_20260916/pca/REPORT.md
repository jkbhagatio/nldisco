# Standalone windowed PCA on the exact synthetic paper recording

**Answer:** this actual 128-component PCA fit does **not provide convincing recovery of four distinct individual place-cell latents**. Its strongest spatial components mix cells and contrast locations. It does encode forward versus backward clean first-pair crossings extremely well in one signed component (AUROC **0.98899**), but that component's unrestricted extremes occur around the other fields, and its two signs are **one component**, not separate localized forward/backward features. No paper files or existing datasets/models were changed.

## Actual fit and reproducibility

- Input is `experiments/synthetic/data/matryoshka_overlap`, the dataset actually used by current Section 4.1 and its appendix. The misleading historical name `paper_simulation` identifies the earlier generation input, with different cells 3/4. Input spike SHA-256: `ce5bd9573966e96cd36a44a68cc4e84836c95f749a59260e87398aaab06db09f`.
- The entire saved 600-s recording is used once: 100 units, 60,000 raw 10-ms samples; sum five successive samples; make ordered 20-token/1-s windows with stride one 50-ms token. The matrix is **11,981 × 2,000**, flattening time then unit. Raw window starts are 0,5,…,59,900.
- `sklearn.decomposition.PCA(n_components=128, svd_solver="full", whiten=False)`, float64, all columns centered, **no variance scaling**, no rotations, rectification, additional latent fitting, or target supervision during fitting. The installed full-SVD implementation was inspected: it calls SciPy's dense SVD, orders by singular value, and fixes arbitrary signs deterministically by loading convention. NumPy 1.23.5, SciPy 1.10.1, sklearn 1.5.2, Python 3.9; four BLAS threads.
- Full-SVD fit took **9.40 s**; main fit/analysis/figures **18.75 s**. Retained components explain **15.6451%** of total centered count variance. This includes abundant independent Poisson/Bernoulli variability across 96 noise units; variance explained is not a place-cell recovery score.
- Every PCA score stays signed. Target-specific sign orientation is explicitly recorded and changes interpretation only. Positive and negative versions of one PC are never counted separately. PC numbers and cell numbers in this report are **one-based**; saved array indices are zero-based.

Reproduce from repository root (only writes inside this standalone PCA directory):

```bash
uv run --no-sync python experiments/synthetic/standalone_baselines_20260916/pca/run_pca.py
uv run --no-sync python experiments/synthetic/standalone_baselines_20260916/pca/supplement.py
uv run --no-sync python -m pytest -q experiments/synthetic/standalone_baselines_20260916/pca/test_analysis.py
```

`pca_fit.npz` contains the complete mean, 128×2,000 components, variances, variance fractions, singular values, 11,981×128 native signed scores, and window starts. `summary.json` has exact configuration, software, timing, input hashes and checks. It is sufficient to reproduce projection as `(X-mean) @ components.T` without refitting. `diagnostics.npz` stores all tuning curves, field templates, full loading energy maps, masks and selected trajectories. `artifact_manifest.json` hashes the deliverables.

## Spatial recovery: four targets, one-to-one assignment, and source identity

Reference position is the average of **raw samples 49 and 50** in each 100-sample window, an explicit midpoint convention for the even-length window. Curves are mean signed scores in **40 fixed equal position bins** over [0,1]. All windows contribute. For each of four exact position-only generative fields, compare its 40-bin curve with every PC's tuning by Pearson correlation, choosing that PC's arbitrary sign per target. The primary display assignment maximizes the **sum of absolute correlations with four distinct PCs** (Hungarian assignment over all 128 candidates). This is a descriptive best match, not a binary SED recovery test.

| Target cell | Distinct assigned PC | Shape Pearson r | Target unit squared-loading fraction | Strongest squared-loading unit | Peak position |
|---|---:|---:|---:|---:|---:|
| 1, asymmetric .2 m | +PC7 | .5838 | 7.48% | cell 3 | .2125 m |
| 2, asymmetric .4 m | +PC2 | .5790 | 57.09% | cell 2 | .3875 m |
| 3, broad .7 m | −PC1 | .8944 | 38.40% | cell 2 | .7125 m |
| 4, narrow .7 m | −PC3 | .5417 | 7.56% | cell 2 | .6625 m |

The independent best match **reuses PC1 for three targets**: +PC1 for cell 2 (r=.7694), −PC1 for cell 3 (.8944), and −PC1 for cell 4 (.7443); cell 1 still prefers +PC7 (.5838). Those are not three distinct place latents. PC1's squared loading fractions on cells 1–4 are **4.07%, 41.34%, 38.40%, 15.45%**: it contrasts the first pair with the shared .7-m population. Similar position peaks therefore do not establish separation of the concentric fields.

Across **all 128 PCs**, cells 1 and 4 are **never the strongest source unit by time-summed squared loading**; their maximal component energy fractions are 30.51% (PC6) and 29.19% (PC4). Cells 2 and 3 dominate 8 and 37 PCs, respectively. The best shape matches among source-dominant PCs are cell 2/+PC1 (r=.7694) and cell 3/−PC7 (r=.2927). These facts support mixing in this native basis; they are not an impossibility theorem about information in the PCA subspace. This signed energy criterion differs from the paper's strongest **positive** decoder/coactivity criterion, so it is not labeled the paper's pass/fail test.

Width diagnostics reinforce the distinction, but require care with signed multilobed curves. Halfway between each curve's minimum and maximum, total exceedance spans for the four assigned PCs are **.875/.550/.500/.675 m**, compared with **.175/.175/.350/.150 m** for the exact fields on this 40-bin grid. These are **spans**, not literal FWHM for disconnected shapes: the assigned cell-1 and cell-4 curves have 3 and 7 disconnected exceedance segments. `spatial_width_audit.csv` also supplies total occupied bin width. Inspect raw signed curves rather than interpreting these span numbers alone.

Window averaging can broaden genuine fields. Accordingly, `spatial_all_candidates.csv` also includes start/end reference correlations, direct correlation of each score with the exact rate averaged over its entire raw window, and comparison with the midpoint-binned window-mean-rate tuning. For the four assigned PCs the latter shape correlations are **.478/.552/.892/.472**; direct per-window exact-rate correlations are **.166/.558/.852/.028**. This sensitivity does not turn them into four isolated source-unit features. The start/end correlations also show PC3 is a temporal contrast: its narrow-field-oriented correlation changes from +.540 at the start to −.561 at the end.

All four source units are partially reconstructed jointly by the full 128-PC subspace: per-unit count reconstruction R² is **.220/.668/.656/.417**. This is population reconstruction, not evidence that one distinct latent recovers each unit.

Files: `spatial_all_candidates.csv` (4×128 rows), `spatial_selected.csv`, `spatial_independent_winners.csv`, `spatial_best_source_dominant.csv`, `source_dominance_audit.json`, `reconstruction_per_unit.csv`. Smaller prefixes 16/32/64 retain the same selected spatial PCs and strongest direction axis; exact prefix results are in `summary.json`.

## Temporal recovery: very strong conditional discrimination, poor unrestricted localization

The **exact saved W-SED analysis metadata** were reused and all columns independently recomputed and checked, including alignment. Clean eligibility is strict nonreversing crossing of both ≤.22 and ≥.38 m, with mean position in [.2,.4]. Eligible pools contain **218 rightward / 232 leftward** overlapping windows. Within each .025-m mean-position bin with ≥10 windows each direction, calculate continuous-score rightward AUROC (half credit for ties), then average the bin AUROCs with weights min(nRight,nLeft). This retains **8 bins, matching weight 209 per direction**. Global analysis uses the saved nonreversing ≥.1-m displacement criterion and .1-m bins: **8 bins, weight 939**. No nonnegative DI denominator is applied to signed scores.

The largest sign-oriented clean AUROC among all 128 PCs is **PC4**. Native +PC4 favors backward crossings; −PC4 favors forward crossings. Both interpretations have clean AUROC **.98899** and orientation-matched full-track AUROC **.73163**. These are **two score tails of ONE PC**, not independently recovered forward/backward latents.

| Same PC4, orientation | Unrestricted top16 preferred direction | Unrestricted clean crossings | Clean-pool top16 preferred direction |
|---|---:|---:|---:|
| −PC4, forward | 7/16 | 0/16 | 16/16 |
| +PC4, backward | 7/16 | 0/16 | 16/16 |

Examples are greedily ranked by oriented continuous score, rejecting windows with raw start separation <100 samples; both directions compete in both pools. No preferred-direction filter or score rectification is applied. Unrestricted tail means lie near **.717 m forward / .687 m backward**, far from the first-pair region. The signed weight pattern has largely positive early and negative late contributions from cells 2/3/4, strongest around cells 3/4. Thus the excellent first-pair AUROC does not establish a feature that activates specifically during that localized traversal.

PC4's unit energy fractions are **0.83% cell 1, 22.62% cell 2, 39.24% cell 3, 29.19% cell 4**. First-pair/noise per-unit energy enrichment is 138.6, but the pair accounts for only **25.52% of known-place energy**, and cell 1 only **3.55% of pair energy**. Noise rejection alone would overlook this wrong-source mixture.

To check alternatives instead of selecting only the largest AUROC, all 128 candidates were also scanned for sign-invariant pair structure: ≥50% of known-place squared energy on cells 1/2, ≥10% of pair energy on each, and ≥2-fold mean pair/noise unit-energy enrichment. **Four PCs (5,6,70,97)** meet these structure criteria. Their best clean AUROC is **PC70, .60922** (global .51737 after the same orientation); the others are .52570/.53254/.52000. PC70's unrestricted forward/backward desired-direction counts are 12/16 each, but only 1/16 and 2/16 are clean first-pair crossings; its conditioned counts are 10/16 and 13/16. These broad, noisy alternatives do not supply convincing two-feature local traversal recovery. This structural screen is explicitly adapted to signed weights, not the original positive-weight SED gate.

PC3 provides a second distinct directional axis with clean AUROC .81609, but virtually excludes cell 1 (0.32% of pair energy). Its unrestricted orientation-matched direction counts are 0/16 forward and 2/16 backward; clean-pool counts become 15/16 and 16/16. The data therefore contain conditional directional information in multiple PCs, without corresponding unconditioned field-1→field-2 feature specificity. Every candidate's two tails and both trajectory pools are available in `all_candidate_top_trajectories.csv`; `direction_all_candidates.csv` contains all AUROCs and pair-structure metrics.

For context, the parent's independent reuse of the paper's two preselected TW-SED examples gives matched clean AUROC .96716 forward and .84332 reverse, and global .55067/.60625. Those SED examples have unrestricted desired-direction counts 15/16 and 12/16, conditioned16/16 each. PCA's superior **conditional classifier** number should not be equated to better **localized feature recovery**. These are selected descriptive examples, not a model-family significance comparison.

## Validation, limits, and figures

Numerical checks passed: maximum component orthogonality error **2.22e−15**, score projection discrepancy **5.38e−14**; reloaded projection matches saved scores; residual sum of squares equals centered total sum of squares minus retained score energy to 1e−12 relative tolerance; reconstruction MSE **.13337516 counts²**; all regenerated saved-motion columns align within 1e−12. Four fast unit tests pass, covering tied signed AUROC/sign reversal, bin matching/count exclusion, all-negative score ranking with nonoverlap, and endpoint-inclusive tuning bins. Existing Matplotlib dependency deprecation warnings only. Three key figures were visually inspected.

All fitting, target selection, and evaluation use the **same entire recording**. Windows overlap heavily. There are no held-out claims, independent confidence intervals, or calibrated discovery p-values. Searching128 candidates and choosing signs inflates apparent selected performance. A single deterministic PCA fit is not comparable to 10 random SED seeds' recovery frequency. The original vanilla/Matryoshka spatial inputs are 10-ms standardized per-bin samples, whereas this explicitly requested windowed baseline uses raw 50-ms counts over1 s; spatial differences combine objective, preprocessing and architecture. Generative firing is position-only with no direction dependence: temporal ordered static fields can create directional discrimination.

Figures (each has PNG and PDF):

- `spatial_all_candidate_matrix`: every field versus every PC.
- `spatial_four_targets`: four unique assigned PCs, display-only curve standardization, exact templates.
- `spatial_raw_signed_tuning`: native signed tuning and exact rates on independently scaled, labeled axes; overlap does not mean equal amplitude or matching baselines.
- `component_loading_maps`: full100-unit×20-time signed weights, separate place-cell traces, and source energy for each selected spatial PC, PC4, and PC70.
- `direction_top_trajectories`: opposite tails of PC4, unrestricted and clean-conditioned pools.
- `direction_localization`: directional mean scores across location, score energy localization, and all-candidate conditional/global AUROC comparison.
- `direction_pair_structural_trajectories`: weaker alternative PC70, both orientations and pools.

The evidence supports **partial spatial information and strong conditional direction encoding in a mixed signed basis**, not four clean individual place features or two separately localized forward/backward traversal features.
