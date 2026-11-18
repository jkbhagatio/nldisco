# change08_dictsize REPORT

**Status: DONE** (macaque, Aeon, and session B). Branch `rebuttal/c08-dictsize`, commit `988756b`. Elapsed about 1 h 05 min of 3 h.

## Headline
- Dead latents rise with D (Aeon 0 -> ~50 (10%) -> ~257 (25%) at D=192/512/1024; macaque 0 -> 3 -> 21 (2%); session B 3 -> 52 -> 248). Latents are preserved: 88-92% of the paper's D=192 latents reappear (max activation correlation >= 0.5, same windows) at D>=256 vs 86% (macaque) / 88% (Aeon) between independent D=192 seeds; best-AUROC latent per paper feature stays 0.77-0.99 at all sizes.
- **Differs from the authors' reply (ALERT event sent 11:13Z):** end-to-end training time does not clearly grow with D. Controlled benchmark (paper schedule, interleaved sizes, min over repeats): macaque 69/68/69/62/68 s for D=128/192/256/512/1024 (flat: loop is data-loader bound, ~50 ms/update; GPU step alone 8.0/9.4/8.8/9.8/10.4 ms, +11% at 1024); Aeon 100/93/102/110/143 s (+18% at 512, +53% at 1024 vs 192; GPU step 9.1/9.4/9.7/12.3/17.0 ms, +82% at 1024). Wall-clock of real runs on the shared GPUs varied 2-5x with other sessions' load (e.g. Aeon D=192 runs 547-1656 s) and cannot rank sizes; they are listed as "recorded" below. The appendix text reports exactly this. Suggested reply wording: per-update compute grows with D; end-to-end training time grows only mildly (Aeon +53% at D=1024).

## What was run (outputs under `/ceph/aeon/aeon/nldisco/rebuttal_20261118/change08_dictsize`)
- Training: macaque `experiments/rebuttal/change08_dictsize/train_macaque.py` (wraps the frozen `s10_endpoint_v1/nldisco.py`; D in {512,1024}, k=16, levels D/2 and D, seeds 0,1,2) -> `runs/macaque/`. Aeon: unchanged `experiments/aeon/sweep_20260921/shared_training.py` (k=12, 5000 steps, batch 1024; D in {512,1024} seeds 0,1,2; extra D=192 seeds 1,2 as seed-to-seed reference) -> `runs/aeon/`. Session B: same runner on change 2's `session_b` data (D in {512,1024}, seeds 0,1,2) -> `runs/session_b/`, reference = change 2's `models/b_only/d192_k12_seed0` (and d256 as existing).
- Existing runs used read-only: macaque D=128/256 (seed 0, k=16, l2), D=192 seeds 0-5 (paper model = seed 1, per `frozen_spec.json`); Aeon D=128/192/256 k12 seed 0 (paper = D=192).
- Analysis: `analyze.py` (per run JSON/NPZ in `analysis/`), `report.py` (tables in `tables/`, figure), `time_training.py` (timing in `timing/`), `dictsize.py` (helpers), `macaque_features.py`/`aeon_features.py` (the paper's feature definitions: macaque v4 cutoffs; Aeon `paper_results.definitions`). Tests: 8 pass (`$PY -m pytest experiments/rebuttal/change08_dictsize`). Small results are copied to `experiments/rebuttal/change08_dictsize/results/`.
- Thresholds fixed before analysing any new run (`thresholds.json`, committed first): match/reappear tau=0.5, near-duplicate tau=0.8 (Pearson, native activations, all exported windows); sensitivity grid {0.3,0.5,0.7,0.9} / {0.5,0.7,0.8,0.9} stored in every run JSON.
- Selection rule: eligible latent (nonconstant, TPR>=0.05, active positives in >=20 trials (macaque) / >=20 contiguous episodes (Aeon)) with max AUROC (ties: lowest id); best-Sel also stored. Native z>0, positive orientation. **Validated:** reproduces the paper's macaque D=192 seed-1 best-Sel and best-AUROC latents and Sel/AUROC values exactly (all four features, v4 labels); on Aeon the best-AUROC rule picks the paper's latents (wheel 52 at D=192; area 107, speed 109, direction 10 at D=256) with identical Sel/AUROC (0.744/0.924 etc.). The Aeon eligibility criterion (episodes) is my analogue of the macaque rule; the Aeon paper latents were chosen by inspection.
- float32 sparse correlations checked against float64: max abs difference 5e-4, no threshold flips.

## Results (mean (min-max) over runs; "recorded" = wall-clock of the actual runs, shared GPUs)

### macaque

| D | runs | dead | dead % | time [s] (recorded) | time [s] (benchmark) | recent_braking AUROC / Sel | early_leftward AUROC / Sel | fast_target_specific AUROC / Sel | late_target_specific AUROC / Sel | paper latents reappearing % | new live latents | new % | near-dup. live latents | near-dup. % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 1 | 0 | 0.0 | 56 | 69 | 0.833 / 0.701 | 0.987 / 0.959 | 0.990 / 0.975 | 0.944 / 0.903 | 74.5 | 10 | 7.8 | 0 | 0.0 |
| 192 | 6 | 0 (0-0) | 0.0 (0.0-0.0) | 55 (52-59) | 68 | 0.836 (0.777-0.893) / 0.787 (0.692-0.907) | 0.981 (0.960-0.992) / 0.950 (0.926-0.979) | 0.943 (0.904-0.990) / 0.973 (0.955-0.986) | 0.940 (0.899-0.976) / 0.849 (0.744-0.955) | 86.0 (83.9-87.5) | 26 (21-31) | 13.6 (10.9-16.1) | 0 (0-2) | 0.2 (0.0-1.0) |
| 256 | 1 | 1 | 0.4 | 55 | 69 | 0.929 / 0.839 | 0.985 / 0.920 | 0.962 / 0.978 | 0.948 / 0.756 | 89.1 | 56 | 22.0 | 0 | 0.0 |
| 512 | 3 | 3 (2-5) | 0.6 (0.4-1.0) | 173 (162-187) | 62 | 0.841 (0.789-0.877) / 0.888 (0.856-0.931) | 0.977 (0.975-0.981) / 0.971 (0.966-0.975) | 0.914 (0.867-0.989) / 0.865 (0.707-0.980) | 0.936 (0.914-0.965) / 0.900 (0.839-0.974) | 89.2 (88.5-89.6) | 223 (217-233) | 43.9 (42.5-45.7) | 1 (0-2) | 0.1 (0.0-0.4) |
| 1024 | 3 | 21 (10-29) | 2.0 (1.0-2.8) | 166 (149-183) | 68 | 0.853 (0.813-0.883) / 0.865 (0.773-0.950) | 0.977 (0.973-0.984) / 0.958 (0.936-0.969) | 0.890 (0.852-0.935) / 0.950 (0.941-0.960) | 0.898 (0.893-0.906) / 0.955 (0.916-0.978) | 88.9 (87.5-90.1) | 691 (684-697) | 68.8 (68.7-69.0) | 0 (0-0) | 0.0 (0.0-0.0) |

### aeon

| D | runs | dead | dead % | time [s] (recorded) | time [s] (benchmark) | wheel AUROC / Sel | area AUROC / Sel | speed AUROC / Sel | direction AUROC / Sel | paper latents reappearing % | new live latents | new % | near-dup. live latents | near-dup. % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 1 | 1 | 0.8 | 1136 | 100 | 0.921 / 0.681 | 0.839 / 0.781 | 0.815 / 0.729 | 0.920 / 0.690 | 80.7 | 14 | 11.0 | 48 | 37.8 |
| 192 | 3 | 0 (0-1) | 0.2 (0.0-0.5) | 1053 (547-1656) | 93 | 0.917 (0.904-0.924) / 0.723 (0.695-0.744) | 0.845 (0.840-0.849) / 0.704 (0.670-0.769) | 0.839 (0.832-0.854) / 0.767 (0.760-0.776) | 0.899 (0.863-0.919) / 0.627 (0.589-0.646) | 87.5 (87.0-88.0) | 23 (22-24) | 12.0 (11.5-12.5) | 64 (56-71) | 33.6 (29.2-37.2) |
| 256 | 1 | 3 | 1.2 | 567 | 102 | 0.925 / 0.698 | 0.849 / 0.778 | 0.778 / 0.767 | 0.872 / 0.812 | 87.5 | 59 | 23.3 | 106 | 41.9 |
| 512 | 3 | 50 (46-56) | 9.8 (9.0-10.9) | 955 (281-1472) | 110 | 0.921 (0.916-0.926) / 0.703 (0.693-0.713) | 0.837 (0.824-0.850) / 0.694 (0.659-0.750) | 0.826 (0.801-0.844) / 0.773 (0.744-0.792) | 0.852 (0.787-0.895) / 0.776 (0.726-0.840) | 90.1 (89.1-90.6) | 187 (171-197) | 40.4 (37.5-42.3) | 137 (122-154) | 29.7 (26.3-33.8) |
| 1024 | 3 | 257 (219-285) | 25.1 (21.4-27.8) | 912 (771-1016) | 143 | 0.897 (0.876-0.939) / 0.742 (0.719-0.785) | 0.840 (0.827-0.855) / 0.699 (0.647-0.730) | 0.779 (0.769-0.796) / 0.746 (0.731-0.775) | 0.912 (0.903-0.921) / 0.689 (0.682-0.703) | 91.7 (90.6-93.2) | 439 (418-451) | 57.2 (56.0-59.1) | 203 (177-255) | 26.4 (23.4-31.7) |

### session_b

| D | runs | dead | dead % | time [s] (recorded) | time [s] (benchmark) | paper latents reappearing % | new live latents | new % | near-dup. live latents | near-dup. % |
|---|---|---|---|---|---|---|---|---|---|---|
| 192 | 1 | 3 | 1.6 | 890 | 93 | - | - | - | 84 | 44.4 |
| 256 | 1 | 5 | 2.0 | 582 | 102 | 89.9 | 38 | 15.1 | 109 | 43.4 |
| 512 | 3 | 52 (38-59) | 10.2 (7.4-11.5) | 331 (212-410) | 110 | 91.0 (89.4-92.1) | 170 (163-175) | 36.9 (36.0-38.6) | 160 (145-168) | 34.7 (32.0-37.1) |
| 1024 | 3 | 248 (212-285) | 24.2 (20.7-27.8) | 546 (451-721) | 143 | 92.4 (91.0-93.1) | 416 (412-419) | 53.7 (51.5-56.7) | 221 (197-250) | 28.4 (25.3-30.8) |

At D=192 the "reappearing / new" columns compare independent seeds with the paper's model (macaque: 5 other seeds; Aeon: 2; session B: none), i.e. they are the seed-to-seed reference. Macaque D=192 feature columns average six seeds (paper model = seed 1: AUROC 0.777/0.991/0.990/0.976).

## Observations to keep in mind
- Macaque fast-leftward and late-reach AUROC at D=1024 (0.89, 0.90) are below the D=192 six-seed means (0.94, 0.94) but inside the seed range at 192 for fast-leftward (0.90-0.99) and just below for late reach (0.90-0.98); recent deceleration varies 0.78-0.89 across D=192 seeds alone. Aeon speed 0.78 at D=1024 vs 0.84 at D=192 (paper latent at D=256: 0.778).
- Aeon near-duplicates (corr >= 0.8) are already 34% of the paper model's live latents (30-44% at D=128-256, 26-30% at 512/1024); macaque has none. Cause not investigated.
- Table 1 difference (0.671/0.868 printed vs 0.571/0.768 saved) not touched; not relevant to this change.
- I deleted one empty directory I had created (a failed first launch, `runs/macaque/d512_k16_l2_seed0` before retry); nothing else deleted. A first full-schedule timing pass, contaminated by other users' load, was aborted and kept as `timing_fullschedule_contended_aborted/` (not used).
- Not verified: no held-out analysis; features for session B were not scored (only dead/shared/new/near-duplicate counts, as scoped); timing is an estimate from short interleaved benchmarks on shared H100s (node gpu-sr675-33), not a dedicated-GPU measurement.

## Deliverables for the orchestrator
1. Appendix (already wired): `paper/iclr_paper/appendix/rebuttal/change08_dictsize.tex` (subsubsection, label `subsubsection:rebuttal_dictionary_size`, figure label `figure:rebuttal_dictionary_size`), one `\input{appendix/rebuttal/change08_dictsize}` line appended to `appendix/additional_results/additional_results.tex` (after the Aeon input). Figure: `paper/iclr_paper/figures/rebuttal/change08_dictsize/dictionary_size.{pdf,png}`. Compile check passed (`latexmk`, exit 0, no undefined refs; adds ~2 appendix pages, appears as 6.4.4 / Figure S21 in my build).
2. Proposed Methods clause (Stage 2 of `methods/methods.tex`, replacing "an SED with configurable dictionary width $D$ is trained to reconstruct target neural activity"):

```latex
an SED with configurable dictionary width $D$ (192 or 256 in Sections~\ref{subsection:churchland_results} and~\ref{subsection:aeon_results}; \autoref{subsubsection:rebuttal_dictionary_size} varies $D$ from 128 to 1024, preserving the discovered latents while increasing dead latents) is trained to reconstruct target neural activity.
```
3. Merging notes: `experiments/rebuttal/__init__.py` is an empty file other changes also create (identical). Helpers (feature scoring, correlation) duplicate those in changes 5/6; merge as you see fit. The appendix says "the paper's four feature descriptions" for Aeon as wheel/area/speed/direction.
