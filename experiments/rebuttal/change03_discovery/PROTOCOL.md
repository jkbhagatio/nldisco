# Change 3: feature-discovery comparison on session B, fixed protocol

Fixed on 2026-11-18, before session B's data existed. Session A was used only to
develop and test the code; no session A result is reported.

## Data and split
- Session B (prepared by change 2), 20 ms bins, 0.4 s windows (20 bins) that end at
  the scored bin. Every method sees the same windows.
- The recording is cut into 20 equal blocks of time. Blocks 4, 8, 12, 16 and 20
  (25%) are the held-out part; the other 15 blocks are the discovery part.
- Inside each discovery block the first 90% is the fit portion and the last 10%
  the validation portion. Windows that span two portions are dropped, so no
  held-out window shares a bin with a discovery window.
- Models are fitted on the fit portion only, with no behavior labels. The
  validation portion is used only to choose between settings of a baseline by that
  baseline's own loss. Latent selection, thresholds and summary cards use the
  discovery part (fit and validation). The held-out part is used once, for scoring.

## Methods (192 latents each)
- NLDisco: Section 4.3 settings (TW-SED, D=192, k=12, Matryoshka 96/192, 5,000
  updates of 1,024 windows, learning rate 0.005, seed 0). One run, no selection.
- CEBRA-Time: Section 4.2 recipe (temperature 1, positive lag 1 bin, 128 hidden
  units, cosine InfoNCE, Adam), with the offset encoder extended to a 20-bin
  receptive field. Settings compared: learning rate 1e-4, 3e-4 (Section 4.2), 1e-3.
- LangevinFlow: Section 4.2 recipe (64-dimensional position, momentum and GRU
  states = 192 latents), adapter extended to 20 bins. Settings compared: learning
  rate {0.001, 0.003} x ramp {5, 500} (the Section 4.2 sweep at the selected width).
- ST-NDT: change 1's runner, settings selected there, on the same windows.
- PCA: first 192 principal components of the z-scored, flattened 20-bin window.
- Every neural-network method gets the same budget as NLDisco: 5,000 updates of
  1,024 windows. A baseline setting is chosen only by its own loss on the
  validation portion. Behavior is never used to choose a setting.
- Each method yields one 192-dimensional vector per window, aligned to the
  window's last bin; each coordinate is a candidate latent.

## Scoring rule, identical for all methods and both parts
Given a latent, an orientation (high or low) and a behavioral condition:
1. Orientation: values are multiplied by -1 for "low".
2. Threshold: the value maximising Youden's J (coverage + specificity - 1) on the
   discovery part; the latent is active when its oriented value is at or above it.
3. Held-out scores with that orientation and threshold: coverage, specificity,
   selectivity and AUROC (paper definitions).
4. Pass criterion (primary): held-out AUROC >= 0.75 and selectivity >= 0.75.
   Sensitivity criteria, also fixed here: both >= 0.70, both >= 0.80, AUROC alone
   >= 0.75.

## Part A (prespecified features)
- 23 features in `PART_A_FEATURES.md` / `features.py`.
- For each method and feature: orient every latent so that its discovery AUROC is
  >= 0.5, pick the latent with the highest discovery AUROC (ties: lowest index),
  then apply the scoring rule. A feature is recovered if it passes the criterion.

## Part B (blind discovery)
- Sample: 32 latents per method, drawn uniformly without replacement (seed
  20261118 + method index in the order NLDisco, CEBRA-Time, LangevinFlow, ST-NDT,
  PCA) from the latents that are not constant on the discovery part. The same
  latents go to every rater.
- Cards: one text card per latent, built from the discovery part only by the same
  code for every method (`cards.py`). Cards carry a random identifier, never the
  method or the latent index.
- Raters: three independent AI agents (Claude Sonnet 5.5, high effort), each a
  fresh agent with the same instructions (`RATER_INSTRUCTIONS.md`), its own random
  card order, one card at a time through `rater_cli.py`, no access to other
  raters' answers, to the key, to model outputs or to the held-out part.
- Answer per card: "no interpretation", or one rule on the behavior variables
  (a boolean expression) with a direction (latent high or low when the rule
  holds) and a one-line description.
- Validation: the rule is evaluated on all windows whose 20-bin span has valid
  samples of the variables it uses, and the scoring rule above is applied with the
  rater's direction. A description is validated if it passes the primary criterion
  and the rule holds in at least 500 and fails in at least 500 held-out windows.
- Reported per method: proportion of inspected latents with a validated
  description (mean over raters, and by majority of raters); Fleiss' kappa between
  raters for "gave an interpretation" and for "validated"; the distinct features;
  effort per validated feature (latents inspected, and rater time measured from
  serving a card to receiving its answer).
- Distinct features: validated descriptions from all methods and raters are
  clustered by the absolute phi correlation of their held-out condition masks
  (average linkage, clusters cut at |phi| = 0.5). A method surfaces a feature if
  at least one of its validated descriptions is in the cluster.

## Decision rule
NLDisco "comes out ahead" if it has the highest (ties allowed) number of recovered
features in Part A and the highest (ties allowed) proportion of validated latents
in Part B under the primary criterion. Otherwise the numbers are reported to Jai
and no paper text is written.
