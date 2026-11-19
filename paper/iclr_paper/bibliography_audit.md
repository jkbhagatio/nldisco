# Bibliography audit — 25 September 2026

## Scope and results

Audited **all 74 entries** in `references.bib`, including all five currently uncited entries. Recursively traversed the paper's 21 active TeX files from `full_paper.tex`: **69 unique citation/nocite keys, no undefined keys, no missing input files**. Every entry identifies a real source via its publisher, proceedings, preprint repository, author-supplied original paper, or official dataset/book documentation. This is a bibliography identity/metadata audit, not a claim-by-claim peer review of the manuscript.

**43 entries received metadata corrections, publication updates, or source-link improvements.** No citation keys were renamed. One further entry (`pei_2022_nlb`) only lost trailing whitespace; mixed line endings and trailing whitespace in the bibliography were normalized. No files were deleted, committed, or pushed.

**No duplicate citation keys, normalized titles, or DOI identifiers remain; no duplicate publications were found to merge (zero merges).** Related works such as the NLB paper/dataset documentation/original experiment and the two attribution-graph papers remain separate. UMAP's full theory preprint was not replaced by the distinct JOSS software paper; Langevin Flows' full preprint was not replaced by a short poster abstract.

All 74 entries now have a source URL. A final GET/redirect sweep returned **64 HTTP 200 responses, seven HTTP 403 access restrictions, and three HTTP 429 rate limits**. All ten restricted sources were independently identified through authoritative metadata or accessible primary records. Therefore, these are access limitations, not evidence of invalid citations; this audit does not claim all publisher pages are accessible to every client. No DOI was invented where the publisher did not provide one. There are no `and others` placeholders in stored author lists.

## Material corrections

- Added the omitted Marius Pachitariu to the 90-author Allen/Siegle paper.
- Corrected NeuroSeeker's reversed Alexandru Andrei name and GPFA's Gopal Santhanam name and actual published title.
- Corrected exact published author forms, including the voltage-imaging paper, MM-GPVAE's Usama Bin Sikandar surname parsing, and current publisher metadata for Rhino Nevers.
- Updated eight preprint entries to verified proceedings versions: KiloSort; Sparse Autoencoders Find Highly Interpretable Features; Matryoshka SAEs; SAEBench; transcoders; sequence-to-sequence learning; Zhang's neural translator; and Schmutz's solvable model. The latter two published author lists additionally include Renee Tung and Matteo Carandini, respectively.
- Added missing stable source links, identifiers, volumes, and page ranges where verified; completed PCA's page range.
- Shortened the verified Hernandez/NBDT URL to its numeric publisher redirect, addressing the bibliography's overflowing URL without suppressing the reference.

## Explicit caveats

1. **ICLR SAE authors:** current ICLR landing page and official BibTeX list Huben, Cunningham, Smith, Ewart, Sharkey; the published PDF still lists Cunningham, Ewart, Riggs, Huben, Sharkey. The bibliography follows current official citation metadata, so the legacy key `cunningham_2023_saes` now renders Huben et al. (2024). Both primary records are linked below. The conflict is not hidden or represented as conclusively resolved.
2. **Anthropic author metadata:** Monosemanticity's official BibTeX includes Zac Hatfield-Dodds, whom its visible byline omits; retained the explicit official citation list. Scaling Monosemanticity's visible byline includes four authors absent from embedded BibTeX; retained all 26 credited authors. These publisher inconsistencies cannot be eliminated by an external audit.
3. **BatchTopK venue:** the arXiv source and all authors are verified. A potential NeurIPS 2024 workshop version could not be conclusively verified from accessible primary acceptance metadata; retained arXiv rather than asserting that venue.
4. **Preprint completeness:** Aeon, Suite2p, UMAP, Langevin Flows, Low/MIND, and BatchTopK retain verified preprint citations where no matching final full-paper publication was established. A negative search cannot prove no newer version exists. Suite2p's 2017 year refers specifically to v2, not its first 2016 posting.
5. **Upstream NLB duplicate:** the proceedings landing page duplicates David Zoltowski. The actual paper PDF has 16 authors; the correct existing 16-author list was retained instead of importing the website error.
6. **Uncited entries/relevance:** `bussmann_2025_msae`, `hattori_2019_history_value_coding`, `masullo_2019_colliculus_orienting`, `sutskever_2014_seq2seq`, and `templeton_2024_future_acts` were all audited despite being unused. Methods describes nested Matryoshka levels but does not currently cite the Matryoshka source; this is a possible attribution omission for author review. No citation was inserted solely because a bibliography entry existed.

## Final URL access check

The following returned access restrictions. Their per-entry evidence below verifies identity and metadata independently. Every other final URL returned HTTP 200 after redirects. A 200 response alone is not treated as proof of metadata correctness; all entries were also checked against their underlying source records.

| Citation key | HTTP response | Final bibliography URL |
| --- | --- | --- |
| `steinmetz_2021_neuropixels2` | 403 | [Source](https://doi.org/10.1126/science.abf4588) |
| `campagner_2025_aeon_platform` | 429 | [Source](https://www.biorxiv.org/content/early/2025/08/01/2025.07.31.664513) |
| `pachitariu_2017_suite2p` | 429 | [Source](https://www.biorxiv.org/content/10.1101/061507v2) |
| `yu_2009_gpfa` | 403 | [Source](https://doi.org/10.1152/jn.90941.2008) |
| `low_2018_mind` | 429 | [Source](https://doi.org/10.1101/418939) |
| `sussillo_2013_rnn_dynamics` | 403 | [Source](https://doi.org/10.1162/NECO_a_00409) |
| `truccolo_2005_pointprocess` | 403 | [Source](https://doi.org/10.1152/jn.00697.2004) |
| `shenoy_2013_dynamical_perspective` | 403 | [Source](https://doi.org/10.1146/annurev-neuro-062111-150509) |
| `doya_2007_bayesian_brain` | 403 | [Source](https://direct.mit.edu/books/edited-volume/2884/Bayesian-BrainProbabilistic-Approaches-to-Neural) |
| `vincent_2008_denoising_autoencoders` | 403 | [Source](https://doi.org/10.1145/1390156.1390294) |

## Manuscript build validation

The root task rebuilt `full_paper.pdf` after all bibliography edits. Result: nine main-text pages and 45 pages total; no LaTeX/BibTeX warnings or errors, undefined references, missing characters, overfull boxes, or unresolved `??` markers. The earlier bibliography URL overflow is resolved. Bibliography expansion adds one reference page; restored methodological criteria add one appendix page.

## Per-entry verification

Publisher-deposited Crossref metadata was used together with original-paper bylines and venue records. Exact publication versions and upstream discrepancies are noted rather than blindly importing one export. Initial rate limits were retried successfully for metadata. ArXiv DOI prefixes belong to DataCite; their absence from Crossref is not a broken DOI.

## Recording, dataset, and foundational references (27 entries)

| Key | Verification and disposition | Primary evidence |
| --- | --- | --- |
| `steinmetz_2021_neuropixels2` | Verified all 39 authors, title, Science 372(6539), article eabf4588, 2021 against publisher deposit. Publisher HTML blocked (403) in browser; DOI identity independently verified. | [Source](https://doi.org/10.1126/science.abf4588); [publisher DOI deposit](https://api.crossref.org/works/10.1126/science.abf4588) |
| `raducanu_2017_neuroseeker` | Corrected reversed Alexandru Andrei name; standardized full list to published byline (Refet F. Yazicioglu, Carolina M. Lopez, Veronique Rochus). All 17 authors and Sensors 17(10):2388 verified. Author-hosted published PDF confirms byline. | [Source](https://eprints.soton.ac.uk/446104/1/sensors_17_02388_v2.pdf); [publisher DOI deposit](https://api.crossref.org/works/10.3390/s17102388) |
| `cai_2016_miniscope` | All 21 authors and Nature 534(7605):115–118 (2016) match publisher deposit; retained. | [Source](https://pubmed.ncbi.nlm.nih.gov/27251287/); [publisher DOI deposit](https://api.crossref.org/works/10.1038/nature17955) |
| `villette_2019_voltage_2p` | Corrected Ivan K. Dimov, Guoqiang Bi, Katalin Toth, and Jun Ding to published byline. All 20 authors/order and Cell 179(7):1590–1608.e23 verified against publisher deposit and deposited author manuscript. | [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC6941988/); [publisher DOI deposit](https://api.crossref.org/works/10.1016/j.cell.2019.11.004) |
| `ouzounov_2017_three_photon` | All 11 authors and Nature Methods 14(4):388–390 (2017) verified; retained. | [Source](https://doi.org/10.1038/nmeth.4183); [publisher DOI deposit](https://api.crossref.org/works/10.1038/nmeth.4183) |
| `ahrens_2013_lightsheet` | All five authors and Nature Methods 10(5):413–420 (2013) verified; retained. | [Source](https://doi.org/10.1038/nmeth.2434); [publisher DOI deposit](https://api.crossref.org/works/10.1038/nmeth.2434) |
| `churchland_2012_population_dynamics` | All seven authors and Nature 487(7405):51–56 (2012) verified; retained. | [Source](https://doi.org/10.1038/nature11129); [publisher DOI deposit](https://api.crossref.org/works/10.1038/nature11129) |
| `azabou_2023_unified` | All ten authors match proceedings PDF, including middle initials omitted on landing page. NeurIPS 36 (2023):44937–44956 and proceedings DOI verified; retained. | [Source](https://proceedings.neurips.cc/paper_files/paper/2023/file/8ca113d122584f12a6727341aaf58887-Paper-Conference.pdf); [publisher DOI deposit](https://api.crossref.org/works/10.52202/075280-1948) |
| `nlb_mcmaze` | Live official dataset documentation verified. Corporate attribution now uses actual site name Neural Latents Benchmark (removed unsupported Team suffix); access date refreshed. This cites the dataset documentation, distinct from the benchmark paper and original experiment. No publication date asserted. | [Source](https://neurallatents.github.io/datasets#mcmaze) |
| `siegle_2021_allen_datasets` | Added omitted Marius Pachitariu between Michael Oliver and Jed Perkins. All 90 authors/order checked against Nature byline and deposit. Nature 592(7852):86–92 (2021) verified. | [Source](https://www.nature.com/articles/s41586-020-03171-x); [publisher DOI deposit](https://api.crossref.org/works/10.1038/s41586-020-03171-x) |
| `campagner_2025_aeon_platform` | All 20 individual author names/initials plus SGEN group corporate author match bioRxiv Crossref deposit; 2025 preprint DOI/title verified. Browser full text blocked (403); no later journal version established. Retained as preprint, not represented as peer reviewed. | [Source](https://www.biorxiv.org/content/early/2025/08/01/2025.07.31.664513); [publisher DOI deposit](https://api.crossref.org/works/10.1101/2025.07.31.664513) |
| `cunningham_2014_neural_dr` | Both authors and Nature Neuroscience 17(11):1500–1509 (2014) verified; added missing DOI and resolvable DOI URL, normalized journal capitalization. | [Source](https://www.nature.com/articles/nn.3776); [publisher DOI deposit](https://api.crossref.org/works/10.1038/nn.3776) |
| `humphries_2021_dr_principles` | Author/title and NBDT 5(2):1–28 (2021) verified against publisher citation and DOI deposit; retained. | [Source](https://nbdt.scholasticahq.com/article/24619-strong-and-weak-principles-of-neural-dimension-reduction); [publisher DOI deposit](https://api.crossref.org/works/10.51628/001c.24619) |
| `pachitariu_2016_kilosort` | Updated original KiloSort preprint citation to the same five-author original algorithm conference paper, Fast and accurate spike sorting of high-channel count probes with KiloSort, NIPS 29 (2016):4448–4456. Official proceedings full author initials used. This is not KiloSort2/3/4. | [Source](https://proceedings.neurips.cc/paper/2016/hash/1145a30ff80745b56fb0cecf65305017-Abstract.html) |
| `pachitariu_2017_suite2p` | All eight authors match bioRxiv v2. Kept 2017 because cited revision was posted 20 July 2017, despite Crossref recording first posting in 2016. Replaced early/date URL with explicit v2 canonical URL. | [Source](https://www.biorxiv.org/content/10.1101/061507v2); [publisher DOI deposit](https://api.crossref.org/works/10.1101/061507) |
| `okeefe_dostrovsky_1971_place_cells` | Both authors and Brain Research 34(1):171–175 (1971) verified; retained. | [Source](https://doi.org/10.1016/0006-8993(71)90358-1); [publisher DOI deposit](https://api.crossref.org/works/10.1016/0006-8993(71)90358-1) |
| `theodoni_2018_theta_ou` | All four authors and eLife 7:e37388 (2018) verified; retained. | [Source](https://doi.org/10.7554/eLife.37388); [publisher DOI deposit](https://api.crossref.org/works/10.7554/eLife.37388) |
| `takens_1981_embedding` | Floris Takens chapter, 1981, pages366–381 and Springer chapter DOI verified; book title/volume898/editor fields retained. | [Source](https://doi.org/10.1007/BFb0091924); [publisher DOI deposit](https://api.crossref.org/works/10.1007/BFb0091924) |
| `sussillo_2013_rnn_dynamics` | Both authors and Neural Computation25(3):626–649 (2013) verified; retained. | [Source](https://doi.org/10.1162/NECO_a_00409); [publisher DOI deposit](https://api.crossref.org/works/10.1162/NECO_a_00409) |
| `goodfellow_2016_rnn` | All three authors, title, MIT Press, 2016 checked against author-provided BibTeX. Chapter10 title verified in official book contents. Upgraded URL to HTTPS. | [Source](https://www.deeplearningbook.org/) |
| `pillow_2008_glms` | All seven authors and Nature454(7207):995–999 (2008) verified; retained. | [Source](https://doi.org/10.1038/nature07140); [publisher DOI deposit](https://api.crossref.org/works/10.1038/nature07140) |
| `truccolo_2005_pointprocess` | All five authors and Journal of Neurophysiology93(2):1074–1089 (2005) verified; retained. | [Source](https://doi.org/10.1152/jn.00697.2004); [publisher DOI deposit](https://api.crossref.org/works/10.1152/jn.00697.2004) |
| `shenoy_2013_dynamical_perspective` | All three authors and Annual Review of Neuroscience36:337–359 (2013) verified; retained. | [Source](https://doi.org/10.1146/annurev-neuro-062111-150509); [publisher DOI deposit](https://api.crossref.org/works/10.1146/annurev-neuro-062111-150509) |
| `rao_1999_predictive_coding` | Both authors and Nature Neuroscience2(1):79–87 (1999) verified; retained. | [Source](https://doi.org/10.1038/4580); [publisher DOI deposit](https://api.crossref.org/works/10.1038/4580) |
| `doya_2007_bayesian_brain` | All four editors/title verified. Retained copyright/citation year2007 confirmed by MIT Press Books Gateway; retail page separately lists hardcover release December2006 and paperback January2011. Replaced paperback-specific URL with edition-neutral publisher book record. | [Source](https://direct.mit.edu/books/edited-volume/2884/Bayesian-BrainProbabilistic-Approaches-to-Neural) |
| `friston_2010_free_energy` | Changed author Karl J. Friston to published byline Karl Friston. Nature Reviews Neuroscience11(2):127–138 (2010) verified. | [Source](https://doi.org/10.1038/nrn2787); [publisher DOI deposit](https://api.crossref.org/works/10.1038/nrn2787) |
| `pei_2022_nlb` | Published proceedings PDF confirms all16 authors/order and 2021 publication. Landing page erroneously lists David M Zoltowski and David Zoltowski twice; did not import duplicate. Citation key retains legacy2022 label, which does not control rendered year. | [Source](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/979d472a84804b9f647bc185a877a8b5-Paper-round2.pdf) |

## Neural models (29 entries)

## hotelling_1933_pca

Status: updated. Correct incomplete pages 417 to 417--441; normalize journal capitalization and add DOI. Single author Harold Hotelling; publisher registers H. Hotelling. This is part I, volume24 issue6, not the separately paginated continuation.

Sources: [1](https://api.crossref.org/works/10.1037/h0071325), [2](https://doi.org/10.1037/h0071325)


## comon_1994_ica

Status: verified; no replacement required. Verified author Pierre Comon, title, Signal Processing 36(3):287--314 (1994), DOI and URL. Title capital A after comma on publisher is only case convention; no substantive change.

Sources: [1](https://www.sciencedirect.com/science/article/abs/pii/0165168494900299)


## hoyer_2004_sparsenmf

Status: verified; no replacement required. Verified sole author Patrik O. Hoyer, exact title, JMLR 5(Nov):1457--1469, 2004. URL correct; no DOI on official record.

Sources: [1](https://jmlr.org/papers/v5/hoyer04a.html)


## vandermaaten_2008_tsne

Status: updated. Verified both authors/order and exact title, JMLR9:2579--2605 (2008). Add article number86 and primary URL. No DOI provided by JMLR.

Sources: [1](https://jmlr.org/papers/v9/vandermaaten08a.html)


## mcinnes_2018_umap

Status: updated. Verified all three authors/order, title and 2018 arXiv origin. Retain theory preprint: JOSS3(29):861 is a DISTINCT software paper, shorter title and different authors (Saul/Großberger instead of Melville), not interchangeable. No later publication identified for this exact work.

Sources: [1](https://arxiv.org/abs/1802.03426), [2](https://doi.org/10.21105/joss.00861)


## song_2025_langevinflow

Status: updated. Verified all five authors/order, title and original2025 posting. Latest arXiv v2 March10 2026 calls itself full version of CCN2025 poster; keep full-paper preprint rather than replace by short poster abstract. No journal/conference full-paper version identified.

Sources: [1](https://arxiv.org/abs/2507.11531), [2](https://2025.ccneuro.org/abstract_pdf/Song_2025_Langevin_Flows_Modeling_Neural_Latent_Dynamics.pdf)


## schneider_2023_cebra

Status: updated. Verified Nature617(7960):360--368,2023 and title. Expand author given names exactly as publisher (Jin Hwa Lee; Mackenzie Weygandt Mathis); same three authors/order.

Sources: [1](https://api.crossref.org/works/10.1038/s41586-023-06031-6), [2](https://doi.org/10.1038/s41586-023-06031-6)


## le_2022_stndt

Status: updated. Both authors/order, title, NeurIPS35(2022):17926--17939 and existing PDF all verified. Add proceedings DOI.

Sources: [1](https://proceedings.neurips.cc/paper_files/paper/2022/hash/72163d1c3c1726f1c29157d06e9e93c1-Abstract-Conference.html)


## keshtkaran_2022_autolfads

Status: verified; no replacement required. Verified ALL10 authors/order, exact title, Nature Methods19(12):1572--1577(2022), DOI and URL against publisher-deposited Crossref record.

Sources: [1](https://api.crossref.org/works/10.1038/s41592-022-01675-0), [2](https://doi.org/10.1038/s41592-022-01675-0)


## yu_2009_gpfa

Status: updated. SUBSTANTIVE: journal article title ends Neural Population Activity, not Population Spike Trains; third author is Gopal Santhanam, not Gokhan. All6 authors/order verified; journal102(1):614--635,2009 correct.

Sources: [1](https://api.crossref.org/works/10.1152/jn.90941.2008), [2](https://doi.org/10.1152/jn.90941.2008)


## macke_2011_plds

Status: updated. Verified all6 authors/order and title, volume24/year2011. Add official URL and normalize proceedings title. PDF spells Lars Büsing, metadata Buesing (equivalent transliteration). Official BibTeX leaves pages empty; no verified DOI or page range added.

Sources: [1](https://proceedings.neurips.cc/paper/2011/hash/7143d7fbadfa4693b9eec507d9d37443-Abstract.html), [2](https://proceedings.neurips.cc/paper/2011/file/7143d7fbadfa4693b9eec507d9d37443-Paper.pdf)


## gao_2016_pflds

Status: updated. All4 authors/order and title exactly match original PDF (Evan Archer; John P. Cunningham). Add verified volume29 and primary URL;2016 correct. Page range/DOI not verified from publisher, omitted.

Sources: [1](https://proceedings.neurips.cc/paper/2016/hash/76dc611d6ebaafc66cc0879c71b5db5c-Abstract.html), [2](https://proceedings.neurips.cc/paper/2016/file/76dc611d6ebaafc66cc0879c71b5db5c-Paper.pdf)


## pandarinath_2018_lfads

Status: updated. Verified all14 authors/order, title and Nature Methods15(10):805--815,2018. Match publisher byline L.F. Abbott rather than expanded Larry F; journal case normalized, DOI added.

Sources: [1](https://api.crossref.org/works/10.1038/s41592-018-0109-9), [2](https://doi.org/10.1038/s41592-018-0109-9)


## low_2018_mind

Status: updated. Current publisher-deposited author metadata names Rhino Nevers (existing Rachel); use current name. Other4 authors/order and exact title,2018 bioRxiv verified. No publication relation in Crossref and no published successor found in targeted search; absence is not conclusive.

Sources: [1](https://api.crossref.org/works/10.1101/418939), [2](https://doi.org/10.1101/418939)


## jensen_2020_mgplvm

Status: updated. All4 authors/order and exact title verified, including Kristopher T. Jensen in PDF. Add verified NeurIPS33,22580--22592,2020. No DOI on proceedings.

Sources: [1](https://proceedings.neurips.cc/paper/2020/hash/fedc604da8b0f9af74b6cfc0fab2163c-Abstract.html), [2](https://proceedings.neurips.cc/paper/2020/file/fedc604da8b0f9af74b6cfc0fab2163c-Paper.pdf)


## hernandez_2020_vind

Status: updated. All6 authors/order, title, journal3(3),2020 verified verbatim on publisher page. No DOI/pages in publisher record. Short numeric URL verified HTTP redirect to exact long article URL; avoids bibliography line overflow.

Sources: [1](https://nbdt.scholasticahq.com/article/13476)


## kim_2021_plnde

Status: updated. All4 authors/order, exact title, ICML2021/PMLR139:5551--5561 and URL verified; use publisher given initials rather than unverified expanded Doyeon/Zhihao. No DOI in PMLR record.

Sources: [1](https://proceedings.mlr.press/v139/kim21h.html)


## hurwitz_2021_tndm

Status: updated. All7 authors/order confirmed from original PDF; title/year2021 correct. Add verified volume34,page range and URL. No DOI found.

Sources: [1](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f5cfbc876972bd0d031c8abc37344c28-Abstract.html), [2](https://proceedings.neurips.cc/paper_files/paper/2021/file/f5cfbc876972bd0d031c8abc37344c28-Paper.pdf)


## schimel_2022_ilqrvae

Status: updated. Correct acronym capitalization iLQR-VAE from official ICLR2022 downloads listing. Four authors/order, title, venue/year and URL consistent with indexed original paper and proceedings TOC. Live OpenReview PDF/API blocked/challenged; no page range or DOI confidently verified; retained existing metadata.

Sources: [1](https://iclr.cc/Downloads/2022), [2](https://openreview.net/pdf?id=wRODLDHaAiW), [3](https://www.proceedings.com/content/075/075297webtoc.pdf)


## kudryashova_2023_ctrltndm

Status: verified; no replacement required. Verified all4 authors/order, exact title, Cosyne2023 abstract2-053 starts printed144 and ends145 (PDF indices165--166). Correctly cited as abstract, not full journal article. No DOI.

Sources: [1](https://www.cosyne.org/s/Cosyne2023_program_book.pdf)


## ye_2023_ndt2

Status: updated. All4 authors/order,title/year2023 verified. Add official volume36/pages80352--80374 and DOI. Existing Jennifer L. consistent fuller name; publisher metadata abbreviates Jennifer.

Sources: [1](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fe51de4e7baf52e743b679e3bdba7905-Abstract-Conference.html)


## gondur_2024_mmgpvae

Status: updated. SUBSTANTIVE name parsing: surname is Sikandar, given names Usama Bin, verified author-owned repository BibTeX and official proceedings. All5/order,title,ICLR2024 verified. Add official proceedings page range14257--14281 and URL. Full Aoi/Keeley names supported by original arXiv and author repository. No DOI verified.

Sources: [1](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3d226fb8fbd6ee6ec70d0427f1319707-Abstract-Conference.html), [2](https://github.com/RabiaGondur/MM-GPVAE), [3](https://arxiv.org/abs/2310.03111)


## pellegrino_2024_slicetca

Status: updated. All3 authors/order,title,Nature Neuroscience27(6):1199--1210,2024 verified. Add DOI and URL.

Sources: [1](https://api.crossref.org/works/10.1038/s41593-024-01626-2), [2](https://doi.org/10.1038/s41593-024-01626-2)


## sani_2024_dpad

Status: updated. All3 authors/order,title,Nature Neuroscience27(10):2033--2045,2024 verified. Add DOI and URL.

Sources: [1](https://api.crossref.org/works/10.1038/s41593-024-01731-2), [2](https://doi.org/10.1038/s41593-024-01731-2)


## pals_2024_smclr_rnn

Status: updated. All5 authors/order (including Sağtekin),title,NeurIPS37,2024 and DOI/URL verified. Add page range18225--18264 from official proceedings (Crossref also confirms).

Sources: [1](https://papers.nips.cc/paper_files/paper/2024/hash/209423f076b6479ab3a4f45886e30306-Abstract-Conference.html)


## zhang_2024_mtm

Status: updated. SUBSTANTIVE publication upgrade to NeurIPS2024,37:80495--80521. Original arXiv has11 authors and stale metadata; published paper has12 including Renee Tung after Blake Richards. Correct Donato M. Jiménez-Benetó spelling/initial and corporate author The International Brain Laboratory. All12/order verified official PDF and proceedings.

Sources: [1](https://proceedings.neurips.cc/paper_files/paper/2024/hash/934eb45b99eff8f16b5cb8e4d3cb5641-Abstract-Conference.html), [2](https://proceedings.neurips.cc/paper_files/paper/2024/file/934eb45b99eff8f16b5cb8e4d3cb5641-Paper-Conference.pdf)


## george_2025_simpl

Status: updated. All5/order,title,ICLR2025 verified. Published PDF and official proceedings say Kimberly (not Kim); use full byline Tom M.George from paper. Add official proceedings metadata pages27662--27686 (third-party printed TOC differs; official site preferred). No DOI verified.

Sources: [1](https://proceedings.iclr.cc/paper_files/paper/2025/hash/44d20d542f3f4e3b7097e5e3f78f99f1-Abstract-Conference.html), [2](https://openreview.net/pdf?id=9kFaNwX6rv)


## perkins_2025_mint

Status: updated. All5 authors/order,title,eLife12:RP89421,2025,DOI .3 verified VERSION OF RECORD Feb3 2025. Volume12/year2025 is unusual but explicitly correct. Match version-of-record John Cunningham rather than John P. (P. appears in earlier reviewed preprint).

Sources: [1](https://elifesciences.org/articles/89421/figures), [2](https://api.crossref.org/works/10.7554/eLife.89421.3)


## schmutz_2025_nce

Status: updated. SUBSTANTIVE publication upgrade: NeurIPS2025,38:118390--118424. Six authors now include Matteo Carandini before Kenneth D.Harris; exact order verified original PDF and proceedings. Official BibTeX year2025 (HTML deposition2026 is not conference year).

Sources: [1](https://papers.nips.cc/paper_files/paper/2025/hash/aba2ac76b4b5df7a7b833c87c9012860-Abstract-Conference.html), [2](https://papers.nips.cc/paper_files/paper/31185-/bibtex)


## Mechanistic interpretability and remaining references (18 entries)

## cunningham_2023_saes — updated; primary-source conflict recorded

Published in ICLR 2024, not merely arXiv 2023. [Official proceedings landing page](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1fa1ab11f4bd5f94b2ec20e794dbfa3b-Abstract-Conference.html) and [official BibTeX](https://proceedings.iclr.cc/paper_files/paper/2024/file/1fa1ab11f4bd5f94b2ec20e794dbfa3b-Bibtex-Conference.bib) agree on Robert Huben, Hoagy Cunningham, Logan Smith, Aidan Ewart, Lee Sharkey, in that order; pages 7827–7845; 2024. Title unchanged. Entry follows those current citation records.

**Uncertainty:** the [actual ICLR paper PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/1fa1ab11f4bd5f94b2ec20e794dbfa3b-Paper-Conference.pdf), first page, retains Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, Lee Sharkey, exactly as the [arXiv record](https://arxiv.org/abs/2309.08600). This is a real publisher landing-page/PDF discrepancy, not an inference of an error in the original citation. The entry chooses current publisher citation metadata; retaining the PDF author order is also defensible and the discrepancy should be disclosed. No publisher DOI appears in official BibTeX; the old DOI identifies the preprint. If the manuscript contains prose saying “Cunningham et al.” it needs consideration if adopting this proceedings author order.

## bricken_2023_towards_monosemanticity — verified with source inconsistency; URL cleanup only

[Official paper and Citation Information](https://transformer-circuits.pub/2023/monosemantic-features/index.html#citation) confirms title, Transformer Circuits Thread, 2023 (4 October), and 25-author supplied BibTeX list. Existing author order matches that list; retaining Nicholas L. Turner is supported by visible byline, although supplied BibTeX shortens him to Nick Turner. Supplied BibTeX uses Christopher Olah, existing entry matches. Replace section-specific appendix URL with article root.

**Source inconsistency:** visible byline has 24 authors and omits Zac Hatfield-Dodds; supplied official BibTeX explicitly includes him after Nicholas Joseph and before Alex Tamkin. Retain him because the publication explicitly instructs that citation. Do not claim the byline independently confirms all 25. No DOI or formal journal/conference upgrade found.

## templeton_2024_scaling_monosemanticity — verified; retain all 26 existing authors

[Official paper](https://transformer-circuits.pub/2024/scaling-monosemanticity/) visible byline exactly confirms all 26 authors and order in current bibliography, including Alex Tamkin, Esin Durmus, Tristan Hume, Francesco Mosconi between Monte MacDiarmid and C. Daniel Freeman. Title, Transformer Circuits Thread, and 2024 (21 May) match. No DOI/formal venue upgrade found.

**Source inconsistency:** official embedded BibTeX omits those four authors and lists only 22. Retaining the full visible byline is safer than deleting real credited authors to reproduce incomplete citation markup.

## ameisen_2025_circuit_tracing — verified; retain

[Official paper](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) confirms all 27 authors in current order, title, publication 27 March 2025, Transformer Circuits Thread, and URL. No DOI/formal venue upgrade found.

**Minor source typo:** supplied BibTeX parses T. Ben Thompson as `Ben Thompson, T.` while visible byline says T. Ben Thompson. Existing `Thompson, T. Ben` is supported by the companion Biology paper's own BibTeX and should remain. Existing full names Thomas Henighan/Thomas Conerly match byline and citation.

## lindsey_2025_biology_llm — verified; retain

[Official paper](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) confirms all 27 authors and their order, title, Transformer Circuits Thread, 27 March 2025, URL. Supplied BibTeX also matches current author list and explicitly uses `Thompson, T. Ben`. No DOI/formal venue upgrade found.

## lindsey_2024_crosscoders — verified; retain

[Official article](https://transformer-circuits.pub/2024/crosscoders/) credits Jack Lindsey, Adly Templeton, Jonathan Marcus, Thomas Conerly, Joshua Batson, Christopher Olah, exactly matching current entry and order. Exact title and 25 October 2024 confirmed. Published as a preliminary research update in Transformer Circuits Thread. No DOI or subsequent formal publication identified.

## bussmann_2025_msae — updated to ICML proceedings

[Official PMLR paper and BibTeX](https://proceedings.mlr.press/v267/bussmann25a.html) confirms Bart Bussmann, Noa Nabeshima, Adam Karvonen, Neel Nanda in existing order and exact title. ICML 2025, Proceedings of the 42nd International Conference on Machine Learning; PMLR series volume 267, pages 6077–6101. Replace arXiv-only entry with inproceedings. No DOI in official publisher citation. The arXiv page does not advertise this upgrade, illustrating why independent venue searches were necessary.

## bussmann_2024_batchtopk — verified preprint; unresolved workshop venue

[arXiv](https://arxiv.org/abs/2412.06410) confirms exact title, Bart Bussmann, Patrick Leask, Neel Nanda in order, 2024 (9 December), identifier, cs.LG and DOI 10.48550/arXiv.2412.06410. Replacement adds that verified DOI only.

**Publication-status uncertainty:** several later papers cite a NeurIPS 2024 Workshop on Scientific Methods for Understanding Deep Learning version at [OpenReview](https://openreview.net/forum?id=d4dpOCqybL). The primary forum, PDF and API were browser-challenged/403; searchable primary PDF text is anonymous and says only submitted to NeurIPS 2024, so it does not conclusively establish acceptance/venue. Do not silently upgrade based on other papers' references. Keeping the verified arXiv record is conservative. This is the sole unresolved formal-publication check in this tranche.

## chanin_2024_feature_absorption — verified; add proceedings pages

[Official NeurIPS 2025 record](https://proceedings.nips.cc/paper_files/paper/2025/hash/764ff7477b8e24dbe01531f6791e8bdf-Abstract-Conference.html) and [publisher DOI](https://doi.org/10.52202/085713-2755) confirm exact title, six authors David Chanin, James Wilken-Smith, Tomáš Dulka, Hardik Bhatnagar, Satvik Golechha, Joseph Bloom in existing order, NeurIPS/Advances in Neural Information Processing Systems 38 (2025). Crossref deposit supplies pages 91489–91526, added in replacement. DOI and URL already correct. Key's 2024 is only an internal identifier; year correctly remains 2025. Some earlier OpenReview citations omit Golechha: current final publisher record includes him and should control.

## karvonen_2025_saebench — updated to ICML proceedings

[Official PMLR paper and BibTeX](https://proceedings.mlr.press/v267/karvonen25a.html) confirms exact title and 15 authors in current order. Publisher expands Joseph Bloom to Joseph Isaac Bloom and Callum McDougall to Callum Stuart Mcdougall; replacement reproduces those publisher spellings. ICML 2025/PMLR 267, pages 29223–29264. No DOI in official citation. Replace arXiv-only article with inproceedings. Mcdougall capitalization is as the official landing page/export, although other publications and the existing record use McDougall; this is an orthographic metadata variation, not a different author.

## dunefsky_2024_transcoders — updated to NeurIPS proceedings

[Official NeurIPS record](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2b8f4db0464cc5b6e9d5e6bea4b9f308-Abstract-Conference.html) and [BibTeX](https://proceedings.neurips.cc/paper_files/paper/2024/file/2b8f4db0464cc5b6e9d5e6bea4b9f308-Bibtex-Conference.bib) confirm Jacob Dunefsky, Philippe Chlenski, Neel Nanda in order, exact title apart from case, Advances in Neural Information Processing Systems 37 (2024), pages 24375–24410, DOI 10.52202/079017-0768. Replace arXiv DOI with actual proceedings DOI and use inproceedings.

## templeton_2024_future_acts — section attribution explicitly verified; retain

[Predicting Future Activations section](https://transformer-circuits.pub/2024/jan-update/index.html#predicting-future-activations) explicitly credits Adly Templeton, Joshua Batson, Adam Jermyn, Chris Olah in precisely current order. This is a section-specific citation, so retaining the section anchor and parenthetical Circuits Updates–January 2024 context is accurate. The whole update has different bylines per section; do not substitute the first section's authors or attribute it generically to all Anthropic authors. No DOI/formal venue upgrade identified.

## sutskever_2014_seq2seq — updated to NIPS proceedings

[Official NIPS 2014 record](https://proceedings.neurips.cc/paper_files/paper/2014/hash/5a18e133cbf9f257297f410bb7eca942-Abstract.html) and [BibTeX](https://proceedings.neurips.cc/paper_files/paper/2014/file/5a18e133cbf9f257297f410bb7eca942-Bibtex.bib) confirm Ilya Sutskever, Oriol Vinyals, Quoc V. Le, exact title, Advances in Neural Information Processing Systems 27 (2014), Curran Associates. Replace arXiv-only article with inproceedings. Official BibTeX pages field is empty; do not fabricate pagination. No proceedings DOI provided.

## vincent_2008_denoising_autoencoders — verified; retain

[Author-hosted ICML paper](https://www-labs.iro.umontreal.ca/~vincentp/Publications/vincent_icml_2008.pdf) confirms exact title and Pascal Vincent, Hugo Larochelle, Yoshua Bengio, Pierre-Antoine Manzagol in existing order. [Publisher DOI record](https://doi.org/10.1145/1390156.1390294), via Crossref deposit, confirms ICML 2008 proceedings, pages 1096–1103, DOI and year. ACM web landing returns 403, but publisher metadata and author PDF suffice. Current venue expansion/capitalization is equivalent to ACM's title. Publisher ACM and Helsinki meeting location valid.

## olshausen_1997_sparse_overcomplete — verified; retain

[Author-hosted published paper](https://www.rctn.org/bruno/papers/VR.pdf) and [publisher DOI](https://doi.org/10.1016/S0042-6989(97)00169-7) confirm exact title, Bruno A. Olshausen then David J. Field, Vision Research 37(23), 3311–3325, 1997, Elsevier. DOI case difference in Crossref has no semantic effect. Existing title protects V1 appropriately.

## lenail_2019_nnsvg — verified; retain

[JOSS publisher page and BibTeX](https://joss.theoj.org/papers/10.21105/joss.00747) exactly confirm Alexander LeNail (including capital N), title, Journal of Open Source Software 4(33):747, 2019, publisher The Open Journal, DOI 10.21105/joss.00747 and DOI URL. Publication is 15 January 2019, despite original submission 2018.

## hattori_2019_history_value_coding — verified; retain

[Publisher DOI](https://doi.org/10.1016/j.cell.2019.04.027), publisher-deposited Crossref metadata, and [archived paper](https://pmc.ncbi.nlm.nih.gov/articles/6663310/) confirm exact title and all five authors/order: Ryoma Hattori, Bethanny Danskin, Zeljana Babic, Nicole Mlynaryk, Takaki Komiyama. Cell 177(7):1858–1872.e15, 2019. DOI and URL correct. No author accents were invented; publisher deposit uses Zeljana Babic.

## masullo_2019_colliculus_orienting — verified; retain

[Archived full published paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6739420/) and [publisher DOI](https://doi.org/10.1016/j.cub.2019.07.083) confirm Laura Masullo, Letizia Mariotti, Nicolas Alexandre, Paula Freire-Pritchett, Jerome Boulanger, Marco Tripodi, exactly in existing order. Exact title, Current Biology 29(17):2892–2904.e8, 2019, DOI and URL correct. Published author spelling is Nicolas (not Nicholas), Jerome (no accent in paper record). No changes.
