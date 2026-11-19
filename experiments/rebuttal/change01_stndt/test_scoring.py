"""Fast checks for ST-NDT latent selection and the hand-inserted table rows."""
import json
import re

import numpy as np
import pytest

from experiments.rebuttal.change01_stndt import figures, score


def _row(latent_id, sel, auroc, eligible=True):
    return dict(latent_id=latent_id, sel=sel, auroc=auroc, eligible=eligible)


def test_best_selective_requires_eligibility_and_breaks_ties():
    rows = [_row(0, .99, .6, eligible=False), _row(1, .9, .7), _row(2, .9, .8), _row(3, .9, .8)]
    assert score.best_selective(rows)["latent_id"] == 2  # Sel, then AUROC, then lowest index
    assert score.best_selective([_row(0, 1., 1., eligible=False)]) is None


def test_scoring_matches_shared_contract_on_dense_columns():
    """The selection path is the shared rate-matched scorer, sign choice included."""
    rng = np.random.default_rng(0)
    labels = rng.random(4000) < .2
    trials = np.repeat(np.arange(200), 20)
    values = rng.normal(size=(4000, 3)).astype(np.float32)
    values[:, 1] -= 3 * labels  # negatively oriented detector
    feature = score.shared.FeatureData(np.arange(4000), labels, trials, .2, {})
    rows = score.shared.score_dense(values, feature)
    best = score.shared.best_eligible(rows)
    assert best["latent_id"] == 1 and best["orientation"] == -1 and best["auroc"] > .95
    assert score.best_selective(rows)["latent_id"] == 1
    assert abs(best["achieved_rate"] - .2) < 1e-3


def _strip(text):
    return re.sub(r"\\textbf\{([^{}]+)\}", r"\1", text)


def test_tables_add_rows_without_changing_existing_numbers():
    values = [.5123, .9876, .9, .95, .8, .97]
    rows = {(feature, criterion): dict(
        latent_id=7, measured=values, main=[f"{v:.3f}" for v in values],
        appendix=[f"{v:.4f}" for v in values])
        for feature in figures.FEATURE_ORDER for criterion in figures.SELECTOR}
    new = figures.tables(rows)
    for name, key in (("churchland_comparison_main.tex", "main"),
                      ("churchland_feature_metrics.tex", "appendix")):
        old = [line for line in (figures.PAPER / name).read_text().splitlines()
               if not line.startswith("ST-NDT")]
        kept = [line for line in new[key].splitlines() if not line.startswith("ST-NDT")]
        assert [_strip(line) for line in kept] == [_strip(line) for line in old]
        added = [line for line in new[key].splitlines() if line.startswith("ST-NDT")]
        assert len(added) == (2 if key == "main" else 8)
    assert r"ST-NDT (best \metric{Sel}) & 0.512 & 0.988" in _strip(new["main"])


def test_rebold_marks_ties_and_only_maxima():
    lines = [r"A & 1 & 0.500 & \textbf{0.900} \\", r"B & 2 & 0.700 & 0.900 \\",
             r"C & 3 & 0.700 & 0.100 \\"]
    assert figures.rebold(lines, slice(2, 4)) == [
        r"A & 1 & 0.500 & \textbf{0.900} \\", r"B & 2 & \textbf{0.700} & \textbf{0.900} \\",
        r"C & 3 & \textbf{0.700} & 0.100 \\"]


def test_displayed_precision_follows_existing_tables():
    results = dict(
        selections=[dict(feature="recent_braking", criterion="sel", latent_id=5, tpr=.697554,
                         fpr=.23257, sel=.74995, auroc=.81844)],  # CEBRA-Time, as printed
        decoding=[dict(feature="recent_braking", latent_id=5,
                       test=dict(balanced_accuracy=.72913)),
                  dict(feature="recent_braking", latent_id=None,
                       test=dict(balanced_accuracy=.89948))])
    row = figures.displayed(results)["recent_braking", "sel"]
    assert row["main"] == ["0.698", "0.767", "0.749", "0.818", "0.729", "0.899"]
    assert row["appendix"] == ["0.6976", "0.7674", "0.749", "0.818", "0.7291", "0.8995"]


def test_additional_features_adds_stndt_and_keeps_printed_panels(tmp_path):
    fitz = pytest.importorskip("fitz")
    rows = figures.displayed(json.loads((figures.HERE / "outputs/results.json").read_text()))
    figures.additional_features(rows, tmp_path)
    old, new = (fitz.open(path)[0] for path in (
        figures.PAPER / "churchland_additional_features.pdf",
        tmp_path / "churchland_additional_features.pdf"))
    assert new.rect == old.rect
    above = [[w[:5] for w in page.get_text("words") if w[3] < 436] for page in (old, new)]
    assert above[0] == above[1]
    assert [w[4] for w in new.get_text("words") if w[1] > 436] == [
        "NLDisco", "CEBRA-Time", "LangevinFlow", "ST-NDT", "Best", "Sel", "Best", "AUROC"]
    purple = [d["rect"] for d in new.get_drawings() if d.get("color") and np.allclose(
        d["color"], (148 / 255, 103 / 255, 189 / 255), atol=.01)]
    centres = [((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2) for r in purple if r.y1 < 436]
    # Printed panel c axes: AUROC .5 -> 1 at x 407.96 -> 526.5; Sel 1 -> .7 over 92.12 pt.
    scores = sorted(((x - 407.96) / 118.54 * .5 + .5,
                     1 - (y - (45.36 if y < 200 else 252.62)) / 92.12 * .3) for x, y in centres)
    expected = sorted([(.959, .980), (.974, .977), (.976, .987), (.976, .987)])
    assert len(purple) == 5 and np.allclose(scores, expected, atol=2e-3)
