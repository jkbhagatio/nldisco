"""Fast checks of the cards, the rater tool and the Part B statistics."""
import json

import numpy as np

from experiments.rebuttal.change03_discovery import cards, part_b, rater_cli
from experiments.rebuttal.change03_discovery.test_c03_core import synthetic_behavior


def test_states_follow_one_rule():
    rng = np.random.default_rng(0)
    sparse = np.where(rng.random(10_000) < 0.1, rng.random(10_000) + 0.5, 0.)
    state = cards.states(sparse)
    assert state["low"] is None and np.array_equal(state["high"], sparse > 0)
    dense = cards.states(rng.normal(size=10_000))
    assert np.isclose(dense["high"].mean(), 0.05, atol=0.002)
    assert np.isclose(dense["low"].mean(), 0.05, atol=0.002)


def test_card_is_anonymous_and_shows_tuning():
    v, support = synthetic_behavior(20_000)
    binnings = cards.build_binnings(v, support)
    assert all((b.index < len(b.labels)).all() for b in binnings)
    values = np.where(v["wheel_speed"] > 0.05, v["wheel_speed"] * 10, 0.)
    endpoints = np.arange(len(values)) + 19
    blocks = np.arange(len(values)) * 4 // len(values)
    text = cards.render_card("C1234", values, v, support, binnings, endpoints, blocks,
                             np.random.default_rng(0))
    assert text.startswith("# Latent card C1234")
    lowered = text.lower()
    assert not any(word in lowered for word in ("nldisco", "cebra", "langevin", "pca", "ndt"))
    row = next(line for line in text.splitlines() if line.startswith("| wheel_speed"))
    high = text.splitlines()[text.splitlines().index(row) + 4]
    cells = [c.strip() for c in high.split("|")[2:-1]]
    assert cells[0] == "0.0" and cells[3] == "100"
    dense = cards.render_card("C1", np.random.default_rng(1).normal(size=len(values)), v, support,
                              binnings, endpoints, blocks, np.random.default_rng(0))
    assert "% LOW" in dense and "% LOW" not in text


def test_rater_tool_serves_in_order_and_validates(tmp_path):
    v, support = synthetic_behavior(2000)
    (tmp_path / "cards").mkdir()
    for card in ("C1", "C2"):
        (tmp_path / "cards" / f"{card}.md").write_text(f"# Latent card {card}")
    (tmp_path / "order.json").write_text(json.dumps(["C2", "C1"]))
    np.savez(tmp_path / "behavior_reference.npz",
             **{f"var_{k}": a for k, a in v.items() if not k.startswith("_")},
             **{f"sup_{k}": a for k, a in support.items()})
    assert "card C2" in rater_cli.next_card(tmp_path)
    assert not rater_cli.submit(tmp_path, "C1", True, None, None, None)[0]
    assert not rater_cli.submit(tmp_path, "C2", False, "high", "speed >", "x")[0]
    assert not rater_cli.submit(tmp_path, "C2", False, "high", "speed >= 0", "x")[0]
    assert not rater_cli.submit(tmp_path, "C2", False, None, "speed > 5", "x")[0]
    accepted, message = rater_cli.submit(tmp_path, "C2", False, "high", "speed > 50", "fast")
    assert accepted and "holds in" in message
    assert "card C1" in rater_cli.next_card(tmp_path)
    assert rater_cli.submit(tmp_path, "C1", True, None, None, None)[0]
    assert "ALL CARDS ANSWERED" in rater_cli.next_card(tmp_path)
    answers = rater_cli.read_jsonl(tmp_path / "answers.jsonl")
    assert [a["interpretation"] for a in answers] == ["rule", "none"]
    events = [(e["event"], e["card"]) for e in rater_cli.read_jsonl(tmp_path / "log.jsonl")]
    assert events == [("serve", "C2"), ("submit", "C2"), ("serve", "C1"), ("submit", "C1")]


def test_sampling_identifiers_agreement_and_clusters():
    rng = np.random.default_rng(0)
    representation = rng.normal(size=(200, 64)).astype(np.float32)
    representation[:, :10] = 0.
    discovery = np.arange(200) < 150
    first = part_b.sample_latents(representation, discovery, "nldisco")
    assert len(first) == 32 and first.min() >= 10 and len(set(first)) == 32
    assert np.array_equal(first, part_b.sample_latents(representation, discovery, "nldisco"))
    assert not np.array_equal(first, part_b.sample_latents(representation, discovery, "pca"))
    ids = [i for m in part_b.METHODS for i in part_b.card_ids(m, 32)]
    assert len(set(ids)) == 160 and part_b.card_ids("pca", 32) == ids[-32:]
    assert part_b.fleiss_kappa(np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]])) == 1.
    assert abs(part_b.fleiss_kappa(rng.integers(0, 2, size=(3000, 3)))) < 0.05
    a = rng.random(5000) < 0.2
    b = a ^ (rng.random(5000) < 0.02)
    c = rng.random(5000) < 0.2
    labels = part_b.cluster_features(np.stack([a, b, ~a, c]))
    assert labels[0] == labels[1] == labels[2] != labels[3]
