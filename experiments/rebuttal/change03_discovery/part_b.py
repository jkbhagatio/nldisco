"""Part B: blind discovery. `build` prepares the cards and rater folders; `analyze`
validates the raters' rules on the held-out part.

    python -m experiments.rebuttal.change03_discovery.part_b build --data DATA --runs RUNS --output OUT
    python -m experiments.rebuttal.change03_discovery.part_b analyze --data DATA --output OUT
"""
from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

import numpy as np
import pandas as pd
from beartype import beartype
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from experiments.rebuttal.change03_discovery import cards, common, data, rules, scoring
from experiments.rebuttal.change03_discovery.part_a import LABELS, METHODS, select_runs
from experiments.rebuttal.change03_discovery.rater_cli import read_jsonl

SEED = 20261118
N_LATENTS = 32
N_RATERS = 3
PHI_CUT = 0.5
HERE = Path(__file__).resolve().parent


@beartype
def sample_latents(representation: np.ndarray, discovery: np.ndarray, method: str,
                   n: int = N_LATENTS) -> np.ndarray:
    """Draw `n` latents that are not constant on the discovery part (fixed seed)."""
    varying = np.flatnonzero(np.ptp(representation[discovery], axis=0) > 0)
    rng = np.random.default_rng(SEED + METHODS.index(method))
    return np.sort(rng.choice(varying, size=min(n, len(varying)), replace=False))


@beartype
def card_ids(method: str, n: int) -> list[str]:
    """Anonymous identifiers; a method's block of identifiers never changes."""
    pool = np.random.default_rng(SEED + 1000).permutation(9000) + 1000
    start = METHODS.index(method) * N_LATENTS
    return [f"C{value}" for value in pool[start:start + n]]


@beartype
def instructions(session: data.Session, rater_dir: Path, python: str) -> str:
    """Fill the instruction template for one rater."""
    discovery = session.discovery
    table = "\n".join(["| variable | meaning |", "|---|---|"]
                      + [f"| `{name}` | {text} |" for name, text in data.VARIABLES.items()])
    occupancy = cards.position_map(session.variables["x"][discovery],
                                   session.variables["y"][discovery],
                                   session.support["camera"][discovery])
    text = (HERE / "RATER_INSTRUCTIONS.md").read_text()
    for key, value in (("VARIABLE_TABLE", table), ("OCCUPANCY_MAP", occupancy),
                       ("WORKTREE", str(HERE.parents[2])), ("PYTHON", python),
                       ("RATER_DIR", str(rater_dir))):
        text = text.replace(key, value)
    return text


@beartype
def build(data_root: Path, runs: Path, output: Path, python: str,
          only: tuple = METHODS) -> None:
    """Sample latents, render cards and set up (or extend) the rater folders."""
    session = data.load_session(data_root)
    discovery = session.discovery
    index = np.flatnonzero(discovery)
    variables = {name: values[index] for name, values in session.variables.items()}
    support = {name: values[index] for name, values in session.support.items()}
    binnings = cards.build_binnings(variables, support)
    blocks = (session.starts[index] * data.N_BLOCKS // len(session.counts)).astype(np.int64)
    blocks = np.unique(blocks, return_inverse=True)[1]
    selected = select_runs(runs)
    key_path = output / "key" / "part_b_key.json"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key = json.loads(key_path.read_text()) if key_path.exists() else {}
    raters_path = output / "key" / "raters.json"
    rater_dirs = json.loads(raters_path.read_text()) if raters_path.exists() else \
        [str(output / "raters" / secrets.token_hex(6)) for _ in range(N_RATERS)]
    common.write_json(raters_path.with_suffix(".tmp"), {"raters": rater_dirs})
    raters_path.write_text(json.dumps(rater_dirs, indent=2))
    new_cards = {}
    for method in METHODS:
        if method not in selected or method not in only or \
                any(entry["method"] == method for entry in key.values()):
            continue
        representation = np.load(selected[method]["path"])
        latents = sample_latents(representation, discovery, method)
        rng = np.random.default_rng(SEED + 3000 + METHODS.index(method))
        for card, latent in zip(card_ids(method, len(latents)), latents):
            key[card] = dict(method=method, latent=int(latent), config=selected[method]["config"],
                             path=selected[method]["path"])
            new_cards[card] = cards.render_card(
                card, np.asarray(representation[index, latent]), variables, support, binnings,
                session.endpoints[index], blocks, rng)
        print("cards built for", method, flush=True)
    key_path.write_text(json.dumps(key, indent=2))
    reference = slice(None, None, 10)
    for number, directory in enumerate(map(Path, rater_dirs)):
        (directory / "cards").mkdir(parents=True, exist_ok=True)
        for card, text in new_cards.items():
            (directory / "cards" / f"{card}.md").write_text(text)
        order_path = directory / "order.json"
        order = json.loads(order_path.read_text()) if order_path.exists() else []
        fresh = sorted(set(new_cards) - set(order))
        rng = np.random.default_rng(SEED + 2000 + number + 10 * len(order))
        order_path.write_text(json.dumps(order + list(rng.permutation(fresh))))
        if not (directory / "behavior_reference.npz").exists():
            np.savez(directory / "behavior_reference.npz",
                     **{f"var_{name}": values[reference] for name, values in variables.items()
                        if not name.startswith("_")},
                     **{f"sup_{name}": values[reference] for name, values in support.items()})
            (directory / "INSTRUCTIONS.md").write_text(instructions(session, directory, python))
    print(json.dumps(dict(cards=len(key), new=len(new_cards), raters=rater_dirs)))


@beartype
def fleiss_kappa(ratings: np.ndarray) -> float:
    """Fleiss' kappa for a [subjects, raters] array of categorical labels."""
    categories = np.unique(ratings)
    counts = np.stack([(ratings == c).sum(1) for c in categories], 1).astype(float)
    n = ratings.shape[1]
    agreement = ((counts ** 2).sum(1) - n) / (n * (n - 1))
    expected = ((counts.sum(0) / counts.sum()) ** 2).sum()
    return float("nan") if expected == 1 else float((agreement.mean() - expected) / (1 - expected))


@beartype
def phi_matrix(masks: np.ndarray) -> np.ndarray:
    """Phi correlation between the rows of a boolean [descriptions, windows] array."""
    x = masks.astype(np.float64)
    x -= x.mean(1, keepdims=True)
    norm = np.sqrt((x ** 2).sum(1))
    norm[norm == 0] = np.inf
    return (x @ x.T) / np.outer(norm, norm)


@beartype
def cluster_features(masks: np.ndarray, cut: float = PHI_CUT) -> np.ndarray:
    """Average-linkage clusters of condition masks at |phi| >= cut; labels from 1."""
    if len(masks) == 1:
        return np.ones(1, dtype=int)
    distance = 1 - np.abs(phi_matrix(masks))
    np.fill_diagonal(distance, 0)
    distance = np.clip((distance + distance.T) / 2, 0, None)
    return fcluster(linkage(squareform(distance, checks=False), "average"),
                    t=1 - cut, criterion="distance")


@beartype
def analyze(data_root: Path, output: Path) -> None:
    """Validate every rater's answers on the held-out part and write the tables."""
    session = data.load_session(data_root)
    key = json.loads((output / "key" / "part_b_key.json").read_text())
    rater_dirs = json.loads((output / "key" / "raters.json").read_text())
    representations = {}
    rows, masks = [], []
    for rater, directory in enumerate(map(Path, rater_dirs), 1):
        served = {}
        for record in read_jsonl(directory / "log.jsonl"):
            if record["event"] == "serve":
                served.setdefault(record["card"], record["time"])
        for answer in read_jsonl(directory / "answers.jsonl"):
            entry = key[answer["card"]]
            row = dict(rater=rater, card=answer["card"], method=entry["method"],
                       latent=entry["latent"], interpretation=answer["interpretation"],
                       seconds=answer["time"] - served.get(answer["card"], np.nan),
                       rule=answer.get("rule", ""), direction=answer.get("direction", ""),
                       description=answer.get("description", ""), passed=False, scorable=False)
            if answer["interpretation"] == "rule":
                if entry["path"] not in representations:
                    representations[entry["path"]] = np.load(entry["path"])
                mask, valid = rules.evaluate(answer["rule"], session.variables, session.support)
                row.update(scoring.evaluate(
                    np.asarray(representations[entry["path"]][:, entry["latent"]]),
                    1 if answer["direction"] == "high" else -1, mask, valid,
                    session.discovery, session.heldout))
                row["mask_index"] = len(masks)
                masks.append(mask[session.heldout][::5])
            rows.append(row)
    table = pd.DataFrame(rows)
    table["cluster"] = 0
    validated = table[table.passed]
    if len(validated):
        stack = np.stack([masks[int(i)] for i in validated.mask_index])
        table.loc[validated.index, "cluster"] = cluster_features(stack)
    table.drop(columns=["mask_index"], errors="ignore").to_csv(output / "part_b_answers.csv",
                                                             index=False)
    methods = [m for m in METHODS if (table.method == m).any()]
    raters = sorted(table.rater.unique())
    summary = []
    for method in methods:
        group = table[table.method == method]
        per_rater = group.groupby("rater").agg(inspected=("card", "size"),
                                              interpreted=("interpretation",
                                                           lambda s: (s == "rule").sum()),
                                              validated=("passed", "sum"),
                                              seconds=("seconds", "sum"))
        wide = group.pivot(index="card", columns="rater", values="passed").reindex(columns=raters)
        complete = wide.dropna()
        interpreted = group.assign(flag=group.interpretation == "rule").pivot(
            index="card", columns="rater", values="flag").reindex(columns=raters).dropna()
        clusters = sorted(set(group.cluster[group.passed]))
        row = dict(
            method=method, label=LABELS[method], inspected=int(per_rater.inspected.max()),
            interpreted_mean=float(per_rater.interpreted.mean()),
            validated_mean=float(per_rater.validated.mean()),
            validated_min=int(per_rater.validated.min()), validated_max=int(per_rater.validated.max()),
            validated_proportion=float((per_rater.validated / per_rater.inspected).mean()),
            validated_majority=int((complete.astype(int).sum(1) >= 2).sum()),
            validated_any=int((complete.astype(int).sum(1) >= 1).sum()),
            kappa_interpreted=fleiss_kappa(interpreted.to_numpy().astype(int)) if len(interpreted) else np.nan,
            kappa_validated=fleiss_kappa(complete.to_numpy().astype(int)) if len(complete) else np.nan,
            distinct_features=len(clusters), clusters=clusters,
            seconds_per_card=float((per_rater.seconds / per_rater.inspected).mean()),
            latents_per_validated=float(per_rater.inspected.sum() / per_rater.validated.sum())
            if per_rater.validated.sum() else float("inf"),
            minutes_per_validated=float(per_rater.seconds.sum() / 60 / per_rater.validated.sum())
            if per_rater.validated.sum() else float("inf"))
        summary.append(row)
    wide_all = table.pivot(index="card", columns="rater", values="passed").reindex(columns=raters).dropna()
    interpreted_all = table.assign(flag=table.interpretation == "rule").pivot(
        index="card", columns="rater", values="flag").reindex(columns=raters).dropna()
    overall = dict(kappa_validated=fleiss_kappa(wide_all.to_numpy().astype(int)),
                   kappa_interpreted=fleiss_kappa(interpreted_all.to_numpy().astype(int)),
                   cards_rated_by_all=int(len(wide_all)), raters=len(raters))
    features = []
    for cluster, group in table[table.passed].groupby("cluster"):
        best = group.sort_values("auroc", ascending=False).iloc[0]
        features.append(dict(cluster=int(cluster), methods=sorted(set(group.method), key=METHODS.index),
                             n_descriptions=len(group), n_latents=int(group.card.nunique()),
                             rule=best.rule, direction=best.direction, description=best.description,
                             best_auroc=float(best.auroc), best_sel=float(best.sel),
                             best_method=best.method))
    (output / "part_b_summary.json").write_text(json.dumps(
        dict(methods=summary, overall=overall, features=features), indent=2, default=float))
    print(pd.DataFrame(summary).drop(columns=["clusters"]).to_string())
    print(json.dumps(overall))
    for feature in features:
        print(feature["cluster"], feature["methods"], feature["direction"], feature["rule"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--runs", type=Path, required=True)
    build_parser.add_argument("--python", default="/nfs/nhome/live/jbhagat/nldisco/.venv/bin/python")
    build_parser.add_argument("--only", nargs="*", default=list(METHODS))
    analyze_parser = commands.add_parser("analyze")
    for sub in (build_parser, analyze_parser):
        sub.add_argument("--data", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.data, args.runs, args.output, args.python, tuple(args.only))
    else:
        analyze(args.data, args.output)


if __name__ == "__main__":
    main()
