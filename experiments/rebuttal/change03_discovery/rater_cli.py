"""Command-line interface for Part B raters: one card at a time, logged with times.

    python -m experiments.rebuttal.change03_discovery.rater_cli next --dir DIR
    python -m experiments.rebuttal.change03_discovery.rater_cli submit --dir DIR --card ID --none
    python -m experiments.rebuttal.change03_discovery.rater_cli submit --dir DIR --card ID \
        --direction high --rule "wheel_speed > 0.05" --description "wheel running"

The tool never reveals the method, the latent index, a score or held-out data.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from beartype import beartype

from experiments.rebuttal.change03_discovery import rules

REMINDER = """
---
Answer this card with ONE of:
  submit --dir {dir} --card {card} --none
  submit --dir {dir} --card {card} --direction high|low --rule "<rule>" --description "<one line>"
(card {position} of {total}. Give a rule only if the latent's mean rank is far from 50 where
the rule holds and near 50 elsewhere; otherwise answer --none. Full instructions, if you
need to re-read them: {dir}/INSTRUCTIONS.md)"""


@beartype
def read_jsonl(path: Path) -> list:
    """Read a JSON-lines file (empty list if it does not exist)."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] \
        if path.exists() else []


@beartype
def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


@beartype
def pending(directory: Path) -> tuple[list, list]:
    """Card order and the cards not yet answered, in order."""
    order = json.loads((directory / "order.json").read_text())
    answered = {record["card"] for record in read_jsonl(directory / "answers.jsonl")}
    return order, [card for card in order if card not in answered]


@beartype
def next_card(directory: Path) -> str:
    """Return the next unanswered card and log that it was served."""
    order, remaining = pending(directory)
    if not remaining:
        return "ALL CARDS ANSWERED. You are done; thank you."
    card = remaining[0]
    append_jsonl(directory / "log.jsonl", dict(event="serve", card=card, time=time.time()))
    text = (directory / "cards" / f"{card}.md").read_text()
    return text + REMINDER.format(dir=directory, card=card, total=len(order),
                                  position=len(order) - len(remaining) + 1)


@beartype
def submit(directory: Path, card: str, none: bool, direction: Optional[str],
           rule: Optional[str], description: Optional[str]) -> tuple[bool, str]:
    """Validate and record an answer. Returns (accepted, message)."""
    order, remaining = pending(directory)
    if not remaining:
        return False, "All cards are already answered."
    if card != remaining[0]:
        return False, (f"Card {card} is not the current card. The current card is "
                       f"{remaining[0]}; run `next` to see it.")
    record = dict(card=card, time=time.time())
    if none:
        if rule or direction:
            return False, "Give either --none or a rule with a direction, not both."
        record.update(interpretation="none")
        message = ""
    else:
        if not rule or direction not in ("high", "low"):
            return False, "A rule needs --direction high|low and --rule \"...\"."
        if not description:
            return False, "Add a one-line --description of the interpretation."
        try:
            reference = np.load(directory / "behavior_reference.npz")
            variables = {key[4:]: reference[key] for key in reference.files if key.startswith("var_")}
            support = {key[4:]: reference[key] for key in reference.files if key.startswith("sup_")}
            mask, valid = rules.evaluate(rule, variables, support)
        except rules.RuleError as error:
            return False, f"Rule not accepted: {error}"
        fraction = float(mask.sum() / max(valid.sum(), 1))
        if not 0 < mask.sum() < valid.sum():
            return False, (f"Rule not accepted: it holds in {100 * fraction:.2f}% of windows, "
                           "so it does not divide the recording. Revise it or answer --none.")
        record.update(interpretation="rule", direction=direction, rule=rule.strip(),
                      description=description.strip())
        message = f" The rule holds in {100 * fraction:.2f}% of windows."
    append_jsonl(directory / "answers.jsonl", record)
    append_jsonl(directory / "log.jsonl", dict(event="submit", card=card, time=record["time"]))
    return True, f"Recorded {card}.{message} {len(remaining) - 1} card(s) remaining; run `next`."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("next", "submit"):
        command = commands.add_parser(name)
        command.add_argument("--dir", type=Path, required=True)
    command.add_argument("--card", required=True)
    command.add_argument("--none", action="store_true")
    command.add_argument("--direction", choices=("high", "low"))
    command.add_argument("--rule")
    command.add_argument("--description")
    args = parser.parse_args()
    if args.command == "next":
        print(next_card(args.dir))
        return
    accepted, message = submit(args.dir, args.card, args.none, args.direction, args.rule,
                               args.description)
    print(message)
    sys.exit(0 if accepted else 1)


if __name__ == "__main__":
    main()
