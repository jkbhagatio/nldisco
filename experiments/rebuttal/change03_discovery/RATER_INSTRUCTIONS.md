# Instructions for raters

You are one of several independent raters in a study of latent variables learned
from neural recordings. Work alone. Do not look for other raters' work and do not
read or list any file other than this one: everything you need is printed by the
two commands at the end. Do not write scripts to produce answers; read each card
yourself.

## Background

A mouse lived for 18 hours in an arena that contains its nest and one foraging
patch. At the patch the mouse can run on a wheel; running a set distance triggers
a food pellet. A camera above the arena tracked the mouse and neural activity was
recorded throughout.

Several models were trained on the neural activity alone, with no behavior, and
each produced many latent variables: one number for every 0.4-second window of
time. You will inspect latents one at a time. They come from different models and
are shown in random order. You are not told which model a latent comes from, and
it does not matter for your task.

## Your task

Each latent comes with one card that summarises how its value relates to the
animal's behavior. For each card, decide whether the latent has a clear behavioral
interpretation.

- If it does, write the interpretation as a **rule**: a condition on the behavior
  variables such that the latent is in its HIGH state when the condition holds and
  not otherwise (direction `high`), or in its LOW state when the condition holds
  and not otherwise (direction `low`).
- If it does not, answer **none**. This is a normal answer. Do not force an
  interpretation.

Your rule will be tested on a separate part of the recording that you cannot see.
It passes only if the latent reliably separates the windows where the rule holds
from all other windows: the latent must be in the stated state in most windows
where the rule holds, and at least three times less often in windows where it
does not. A rule that fails counts against you just as a missed interpretation
does, so answer `none` when the card does not support a rule, and give a rule when
it does. Spend a similar, modest effort on every card.

## Reading a card

- **Values**: percentiles of the latent, and the definition of its HIGH state and
  (if it has one) its LOW state. Some latents sit at a floor most of the time and
  are HIGH whenever they leave it; others vary continuously, and then HIGH is the
  top 5% and LOW the bottom 5% of values. The card also gives the number and
  duration of HIGH episodes and the share of HIGH windows in each consecutive
  block of the recording; an interpretation should hold across blocks.
- **Tuning tables**, one per behavior variable. The header row gives the bins.
  `% windows` is how much of the recording falls in each bin. `mean rank` is the
  average rank of the latent's value in that bin on a 0-100 scale (50 means no
  preference; well above 50 means the latent is elevated there, well below 50
  means it is reduced). `% HIGH` is the share of that bin's windows in which the
  latent is HIGH, and `% LOW` likewise. `-` marks bins with too few windows.
- **Position map**: the share of windows that are HIGH (or LOW) in each 100-px
  cell of the arena; rows are y, columns are x, and labels give the lower edge of
  each cell.
- **Example windows**: behavior at a few randomly chosen HIGH (or LOW) episodes.

A good rule names the bins where the latent is clearly elevated (or reduced) and
leaves out the bins where it is not. Use the smallest set of variables that
explains the latent; combine two variables with `and` when the card shows that
both are needed. The rule need not reproduce the 5% cut-off exactly: it should
say when the latent is elevated (direction `high`) or reduced (direction `low`)
compared with the rest of the recording.

## Behavior variables

All variables are read at the end of each 0.4-second window.

VARIABLE_TABLE

Share of all windows in each 100-px cell of the arena (rows y, columns x):

```
OCCUPANCY_MAP
```

## Rule language

A rule is one expression that uses the variable names above, numbers, the
comparisons `<  <=  >  >=`, the words `and`, `or`, `not`, parentheses, arithmetic
(`+ - * /`), `abs(...)` and `hypot(a, b)`. Examples:

- `speed >= 100`
- `patch_distance <= 100 and wheel_speed <= 0.02`
- `abs(heading) >= 135 and speed >= 50` (moving left)
- `hypot(x - 400, y - 230) <= 80` (within 80 px of a location)
- `time_since_pellet <= 2`

`heading` and `heading_error_to_patch` are undefined when the animal moves slower
than 20 px/s, and the pellet clocks are undefined before the first or after the
last pellet; a comparison with an undefined value is false.

## Commands

Run both commands from `WORKTREE`.

Show the next card:

```
PYTHON -m experiments.rebuttal.change03_discovery.rater_cli next --dir RATER_DIR
```

Answer the card that was just shown (use its identifier):

```
PYTHON -m experiments.rebuttal.change03_discovery.rater_cli submit --dir RATER_DIR --card CARD_ID --none
PYTHON -m experiments.rebuttal.change03_discovery.rater_cli submit --dir RATER_DIR --card CARD_ID --direction high --rule "speed >= 100" --description "fast locomotion"
```

The tool checks that the rule can be read and tells you in how many windows it
holds; it never tells you whether the rule is right. If it rejects a rule, fix the
rule and submit again. Cards are answered strictly one at a time and cannot be
revisited. Repeat `next` and `submit` until `next` reports that all cards are
answered, then stop and reply with the single word `finished`.
