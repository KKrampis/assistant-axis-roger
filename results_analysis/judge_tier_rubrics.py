"""Rubric variants for judge_tier_cost_eval.py's description-mode axis judging
(``--rubric``, 2026-11-01).

The tier evaluation compares judges on one prompt: the v3 descriptions rubric
of ``axis_judge_correlation.build_static_prompt``, written on 2026-04-20 in the
style of 2023-era judge prompts (a seven-point table whose 0 row reads
"Neutral, ambiguous, or does not map onto this axis", two to three sentences of
reasoning, a SCORE line).  On 2026-11-01 the newer cheap judges were found to
answer 0 for 52-70% of entities against gpt-4.1-mini's 32%, and across judges
the share of zeros tracks lower rho (Pearson -0.85): the activation projection
orders every entity, and a 0 ties all the ones the judge would not place.
These variants test whether a rubric written for current models changes that.

Each variant: ``build(axis_spec, pair_type, etype, name, desc)`` -> prompt;
``parse(text)`` -> score or None; ``version`` (recorded in each cell's
provenance and answer log, so cells of different rubrics never mix).

- ``v3``: the existing rubric, unchanged (``build_static_prompt``,
  ``parse_signed_score`` with its first-integer fallback), so that existing
  cells stay comparable.
- ``v3fix``: v3 with one fix.  The reference line named the pole entities by
  the type of the entity being scored ("roles that score near -3 include:
  indecisive" when a role is scored on a trait axis); it now uses the poles'
  own type.  Parser unchanged.
- ``c1`` to ``c4``: contemporary rubrics.  They say what the score is for
  (rank among several hundred entities, compared with an independent
  measurement; relative position and small differences matter a great
  deal), give a standard for indirect cases (typical behaviour, values and
  associations), keep 0 for genuinely balanced entities and ask for the lean
  otherwise, keep reasoning before the score (project rule), and read the
  score strictly: exactly one ``SCORE: <int>`` in range, else unparseable.
  ``c1``: -3..+3, one or two sentences.  ``c2``: -10..+10, one or two
  sentences.  ``c3``: -3..+3, one short sentence.  ``c4``: -10..+10, one
  short sentence.  Round 2, around c4: ``c5`` -100..+100; ``c6`` a few words
  of reasoning; ``c7`` a condensed prompt (input is most of a cheap judge's
  cost).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

from assistant_axis.entity_id import parse_entity_id
from results_analysis.axis_judge_correlation import (
    AxisSpec,
    RUBRIC_VERSION,
    build_static_prompt,
    display_form_name,
    parse_signed_score,
    _entity_words,
)


@dataclass(frozen=True)
class RubricSpec:
    name: str
    version: str
    build: Callable[[AxisSpec, str, str, str, str], str]
    parse: Callable[[Optional[str]], Optional[float]]
    lo: int
    hi: int
    summary: str
    #: Batched judging (--entities_per_call): ``build_batch(axis_spec, pair_type, [(etype, name, desc), ...])``
    #: and ``parse_batch(text, n) -> {1-based index: score or None}``; None when the rubric judges one
    #: entity per call only.
    build_batch: Optional[Callable[[AxisSpec, str, Sequence[Tuple[str, str, str]]], str]] = None
    parse_batch: Optional[Callable[[Optional[str], int], Dict[int, Optional[float]]]] = None


# ---------------------------------------------------------------------------
# v3 and v3fix
# ---------------------------------------------------------------------------

def _build_v3(spec: AxisSpec, pair_type: str, etype: str, name: str, desc: str) -> str:
    return build_static_prompt(spec, etype, name, desc)


def _build_v3fix(spec: AxisSpec, pair_type: str, etype: str, name: str, desc: str) -> str:
    """v3 with the reference line naming the poles by their own type."""
    prompt = build_static_prompt(spec, etype, name, desc)
    sing, plur, sing_t, plur_t = _entity_words(etype)
    psing, pplur, psing_t, pplur_t = _entity_words(pair_type)
    old = (f"For reference, {plur} that score near -3 include: ", f"\n{plur_t} that score near +3 include: ")
    new = (f"For reference, {pplur} that score near -3 include: ", f"\n{pplur_t} that score near +3 include: ")
    for o, n in zip(old, new):
        assert prompt.count(o) == 1, (o, prompt[:400])
        prompt = prompt.replace(o, n)
    return prompt


# ---------------------------------------------------------------------------
# Contemporary rubrics
# ---------------------------------------------------------------------------

_INTRO = (
    "We are measuring where several hundred roles and traits sit along one semantic axis, and comparing your "
    "judgements with an independent measurement of the same roles and traits.  Each score you give is used to "
    "rank this {sing} among all of them on the axis, so relative position matters a great deal, and small "
    "differences are worth getting right.\n\n"
    "The axis runs from {neg} ({lo}) to {pos} ({hi}):\n"
    "- {neg}: {neg_pole}\n"
    "- {pos}: {pos_pole}\n\n"
    "The {sing} to place is {name}: {desc}\n\n"
    "Consider how {someone} typically behaves, what they value and what they are associated with, and decide "
    "which end of the axis that sits closer to, and how far.  Most roles and traits relate to an axis only "
    "partly or indirectly; still say which way such a {sing} leans.  "
)
_SCALE_3 = ("Use +1 or -1 for a slight lean, +2 or -2 for a clear one, and +3 or -3 when the {sing} is defined by "
            "that end.  Use 0 only when it is genuinely balanced between the two ends.\n\n")
_SCALE_10 = ("Use the whole range: a score of 1 to 3 in either direction for a slight lean, 4 to 6 for a clear one, "
             "7 to 9 for a strong one, and 10 only when the {sing} is defined by that end.  Use 0 only when it is "
             "genuinely balanced between the two ends.\n\n")
_REASON_TWO = "Give one or two sentences of reasoning, then on a new line write exactly: SCORE: <integer from {lo} to {hi_signed}>"
_REASON_ONE = "Give one short sentence of reasoning, then on a new line write exactly: SCORE: <integer from {lo} to {hi_signed}>"
_SOMEONE = {"role": "someone in this role", "trait": "someone with this trait"}
# Round 2 (2026-11-01), around c4, the development round's leader: a finer scale, shorter
# replies, and a condensed prompt (input is about two thirds of a cheap judge's cost).
_SCALE_100 = ("Use the whole range: roughly 1 to 30 in either direction for a slight lean, 31 to 60 for a clear "
              "one, 61 to 90 for a strong one, and above 90 only when the {sing} is defined by that end.  Use 0 only "
              "when it is genuinely balanced between the two ends.\n\n")
_REASON_FEW = "Give a few words of reasoning, then on a new line write exactly: SCORE: <integer from {lo} to {hi_signed}>"
# Round 3 (2026-11-01, Fable's review): the metric is a rank correlation, so ask for
# the rank directly (c8, a percentile among all roles and traits: no slight/clear/strong
# categories to quantise on, and the ranking frame is explicit); and test whether
# forcing a lean on entities with no relation to the axis adds noise (c9, a
# calibrated 0: about one entity in ten).
_PERCENTILE = (
    "We are placing several hundred roles and traits along one semantic axis and comparing your placements with "
    "an independent measurement of the same roles and traits, so what matters is where this {sing} ranks among "
    "all of them, and small differences are worth getting right.\n\n"
    "The axis runs from {neg} to {pos}:\n"
    "- {neg}: {neg_pole}\n"
    "- {pos}: {pos_pole}\n\n"
    "The {sing} to place is {name}: {desc}\n\n"
    "Consider how {someone} typically behaves, what they value and what they are associated with.  Then estimate "
    "this {sing}'s percentile among all the roles and traits on this axis: 0 means further toward {neg} than any "
    "other, 100 means further toward {pos} than any other, 50 means exactly in the middle of them all.  Most roles "
    "and traits relate to an axis only partly or indirectly, so most percentiles sit somewhere between 20 and 80; "
    "still place every {sing} on one side or the other of 50 unless it is genuinely balanced.\n\n"
    "Give one short sentence of reasoning, then on a new line write exactly: SCORE: <integer from 0 to 100>"
)
# c10 (2026-11-01): c8 without the "between 20 and 80" hint, which Haiku 5.5 took as a
# prior and parked 59% of entities at about 40 or 60 (c8 rho 0.594 against 0.619 for c5).
_PERCENTILE_FREE = _PERCENTILE.replace(
    "Most roles and traits relate to an axis only partly or indirectly, so most percentiles sit somewhere between "
    "20 and 80; still place every {sing} on one side or the other of 50 unless it is genuinely balanced.",
    "Most roles and traits relate to an axis only partly or indirectly, but a percentile is a rank among all of them, "
    "so even a partial relation places a {sing} well away from 50: use the whole range from 0 to 100, and put every "
    "{sing} on one side or the other of 50 unless it is genuinely balanced.")
assert _PERCENTILE_FREE != _PERCENTILE
_SCALE_10_CAL = ("Use the whole range: a score of 1 to 3 in either direction for a slight lean, 4 to 6 for a clear one, "
                 "7 to 9 for a strong one, and 10 only when the {sing} is defined by that end.  Roughly one role or "
                 "trait in ten has no real relation to a given axis; give those 0.  For the rest, give the lean even "
                 "when it is slight.\n\n")
_CONDENSED = (
    "Your score ranks this {sing} among several hundred roles and traits on one semantic axis, and is compared "
    "with an independent measurement, so relative position and small differences matter a great deal.\n\n"
    "Axis: {neg} ({lo}) to {pos} ({hi}).\n"
    "- {neg}: {neg_pole}\n"
    "- {pos}: {pos_pole}\n\n"
    "{Sing}: {name}: {desc}\n\n"
    "Judge where {someone} sits, from how they typically behave, what they value and what they are associated "
    "with.  Most roles and traits relate to an axis only indirectly; still give the lean.  Scale: 1 to 3 either "
    "way for a slight lean, 4 to 6 clear, 7 to 9 strong, 10 only if defined by that end; 0 only if genuinely "
    "balanced.\n\n"
    "One short sentence of reasoning, then on a new line exactly: SCORE: <integer from {lo} to {hi_signed}>"
)


def shown_name(name: str) -> str:
    """The entity name as a prompt shows it: the kind suffix of a collision key
    (``amateur|R``) dropped, underscores to spaces.  v3 shows the key as it is
    (``display_form_name`` keeps the suffix), so on the 38 suffix-keyed axes
    every v3 prompt reads "**amateur|R**: ..."; the judges cope, and v3 is
    left as it is for comparability with the existing cells."""
    return display_form_name(parse_entity_id(name).name if "|" in name else name)


def _contemporary(scale: str, reason: str, lo: int, hi: int, template: str = "", lo_label: str = ""):
    template = template or (_INTRO + scale + reason)

    def build(spec: AxisSpec, pair_type: str, etype: str, name: str, desc: str) -> str:
        sing, plur, _, _ = _entity_words(etype)
        neg, pos = spec.neg_examples[0], spec.pos_examples[0]
        return template.format(
            sing=sing, Sing=sing.capitalize(), plur=plur, lo=lo_label or lo, hi=f"+{hi}" if lo < 0 else hi,
            hi_signed=f"+{hi}" if lo < 0 else hi, someone=_SOMEONE[sing],
            neg=display_form_name(neg), pos=display_form_name(pos),
            neg_pole=spec.neg_pole, pos_pole=spec.pos_pole,
            name=shown_name(name), desc=desc)
    return build


# Batched judging (2026-11-01, Roger: "rho makes it fairly easy to see if it helps or hurts,
# and reshuffling lets us test the noise"): several entities of one axis per call, so the
# judge has a comparison frame and the rubric is shared; the crosstalk between batch-mates is
# measured by judging each entity in two different random batches (--shuffle_seed).
_BATCH = (
    "We are placing several hundred roles and traits along one semantic axis and comparing your placements with "
    "an independent measurement of the same roles and traits, so what matters is where each one ranks among all "
    "of them, and small differences are worth getting right.\n\n"
    "The axis runs from {neg} ({lo}) to {pos} ({hi}):\n"
    "- {neg}: {neg_pole}\n"
    "- {pos}: {pos_pole}\n\n"
    "Place each of these {n} roles and traits on the axis.  Consider how the person described typically behaves, "
    "what they value and what they are associated with.  Most roles and traits relate to an axis only partly or "
    "indirectly; still say which way each one leans.  Use the whole range: roughly 1 to 30 in either direction "
    "for a slight lean, 31 to 60 for a clear one, 61 to 90 for a strong one, and above 90 only when the role or "
    "trait is defined by that end.  Use 0 only when one is genuinely balanced between the two ends.\n\n"
    "{items}\n\n"
    "Answer with exactly {n} lines, one per role or trait, in the order given: its number, one short sentence of "
    "reasoning, then SCORE: <integer from {lo} to {hi}>."
)


def _build_batch(spec: AxisSpec, pair_type: str, group: Sequence[Tuple[str, str, str]]) -> str:
    neg, pos = spec.neg_examples[0], spec.pos_examples[0]
    items = "\n".join(f"{k}. {shown_name(name)} ({_entity_words(etype)[0]}): {desc}"
                      for k, (etype, name, desc) in enumerate(group, 1))
    return _BATCH.format(n=len(group), lo=-100, hi="+100", neg=display_form_name(neg), pos=display_form_name(pos),
                         neg_pole=spec.neg_pole, pos_pole=spec.pos_pole, items=items)


# The batch form of the percentile rubric (c10's wording), b10.
_BATCH_PCT = (
    "We are placing several hundred roles and traits along one semantic axis and comparing your placements with "
    "an independent measurement of the same roles and traits, so what matters is where each one ranks among all "
    "of them, and small differences are worth getting right.\n\n"
    "The axis runs from {neg} to {pos}:\n"
    "- {neg}: {neg_pole}\n"
    "- {pos}: {pos_pole}\n\n"
    "For each of these {n} roles and traits, consider how the person described typically behaves, what they value "
    "and what they are associated with, then estimate its percentile among all the roles and traits on this axis: "
    "0 means further toward {neg} than any other, 100 means further toward {pos} than any other, 50 means exactly "
    "in the middle of them all.  Most roles and traits relate to an axis only partly or indirectly, but a "
    "percentile is a rank among all of them, so even a partial relation places one well away from 50: use the "
    "whole range from 0 to 100, and put every one on one side or the other of 50 unless it is genuinely "
    "balanced.\n\n"
    "{items}\n\n"
    "Answer with exactly {n} lines, one per role or trait, in the order given: its number, one short sentence of "
    "reasoning, then SCORE: <integer from 0 to 100>."
)


def _build_batch_pct(spec: AxisSpec, pair_type: str, group: Sequence[Tuple[str, str, str]]) -> str:
    neg, pos = spec.neg_examples[0], spec.pos_examples[0]
    items = "\n".join(f"{k}. {shown_name(name)} ({_entity_words(etype)[0]}): {desc}"
                      for k, (etype, name, desc) in enumerate(group, 1))
    return _BATCH_PCT.format(n=len(group), neg=display_form_name(neg), pos=display_form_name(pos),
                             neg_pole=spec.neg_pole, pos_pole=spec.pos_pole, items=items)


_BATCH_ITEM_RE = re.compile(r"^\s*\**(\d+)[.):\]]", flags=re.MULTILINE)
_BATCH_SCORE_RE = re.compile(r"SCORE\s*:\s*\**([+-]?\d+)\b", flags=re.IGNORECASE)


def batch_parser(lo: int, hi: int):
    """``{index: score}`` for each index 1..n whose item (from its numbered line
    to the next numbered line; Haiku 5.5 puts the SCORE on the line after the
    reasoning, with a blank line between items) holds exactly one SCORE marker
    with a value in range; a missing, repeated or out-of-range index gives
    None."""
    def parse(text: Optional[str], n: int) -> Dict[int, Optional[float]]:
        out: Dict[int, Optional[float]] = {k: None for k in range(1, n + 1)}
        if not text:
            return out
        starts = [(m.start(), int(m.group(1))) for m in _BATCH_ITEM_RE.finditer(text)]
        seen: Dict[int, int] = {}
        for j, (pos, k) in enumerate(starts):
            if not 1 <= k <= n:
                continue
            block = text[pos:starts[j + 1][0] if j + 1 < len(starts) else len(text)]
            found = _BATCH_SCORE_RE.findall(block)
            seen[k] = seen.get(k, 0) + 1
            if len(found) == 1:
                v = int(found[0])
                out[k] = v if lo <= v <= hi else None
            else:
                out[k] = None
        for k, c in seen.items():
            if c != 1:
                out[k] = None
        return out
    return parse


_SCORE_RE = re.compile(r"SCORE\s*:\s*([+-]?\d+)\b", flags=re.IGNORECASE)


def strict_parser(lo: int, hi: int) -> Callable[[Optional[str]], Optional[float]]:
    """Exactly one ``SCORE: <int>`` marker with the value in [lo, hi], else None
    (as pipeline step 3 and axis_judge_correlation v5 read scores; no fallback)."""
    def parse(text: Optional[str]) -> Optional[float]:
        if not text:
            return None
        found = _SCORE_RE.findall(text)
        if len(found) != 1:
            return None
        v = int(found[0])
        return v if lo <= v <= hi else None
    return parse


RUBRICS: Dict[str, RubricSpec] = {
    "v3": RubricSpec("v3", RUBRIC_VERSION, _build_v3, parse_signed_score, -3, 3,
                     "the 2026-04-20 descriptions rubric, unchanged"),
    "v3fix": RubricSpec("v3fix", f"{RUBRIC_VERSION}-polefix", _build_v3fix, parse_signed_score, -3, 3,
                        "v3 with the reference line naming the poles by their own type"),
    "c1": RubricSpec("c1", "c1", _contemporary(_SCALE_3, _REASON_TWO, -3, 3), strict_parser(-3, 3), -3, 3,
                     "contemporary, -3..+3, one or two sentences of reasoning"),
    "c2": RubricSpec("c2", "c2", _contemporary(_SCALE_10, _REASON_TWO, -10, 10), strict_parser(-10, 10), -10, 10,
                     "contemporary, -10..+10, one or two sentences of reasoning"),
    "c3": RubricSpec("c3", "c3", _contemporary(_SCALE_3, _REASON_ONE, -3, 3), strict_parser(-3, 3), -3, 3,
                     "contemporary, -3..+3, one short sentence of reasoning"),
    "c4": RubricSpec("c4", "c4", _contemporary(_SCALE_10, _REASON_ONE, -10, 10), strict_parser(-10, 10), -10, 10,
                     "contemporary, -10..+10, one short sentence of reasoning"),
    "c5": RubricSpec("c5", "c5", _contemporary(_SCALE_100, _REASON_ONE, -100, 100), strict_parser(-100, 100),
                     -100, 100, "contemporary, -100..+100, one short sentence of reasoning"),
    "c6": RubricSpec("c6", "c6", _contemporary(_SCALE_10, _REASON_FEW, -10, 10), strict_parser(-10, 10), -10, 10,
                     "contemporary, -10..+10, a few words of reasoning"),
    "c7": RubricSpec("c7", "c7", _contemporary("", "", -10, 10, template=_CONDENSED), strict_parser(-10, 10),
                     -10, 10, "contemporary, condensed prompt, -10..+10, one short sentence of reasoning"),
    "c8": RubricSpec("c8", "c8", _contemporary("", "", 0, 100, template=_PERCENTILE), strict_parser(0, 100),
                     0, 100, "contemporary, percentile among all roles and traits (0..100), one short sentence"),
    "c9": RubricSpec("c9", "c9", _contemporary(_SCALE_10_CAL, _REASON_ONE, -10, 10), strict_parser(-10, 10),
                     -10, 10, "contemporary, -10..+10, calibrated 0 (about one entity in ten), one short sentence"),
    "c10": RubricSpec("c10", "c10", _contemporary("", "", 0, 100, template=_PERCENTILE_FREE), strict_parser(0, 100),
                      0, 100, "contemporary, percentile (0..100) without the 20-80 hint, one short sentence"),
    "b10": RubricSpec("b10", "b10", _contemporary("", "", 0, 100, template=_PERCENTILE_FREE), strict_parser(0, 100),
                      0, 100, "c10 judged several entities per call (--entities_per_call), percentile 0..100",
                      build_batch=_build_batch_pct, parse_batch=batch_parser(0, 100)),
    "b5": RubricSpec("b5", "b5", _contemporary(_SCALE_100, _REASON_ONE, -100, 100), strict_parser(-100, 100),
                     -100, 100, "c5 judged several entities per call (--entities_per_call), -100..+100",
                     build_batch=_build_batch, parse_batch=batch_parser(-100, 100)),
}


def get(name: str) -> RubricSpec:
    if name not in RUBRICS:
        raise ValueError(f"unknown rubric {name!r}; choose from {', '.join(RUBRICS)}")
    return RUBRICS[name]
