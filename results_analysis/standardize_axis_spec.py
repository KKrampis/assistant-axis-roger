"""Add standardized ``"This means..."``-form pole descriptions to an
auto-generated axis spec, so the desc+inst judge sees the project
standard format.

Auto-generated axis specs from
:mod:`results_analysis.infer_axis_description` typically open their pole
descriptions with phrases like ``"This pole represents..."`` or
``"Someone who is..."``, which describe the *pole as a category* rather
than the *behavior of an entity at that pole*. The desc+inst judge
prompt template (see ``RUBRIC_STATIC`` in
:mod:`results_analysis.axis_judge_correlation`) shows pole text in the
slot ``"+3 (strong positive pole): {pos_pole}"``; the project convention
is to fill that slot with a behavioral description starting
``"This means [verb-ing/being]..."`` so the judge sees a clean drop-in.

This shim runs one Claude Sonnet call per axis spec to rephrase
``pos_pole`` and ``neg_pole`` into the standard form, **adding** them as
new fields ``pos_pole_standardized`` and ``neg_pole_standardized``
alongside the originals (which are preserved verbatim for inspection /
provenance). All other fields (``axis_name``, ``pos_examples``,
``neg_examples``, ``_metadata``, etc.) are passed through unchanged.

The shim is idempotent: if ``pos_pole_standardized`` already exists in
the input spec, the spec is returned untouched (no API call).

Empirical pattern across 16 auto-generated specs (PCs at slot 3, layer
25, L=2 shear, both ``glossary`` and ``inline`` styles, Apr 2026)::

    "This pole represents..."   x 8
    "This end represents..."    x 6
    "Someone who..."            x 1
    "This means being..."       x 1   (already in standard form -- pass-through)

Pipeline position::

    infer_axis_description.py        this shim                  axis_judge_correlation.py
    --------------------------       ---------------------      -------------------------
    sorted projection list  ->       raw axis spec JSON   ->    extended JSON  ->  judge run
    (Opus, with thinking)            (Sonnet rephrase)          (use *_standardized for --pos_pole etc.)

By default :mod:`infer_axis_description` invokes this shim
automatically, so a spec with both fields is the normal output. Pass
``--no_standardize`` to that tool to skip it; pass ``--skip_standardize``
to this tool to bulk-rebuild the original-only form.

Usage::

    # Standalone CLI (single file)
    uv run python results_analysis/standardize_axis_spec.py \\
      --input raw_spec.json --output spec.json

    # In-place, several files at once
    uv run python results_analysis/standardize_axis_spec.py \\
      --in_place roger/pc_axis_describer_sweep/*/spec.json

    # Programmatic
    from results_analysis.standardize_axis_spec import standardize_axis_spec
    spec_extended = standardize_axis_spec(spec)

After standardization, drive the judge run with::

    SPEC=$(cat spec.json)
    uv run python results_analysis/axis_judge_correlation.py \\
      --axis_file my_axis.pt --layer 25 --whiten_K 2 \\
      --axis_name "$(jq -r .axis_name <<<"$SPEC")" \\
      --pos_pole "$(jq -r .pos_pole_standardized <<<"$SPEC")" \\
      --neg_pole "$(jq -r .neg_pole_standardized <<<"$SPEC")" \\
      --pos_examples "$(jq -r '.pos_examples | join(",")' <<<"$SPEC")" \\
      --neg_examples "$(jq -r '.neg_examples | join(",")' <<<"$SPEC")" \\
      --provider openai --score_descriptions --score_instructions ...

Note the ``_standardized`` suffix on the pole fields. The unsuffixed
``pos_pole`` / ``neg_pole`` remain available for inspection / cross-check.

Cost: ~$0.002 per spec (Sonnet 4.x, ~500 input + 400 output tokens).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 1500


SYSTEM_PROMPT = (
    "You rephrase pole descriptions of semantic axes into the project's "
    "standard form: 'This means [verb-ing/being]...' which describes the "
    "behavior or character of an entity scoring at that pole, rather than "
    "describing the pole as a category. You preserve the meaning, the "
    "specificity, and the length of the original; you only change the "
    "framing."
)


# Multi-shot examples covering every opener we've seen in auto-generated specs,
# plus the canonical existing-corpus form (a no-op example).
USER_PROMPT_TEMPLATE = """\
You are rephrasing the positive and negative pole descriptions of a semantic
axis into the project's standard form. Below are examples covering every
input form we've encountered.

# Standard form

The standard form starts with `This means [verb-ing/being]...` and describes
the behavior, attitudes, or character of an entity that scores at that pole
of the axis. Compare:

  RAW (meta-categorical -- describes the POLE as a class):
    "This pole represents beings dedicated to reliably helping others..."
  STANDARD (behavioral -- describes ENTITY at the pole):
    "This means being dedicated to reliably helping others..."

  RAW (entity-noun-phrase):
    "Someone who is sincerely dedicated to serving others..."
  STANDARD:
    "This means being sincerely dedicated to serving others..."

# Examples (drawn from real auto-generated specs)

## Example 1: "This pole represents..." form

INPUT_POS_POLE:
This pole represents communication and behavior characterized by indirectness, strategic ambiguity, and layered concealment. Concepts here favor evasion, hedging, oblique expression, and keeping true intentions partially hidden.

INPUT_NEG_POLE:
This pole represents communication and behavior characterized by unmediated directness, raw intensity, and unfiltered expression. Concepts here deliver their content without strategic softening.

OUTPUT:
<pos_pole>This means communicating and behaving with indirectness, strategic ambiguity, and layered concealment, favoring evasion, hedging, oblique expression, and keeping true intentions partially hidden.</pos_pole>
<neg_pole>This means communicating and behaving with unmediated directness, raw intensity, and unfiltered expression, delivering content without strategic softening.</neg_pole>

## Example 2: "This end represents..." form

INPUT_POS_POLE:
This end represents beings dedicated to reliably helping, guiding, healing, and nurturing others through structured, earnest care. They are trustworthy, benevolent, and focused on others' wellbeing.

INPUT_NEG_POLE:
This end represents beings who mock, disrupt, provoke, or harm rather than serve, communicating through sarcasm, irreverence, and cutting wit, or acting out of selfishness, cruelty, and chaotic self-expression.

OUTPUT:
<pos_pole>This means being dedicated to reliably helping, guiding, healing, and nurturing others through structured, earnest care, behaving in ways that are trustworthy, benevolent, and focused on others' wellbeing.</pos_pole>
<neg_pole>This means mocking, disrupting, provoking, or harming rather than serving, communicating through sarcasm, irreverence, and cutting wit, or acting out of selfishness, cruelty, and chaotic self-expression.</neg_pole>

## Example 3: "Someone who..." form

INPUT_POS_POLE:
Someone who is sincerely dedicated to serving, helping, guiding, and nurturing others, approaching interactions with genuine compassion, patient instruction, and a wholehearted desire to improve others' situations.

INPUT_NEG_POLE:
Someone who communicates through cutting wit, biting mockery, and irreverent detachment, treating serious matters with casual dismissiveness or sharp-tongued humor.

OUTPUT:
<pos_pole>This means being sincerely dedicated to serving, helping, guiding, and nurturing others, approaching interactions with genuine compassion, patient instruction, and a wholehearted desire to improve others' situations.</pos_pole>
<neg_pole>This means communicating through cutting wit, biting mockery, and irreverent detachment, treating serious matters with casual dismissiveness or sharp-tongued humor.</neg_pole>

## Example 4: Already standard -- pass through unchanged

INPUT_POS_POLE:
This means presenting information accurately and completely, avoiding lies, fabrication, or misrepresentation of facts, and correcting misunderstandings rather than exploiting them.

INPUT_NEG_POLE:
This means deliberately lying, fabricating information, and misrepresenting facts to mislead others about what is true.

OUTPUT:
<pos_pole>This means presenting information accurately and completely, avoiding lies, fabrication, or misrepresentation of facts, and correcting misunderstandings rather than exploiting them.</pos_pole>
<neg_pole>This means deliberately lying, fabricating information, and misrepresenting facts to mislead others about what is true.</neg_pole>

# Now rephrase the following

INPUT_POS_POLE:
{pos_pole}

INPUT_NEG_POLE:
{neg_pole}

Output exactly two tagged blocks (and no other text) with the rephrased
versions. Preserve all of the meaning and specificity of the input; preserve
the approximate length; only change the framing to start with 'This means...'.
"""


def build_prompt(pos_pole: str, neg_pole: str) -> str:
    """Build the user prompt that asks the LLM to rephrase the
    pos/neg pole descriptions into the standardized "This means..."
    form.

    Display-form note (see AGENT_NOTES.md "File-name vs
    display-name convention" / "LLM prompts are display sites"):
    this builder injects long-form pole *descriptions* (full
    sentences), not bare entity names, so it is intrinsically
    immune to the underscore-leak issue that affects
    ``axis_judge_correlation.py``'s rubric builders.  Kept noted
    here so a future refactor that adds entity-name fields is
    aware of the convention.
    """
    return USER_PROMPT_TEMPLATE.format(
        pos_pole=pos_pole.strip(),
        neg_pole=neg_pole.strip(),
    )


def _extract_tag(text: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    return m.group(1).strip() if m else None


def parse_response(text: str) -> tuple[str, str]:
    pos = _extract_tag(text, "pos_pole")
    neg = _extract_tag(text, "neg_pole")
    if pos is None:
        raise ValueError(f"Missing <pos_pole> in response: {text[:200]!r}")
    if neg is None:
        raise ValueError(f"Missing <neg_pole> in response: {text[:200]!r}")
    return pos, neg


async def _call_sonnet(prompt: str, *, model: str, max_tokens: int) -> str:
    """One Anthropic call at temperature 0; returns the joined text content."""
    import anthropic

    if os.getenv("ANTHROPIC_API_KEY") is None:
        raise EnvironmentError("ANTHROPIC_API_KEY not set (load via .env)")

    client = anthropic.AsyncAnthropic()
    resp = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            t = getattr(block, "text", None)
            if t:
                parts.append(t)
    if not parts:
        raise RuntimeError("Anthropic returned no text content blocks")
    return "".join(parts)


def standardize_axis_spec(
    spec: dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    force: bool = False,
) -> dict[str, Any]:
    """Add standardized pole descriptions to an axis spec.

    The original ``pos_pole`` and ``neg_pole`` fields are preserved
    unchanged; the ``"This means..."``-form rephrasings are added as new
    fields ``pos_pole_standardized`` and ``neg_pole_standardized``.

    Idempotent: if both standardized fields already exist on the input
    spec, the spec is returned untouched (no API call), unless
    ``force=True``.

    Parameters
    ----------
    spec : dict
        An axis spec, typically the JSON output of
        :mod:`infer_axis_description`. Must contain ``pos_pole`` and
        ``neg_pole`` keys; other fields are passed through.
    model : str, optional
        Anthropic model name. Default ``claude-sonnet-4-6``.
    max_tokens : int, optional
        Output token budget. Default ``1500`` (covers two rephrased poles
        plus tag scaffolding).
    force : bool, optional
        If True, re-run even if standardized fields are already present.

    Returns
    -------
    dict
        Same keys as ``spec``, plus ``pos_pole_standardized`` and
        ``neg_pole_standardized``.
    """
    if (
        not force
        and isinstance(spec.get("pos_pole_standardized"), str)
        and isinstance(spec.get("neg_pole_standardized"), str)
    ):
        return dict(spec)
    return asyncio.run(_standardize_axis_spec_async(
        spec=spec, model=model, max_tokens=max_tokens,
    ))


async def _standardize_axis_spec_async(
    *,
    spec: dict[str, Any],
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    pos = spec.get("pos_pole")
    neg = spec.get("neg_pole")
    if not isinstance(pos, str) or not isinstance(neg, str):
        raise ValueError("spec must have string pos_pole and neg_pole fields")

    prompt = build_prompt(pos, neg)
    logger.info(
        f"calling {model} to rephrase {len(pos)+len(neg)} chars of pole text"
    )
    text = await _call_sonnet(prompt, model=model, max_tokens=max_tokens)

    try:
        new_pos, new_neg = parse_response(text)
    except ValueError as e:
        logger.warning(f"first parse failed ({e}); retrying with clarifying note")
        retry = prompt + (
            "\n\n[Note: a previous attempt did not match the schema. "
            "Reply with EXACTLY two blocks: <pos_pole>...</pos_pole> and "
            "<neg_pole>...</neg_pole>, with no other text.]"
        )
        text = await _call_sonnet(retry, model=model, max_tokens=max_tokens)
        new_pos, new_neg = parse_response(text)

    out = dict(spec)
    out["pos_pole_standardized"] = new_pos
    out["neg_pole_standardized"] = new_neg
    return out


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Add standardized 'This means...'-form pole descriptions to "
            "axis spec(s). Shim between infer_axis_description.py and "
            "axis_judge_correlation.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path,
                     help="Single input axis spec JSON to standardize.")
    src.add_argument("--in_place", type=Path, nargs="+",
                     help="One or more spec JSON paths to standardize and "
                          "overwrite (idempotent: skips files that already "
                          "have *_standardized fields unless --force).")
    p.add_argument("--output", type=Path,
                   help="Output path (required with --input). With --in_place "
                        "this is unused.")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if *_standardized fields already exist.")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Anthropic model.")
    p.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help="Max output tokens.")
    a = p.parse_args(argv)
    if a.input and not a.output:
        p.error("--input requires --output")
    return a


def _process_single(input_path: Path, output_path: Path, *, force: bool,
                    model: str, max_tokens: int) -> bool:
    spec = json.loads(input_path.read_text())
    if (
        not force
        and isinstance(spec.get("pos_pole_standardized"), str)
        and isinstance(spec.get("neg_pole_standardized"), str)
    ):
        print(f"  SKIP {input_path}: already has *_standardized")
        if input_path != output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(spec, indent=2))
        return False
    out = standardize_axis_spec(spec, model=model, max_tokens=max_tokens, force=force)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    print(f"  + {input_path} -> {output_path}")
    print(f"      pos_standardized: {out['pos_pole_standardized'][:90]}...")
    return True


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    if args.input:
        n = _process_single(args.input, args.output,
                             force=args.force, model=args.model,
                             max_tokens=args.max_tokens)
        print(f"\nDone: {1 if n else 0} call, {1 if not n else 0} skip.")
    else:
        n_done = 0; n_skip = 0
        for p in args.in_place:
            if _process_single(p, p, force=args.force, model=args.model,
                                max_tokens=args.max_tokens):
                n_done += 1
            else:
                n_skip += 1
        print(f"\nDone: {n_done} updated, {n_skip} skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
