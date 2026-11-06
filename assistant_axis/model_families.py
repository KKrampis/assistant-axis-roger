"""Per-family facts that steps 1 and 2 of the pipeline both read.

A *family* is what the chat template, the reasoning switch and the special
tokens have in common across the models of one architecture: every size and
checkpoint of Qwen3 renders a conversation the same way.  It is keyed on
``model.config.model_type`` (the ``model_type`` field of the model's
``config.json``: ``qwen3``, ``llama``, ``gemma2``, ...), not on a substring of
the model's name, so a local path or a quantised copy finds its row.

Only what is genuinely per family lives here.  The header tolerance is not
per family either: ``internals.spans.allowed_header_mismatches`` accepts one
wrong header token from three tokens on and none below (OLMo 2's two); the
TODO there says the tolerance is almost certainly unnecessary.
The header slots are *not*
listed per family: they are derived from the model's own chat template at
startup (:func:`assistant_axis.internals.chat_layout.derive_chat_layout`).
A row's ``expected_header_tokens`` is an assertion on that derivation, set
only once a smoke run has shown the derived header and Roger has read it
(today only ``qwen3``'s, from the 8slot metadata).

**Adding a family** (a new ``model_type``): add a :class:`FamilySpec` row to
:data:`FAMILIES` with ``expected_header_tokens=None``, the cells you have not
checked against the real tokenizer listed in :data:`TO_CHECK`, and the
header you expect in :data:`DRAFT_HEADERS`; run the real-tokenizer test
(``ASSISTANT_AXIS_HF_TESTS=1``, assistant_axis/tests/test_chat_layout_real.py)
and the smoke run of the plan's § 11; pin ``expected_header_tokens`` from
the header the smoke run printed.  An unknown ``model_type`` refuses to run
in both steps (:class:`UnknownFamilyError`; plan § 15 Q8).

Source: reports/pipeline/coding_plan_multi_model_extraction.md § 5.2 (the
draft rows) and its § 15 decisions of 2026-10-12.  Standard library only at
module level (the ``AutoConfig`` import is inside :func:`family_for_model`),
so importing this module stays light.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

REASONING_MODES = ("none", "template_switch", "always_on")
SYSTEM_PROMPT_MODES = ("native", "fold_into_user")
SPAN_BUILDERS = ("chatml", "generic")
RESPONSE_CLEANUPS = ("none", "strip_leaked_think", "harmony_final_channel")

# The seven tokens before a Qwen3 non-thinking assistant turn's content, as
# every 8slot vector's metadata["header_tokens"] has them (read 2026-10-12),
# and their ids in the Qwen3 tokenizer (metadata["header_ids"]).
QWEN3_NON_THINKING_HEADER_TOKENS: Tuple[str, ...] = (
    "<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n")
QWEN3_NON_THINKING_HEADER_IDS: Tuple[int, ...] = (
    151644, 77091, 198, 151667, 271, 151668, 271)


@dataclass(frozen=True)
class FamilySpec:
    """One family's row (plan § 5.2).

    ``template_kwargs`` are passed to every ``apply_chat_template`` in steps 1
    and 2; ``reasoning`` is ``"none"``, ``"template_switch"`` (switched off by
    ``template_kwargs``) or ``"always_on"``; ``reasoning_stored`` (always_on
    only) means the model's reasoning is kept in the response record and
    re-rendered by step 2; ``system_prompt`` is what the template is expected
    to do with a system message (``"native"`` or ``"fold_into_user"``; step 1
    probes it and warns on disagreement); ``span_builder`` picks the
    assistant-span code (``"chatml"``: the Qwen pattern path; ``"generic"``:
    the longest-common-prefix path); ``response_cleanup`` is what step 1 does
    to each generated text; ``bos_in_template`` and ``end_token`` are
    informational; ``expected_header_tokens``, when set, is asserted on the
    derived header; ``hf_load_kwargs`` go to step 2's Hugging Face load.
    """

    model_type: str
    template_kwargs: Mapping[str, Any]
    reasoning: str
    reasoning_stored: bool
    system_prompt: str
    span_builder: str
    response_cleanup: str
    bos_in_template: bool
    end_token: str
    expected_header_tokens: Optional[Tuple[str, ...]]
    trust_remote_code: bool = False
    hf_load_kwargs: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self):
        # Own copies, so a row never shares a dict with its caller.  Plain
        # dicts, not read-only proxies: a row must pickle (multiprocessing
        # workers receive them).  Callers pass dict(row.template_kwargs) on.
        object.__setattr__(self, "template_kwargs", dict(self.template_kwargs))
        object.__setattr__(self, "hf_load_kwargs", dict(self.hf_load_kwargs))
        if self.expected_header_tokens is not None:
            object.__setattr__(self, "expected_header_tokens",
                               tuple(self.expected_header_tokens))


FAMILIES: Dict[str, FamilySpec] = {
    "qwen3": FamilySpec(
        model_type="qwen3",
        template_kwargs={"enable_thinking": False},
        reasoning="template_switch",
        reasoning_stored=False,
        system_prompt="native",
        span_builder="chatml",
        response_cleanup="strip_leaked_think",
        bos_in_template=False,
        end_token="<|im_end|>",
        expected_header_tokens=QWEN3_NON_THINKING_HEADER_TOKENS,
        notes=("Qwen/Qwen3-32B, the 8slot tree.  Every cell verified (8slot metadata, the "
               "code steps 1 and 2 ran).  Qwen3-Instruct-2507 (no think block) derives 3 "
               "header tokens and fails this row's expectation on purpose: it is another row."),
    ),
    "llama": FamilySpec(
        model_type="llama",
        template_kwargs={},
        reasoning="none",
        reasoning_stored=False,
        system_prompt="native",
        span_builder="generic",
        response_cleanup="none",
        bos_in_template=True,
        end_token="<|eot_id|>",
        expected_header_tokens=None,
        notes=("meta-llama/Llama-3.3-70B-Instruct; gated (Roger's HF token); ~140 GB.  TODO "
               "when Llama is confirmed (plan § 15 Q6): pin the template's date "
               "('Today Date: ...' comes from strftime_now unless date_string is given), "
               "else step 2 on another day renders another prompt; the side-car's "
               "sample_prompt_ids check stops such a run."),
    ),
    "gemma2": FamilySpec(
        model_type="gemma2",
        template_kwargs={},
        reasoning="none",
        reasoning_stored=False,
        system_prompt="fold_into_user",
        span_builder="generic",
        response_cleanup="none",
        bos_in_template=True,
        end_token="<end_of_turn>",
        expected_header_tokens=None,
        hf_load_kwargs={"attn_implementation": "eager"},
        notes=("google/gemma-2-27b-it; gated; ~54 GB.  The template raises on a system role; "
               "format_conversation folds the instruction into the user turn.  Its three-token "
               "header is accepted with one wrong token, like Qwen3's seven and Llama 3's four "
               "(spans.allowed_header_mismatches; Roger, 2026-10-13)."),
    ),
    "olmo2": FamilySpec(
        model_type="olmo2",
        template_kwargs={},
        reasoning="none",
        reasoning_stored=False,
        system_prompt="native",
        span_builder="generic",
        response_cleanup="none",
        bos_in_template=True,
        end_token="<|endoftext|>",
        expected_header_tokens=None,
        notes=("allenai/OLMo-2-0325-32B-Instruct (or 7B/13B); which OLMo is open (§ 15 Q6).  "
               "Its two-token header must match exactly (spans.allowed_header_mismatches; the "
               "search is anchored on the opening tag, so one wrong token of two would accept "
               "anything; Roger, 2026-10-13)."),
    ),
    "olmo3": FamilySpec(
        model_type="olmo3",
        template_kwargs={},
        reasoning="none",
        reasoning_stored=False,
        system_prompt="native",
        span_builder="generic",
        response_cleanup="none",
        bos_in_template=False,
        end_token="",
        expected_header_tokens=None,
        notes=("allenai/Olmo-3-*-Instruct; the Think variants reason in <think> blocks with "
               "no documented switch (a separate row, always_on, if ever used).  Every cell "
               "to check; recommended: the Instruct variant (§ 15 Q6)."),
    ),
    "gpt_oss": FamilySpec(
        model_type="gpt_oss",
        template_kwargs={"reasoning_effort": "low"},
        reasoning="always_on",
        reasoning_stored=True,
        system_prompt="native",
        span_builder="generic",
        response_cleanup="harmony_final_channel",
        bos_in_template=False,
        end_token="<|return|>",
        expected_header_tokens=None,
        notes=("openai/gpt-oss-20b, -120b.  TODO, not built (§ 15 Q3: 'Leave this as a TODO "
               "until we know if we're actually using gpt-oss').  Roger's intended layout: "
               "slot 0 from the response text tokens, slot 1 from the thinking text tokens, "
               "and a slot for every fixed token before or between them.  Steps 1 and 2 "
               "refuse this row until the spike (plan § 10 task 9) is done."),
    ),
}

# The header each row is expected to derive, from the plan's table (tokens,
# and ids where they are known), for the real-tokenizer test to assert before
# a smoke run.  Not asserted by the steps: only expected_header_tokens is.
DRAFT_HEADERS: Dict[str, Tuple[Tuple[str, ...], Optional[Tuple[int, ...]]]] = {
    "qwen3": (QWEN3_NON_THINKING_HEADER_TOKENS, QWEN3_NON_THINKING_HEADER_IDS),
    "llama": (("<|start_header_id|>", "assistant", "<|end_header_id|>", "\n\n"),
              (128006, 78191, 128007, 271)),
    "gemma2": (("<start_of_turn>", "model", "\n"), (106, 2516, 108)),
    "olmo2": (("<|assistant|>", "\n"), None),
    "gpt_oss": (("<|end|>", "<|start|>", "assistant", "<|channel|>", "final", "<|message|>"), None),
}

# The cells of each row not yet checked against the real tokenizer or model,
# and how each is checked (plan § 5.2 "to check"; § 9 test 11 and § 11).
TO_CHECK: Dict[str, Tuple[str, ...]] = {
    "qwen3": (),
    "llama": (
        "header: derive on the real tokenizer; expect DRAFT_HEADERS['llama'] (4 tokens)",
        "system_prompt native: the probe keeps the system message (the template also "
        "injects its date block)",
        "template_kwargs: the date pin's kwarg name (date_string?) in the template text; TODO "
        "until Llama is confirmed",
    ),
    "gemma2": (
        "header: derive on the real tokenizer; expect DRAFT_HEADERS['gemma2'] (3 tokens)",
        "hf_load_kwargs eager attention: memory at batch 8, length 2048 (smoke run)",
    ),
    "olmo2": (
        "header: derive; expect DRAFT_HEADERS['olmo2'] (2 tokens, from the Tulu 3 template "
        "as recalled)",
        "system_prompt native (<|system|>): the probe",
        "bos_in_template: whether the template starts with {{ bos_token }}",
        "end_token <|endoftext|>",
    ),
    "olmo3": (
        "every cell: template_kwargs, reasoning, header, system prompt, BOS, end token, "
        "clean-up; derive at the smoke run",
    ),
    "gpt_oss": (
        "not built (§ 15 Q3): template_kwargs reasoning_effort and the date pin, the header "
        "with and without the stored thinking, the developer block, the end tokens, how vLLM "
        "returns the channels; the spike (plan § 10 task 9)",
    ),
}

# Clean-ups whose code exists (assistant_axis/generation.py).
BUILT_RESPONSE_CLEANUPS = ("none", "strip_leaked_think")


class UnknownFamilyError(ValueError):
    """A ``model_type`` with no row in :data:`FAMILIES`."""


class FamilyNotBuiltError(RuntimeError):
    """A row whose needs are not built yet (the gpt-oss TODO)."""


def family_for_model_type(model_type: str) -> FamilySpec:
    """The row for ``model_type``; :class:`UnknownFamilyError` naming the
    known keys and where to add a row when there is none."""
    try:
        return FAMILIES[model_type]
    except KeyError:
        raise UnknownFamilyError(
            f"model_type {model_type!r} has no row in the family table; known: "
            f"{', '.join(sorted(FAMILIES))}.  Add a FamilySpec row to FAMILIES in "
            f"assistant_axis/model_families.py (how: that module's docstring) before "
            f"running steps 1 or 2 on this model.") from None


def model_type_of(name_or_path: str) -> str:
    """``model_type`` from the model's ``config.json`` (``AutoConfig``; a few
    KB, fetched from where the tokenizer comes from)."""
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(name_or_path).model_type


def family_for_model(name_or_path: str) -> FamilySpec:
    """The row for a Hugging Face model id or local path, by its
    ``config.json``'s ``model_type``."""
    return family_for_model_type(model_type_of(name_or_path))


def unbuilt_reason(family: FamilySpec) -> Optional[str]:
    """Why steps 1 and 2 must not run this row yet, or None.  A row whose
    clean-up has no code, or whose reasoning is stored (nothing writes it, and
    Roger's intended slot layout for it is a TODO: plan § 15 Q3)."""
    reasons = []
    if family.response_cleanup not in BUILT_RESPONSE_CLEANUPS:
        reasons.append(f"its response clean-up {family.response_cleanup!r} is not built")
    if family.reasoning_stored:
        reasons.append("its reasoning is stored and re-rendered, which nothing builds yet")
    if not reasons:
        return None
    return (f"the {family.model_type!r} row is a TODO: {'; '.join(reasons)} "
            f"(reports/pipeline/coding_plan_multi_model_extraction.md § 15 Q3, § 10 task 9)")


def require_built(family: FamilySpec) -> FamilySpec:
    """``family``, or :class:`FamilyNotBuiltError` when :func:`unbuilt_reason`
    names a reason."""
    reason = unbuilt_reason(family)
    if reason:
        raise FamilyNotBuiltError(reason)
    return family
