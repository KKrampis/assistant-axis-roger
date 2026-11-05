"""Transport choice for ``judge_tier_cost_eval.py`` (OpenAI Flex processing, the providers' batch
APIs, real time), over the one request layer, ``assistant_axis/judge.py``.

A model's requests, its Flex and retry behaviour, what a reply means and what a refusal or an empty
account does are all ``judge.py``'s, the code pipeline step 3 and the axis judge use; this module
re-exports them under the names the tool uses and adds what only the tool needs: which transport a
model goes by, and the two helpers its batch paths need.  Until the port of 2026-11-02 this was an
adapted copy, on a branch whose ``judge.py`` was two months old.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from assistant_axis import judge as _judge
from assistant_axis.judge import (  # noqa: F401 -- re-exported
    DEFAULT_TEMPERATURE, FAILED, FLEX_FALLBACK_AFTER_S, FLEX_MAX_RETRIES, FLEX_MIN_ATTEMPT_S,
    FLEX_RETRY_SLEEP_S, FLEX_TIMEOUT_S, GEMINI_ATTEMPTS, GEMINI_BASE_URL, GEMINI_FLEX_TIER, GEMINI_RETRY_SLEEP_S,
    GeminiCache, gemini_create_cache, gemini_delete_cache, gemini_served_tier,
    GEMINI_RETRY_STATUSES, JUDGE_REQUEST_SETTINGS, MODEL_CAPABILITIES, SERVICE_TIERS, SKIPPED,
    BillingError, CallStats, FlexRefusedError, JudgeRequestError, RequestOverride, Truncated,
    accepts_flex, accepts_temperature, anthropic_reply, anthropic_request_params, chat_create_kwargs,
    chat_request_body, content_flag, flex_attempt_client, gemini_quota_ids, gemini_reply,
    gemini_request_body, is_billing_error, is_flex_refusal, judge_request_settings,
    make_gemini_client, model_capabilities, parse_request_setting, prompt_parts, provider_for_model,
    request_temperature, usage_key,
)
from assistant_axis.judge import call_anthropic_judge as call_anthropic  # noqa: F401
from assistant_axis.judge import call_gemini_judge as call_gemini  # noqa: F401
from assistant_axis.judge import call_judge_model as call_model  # noqa: F401
from assistant_axis.judge import call_openai_judge as call_openai  # noqa: F401
from assistant_axis.judge_pricing import BATCH_SUFFIX, FLEX_SUFFIX, openai_usage

#: The finish reason of a reply cut at ``max_tokens``: OpenAI's ``finish_reason``, and what
#: ``judge.anthropic_reply`` makes of Anthropic's ``stop_reason == "max_tokens"``.
CUT_AT_CAP = "length"


def request_for(model: str, prompt: str, max_tokens: int, settings: Dict[str, Any],
                service_tier: Optional[str] = None, cached_content: Optional[str] = None) -> Dict[str, Any]:
    """The provider's request for ``model`` (a chat body, Messages params or a generateContent
    body): the same request a live call sends, which a Batch API line carries unchanged.
    ``cached_content``: a Gemini request on an explicit cache of the prompt's prefix (its
    suffix only; the Gemini batch runner puts the cache's name in at submission)."""
    provider = provider_for_model(model)
    if provider == "anthropic":
        return anthropic_request_params(prompt, model, max_tokens, settings=settings)
    if provider == "google":
        return gemini_request_body(prompt, model, max_tokens, settings=settings, cached_content=cached_content)
    return chat_request_body(prompt, model, max_tokens, settings=settings, service_tier=service_tier)


def prompt_of_request(request: Dict[str, Any]) -> str:
    """The prompt of a request built by :func:`request_for`."""
    if "contents" in request:
        return "".join(part.get("text", "") for c in request["contents"] for part in c.get("parts", []))
    content = request["messages"][0]["content"]
    return content if isinstance(content, str) else "".join(part.get("text", "") for part in content)


def openai_reply(response: Any) -> Tuple[Optional[str], Optional[str], int, int]:
    """``(text, finish_reason, prompt_tokens, completion_tokens)`` of a chat completion, an SDK
    object or a plain dict (a Batch API result body)."""
    def g(obj: Any, name: str) -> Any:
        return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
    choices = g(response, "choices") or []
    first = choices[0] if choices else None
    message = g(first, "message") if first is not None else None
    text = g(message, "content") if message is not None else None
    u = openai_usage(response)
    return (text or None, g(first, "finish_reason") if first is not None else None,
            u.prompt_tokens, u.completion_tokens)


TRANSPORT_CHOICES = ("auto", "flex", "batch", "realtime")


@dataclass(frozen=True)
class TransportChoice:
    transport: str      # "flex", "batch" or "realtime"
    probe: bool         # Flex untested: the first request probes it, batch if refused
    reason: str

    def describe(self) -> str:
        return f"{self.transport}{' (probe)' if self.probe else ''}: {self.reason}"


def choose_transport(model: str, requested: str = "auto") -> TransportChoice:
    """The transport for ``model`` under ``--transport requested``.

    ``auto``: an Anthropic model goes by Message Batches; a Gemini model by its Batch API (half price,
    ``gemini_batch.py``; since 2026-11-03, once a round of 1,769 requests came back in about 3 minutes);
    an OpenAI model by Flex when ``judge.MODEL_CAPABILITIES`` says OpenAI takes it, by the
    Batch API when it says OpenAI refuses it, and, untested, by Flex with a probe (the first request
    alone; a service-tier HTTP 400 switches the model to batch for the run).  Gemini's Flex tier on
    generateContent (``"serviceTier": "flex"``, half price, measured live on 2026-11-05) is
    ``--transport flex``, never auto's choice: its latency target is 1 to 15 minutes, its batch the
    same price.  A transport the model cannot take raises :class:`JudgeRequestError`."""
    if requested not in TRANSPORT_CHOICES:
        raise ValueError(f"--transport {requested!r}: expected one of {TRANSPORT_CHOICES}")
    provider = provider_for_model(model)
    if provider == "google":
        if requested == "flex":
            return TransportChoice("flex", False, "--transport flex: Gemini generateContent on the Flex tier "
                                                  "(serviceTier flex, half price)")
        if requested == "realtime":
            return TransportChoice("realtime", False, "--transport realtime: Gemini generateContent")
        return TransportChoice("batch", False, "--transport batch: Gemini Batch API" if requested == "batch" else
                               "Gemini model: Batch API at half price; --transport realtime for real time")
    if requested == "realtime":
        return TransportChoice("realtime", False, "--transport realtime")
    if requested == "batch":
        return TransportChoice("batch", False, "--transport batch")
    flex = accepts_flex(model) if provider == "openai" else False
    if requested == "flex":
        if provider != "openai":
            raise JudgeRequestError(f"--transport flex: Flex processing is OpenAI's, not available for {model}")
        if flex is False:
            raise JudgeRequestError(f"--transport flex: OpenAI refuses Flex processing for {model} "
                                    f"(MODEL_CAPABILITIES); use --transport batch")
        return TransportChoice("flex", False, "--transport flex" + (
            "" if flex else " (Flex untested for this model: a refusal stops the model)"))
    if provider == "anthropic":
        return TransportChoice("batch", False, "Anthropic model: Message Batches")
    if flex is True:
        return TransportChoice("flex", False, "MODEL_CAPABILITIES: OpenAI takes Flex for this model")
    if flex is False:
        return TransportChoice("batch", False, "MODEL_CAPABILITIES: OpenAI refuses Flex for this model")
    return TransportChoice("flex", True, "Flex untested for this model: the first request probes it, "
                                         "batch if OpenAI refuses the tier")


def price_key(model: str, transport: str) -> str:
    """The usage key a transport's calls are charged under (and priced by)."""
    if transport == "batch":
        return f"{model}{BATCH_SUFFIX}"
    if transport == "flex":
        return f"{model}{FLEX_SUFFIX}"
    return model


__all__: List[str] = [name for name in dir() if not name.startswith("_")]
