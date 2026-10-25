"""Pilot-only API contract and usage accounting; legacy runs are unchanged."""
import json
import os
from datetime import datetime
from pathlib import Path


class PilotAPIError(BaseException):
    """Abort a pilot instead of letting legacy broad catches fabricate defaults."""


CONTEXT = {"phase": "preflight", "sim_time": None, "step": None}
_FAILURE = None


def raise_if_failed():
    if os.environ.get("NCN_PILOT") == "1" and _FAILURE is not None:
        raise PilotAPIError(f"Pilot API previously failed: {_FAILURE}")


def set_context(phase, sim_time, step, **extra):
    CONTEXT.clear()
    CONTEXT.update(phase=phase, sim_time=sim_time.isoformat() if sim_time else None,
                   step=step, **extra)


def chat_arguments(model, prompt, max_tokens=None):
    if model != "gpt-6-luna":
        raise PilotAPIError("Pilot model lock rejected a non-Luna request")
    # Deliberately use a single independent user turn; never reuse survey history.
    return {"model": model, "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": "none", "max_completion_tokens": max_tokens or 2048}


def record_usage(kind, model, response=None, error=None):
    if os.environ.get("NCN_PILOT") != "1":
        return
    global _FAILURE
    if error is not None:
        _FAILURE = type(error).__name__
    path = Path(os.environ["NCN_USAGE_LOG"])
    path.parent.mkdir(parents=True, exist_ok=True)
    usage = getattr(response, "usage", None)
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif usage is not None:
        usage = dict(usage)
    chain, seen = [], set()
    cause = error
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        chain.append(type(cause).__name__)
        cause = cause.__cause__ or cause.__context__
    record = {"wall_time": datetime.now().isoformat(), **CONTEXT,
              "kind": kind, "requested_model": model,
              "response_model": getattr(response, "model", None),
              "request_id": getattr(response, "_request_id", None),
              "usage": usage, "status": "error" if error else "ok",
              "error_type": type(error).__name__ if error else None,
              "error_chain_types": chain}
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(record, ensure_ascii=False) + "\n")


def estimated_cost(record):
    """Standard public USD estimate, not an invoice; raw usage is retained."""
    usage = record.get("usage")
    if not usage:
        return None
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    output = usage.get("completion_tokens", usage.get("output_tokens", 0))
    if record["kind"] == "embed":
        if record["requested_model"] != "text-embedding-3-small":
            return None
        return prompt * 0.02 / 1_000_000
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens", 0) or 0
    writes = details.get("cache_write_tokens", 0) or 0
    return (max(0, prompt-cached-writes)*0.10 + cached*0.01 + writes*0.125 + output*0.50)/1_000_000
