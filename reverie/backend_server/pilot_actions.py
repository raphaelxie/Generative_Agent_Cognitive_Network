"""Pilot action-path contracts (v4): strict JSON for location, object, event and
conversation decisions.

Why this exists: the original location/triple/decision prompts are completion
style. They end in an open ``{`` or ``(Name,`` and the legacy validators insist on
a closing brace or an exact trailing phrase. A chat model answers ``Hobbs Cafe``
instead, every retry is rejected, and the legacy fail-safes then silently send the
agent home (sector), keep it in its room (arena), record ``(name, name, 'is')``
(triple) or answer ``yes`` (decide_to_talk). In the first pilot trajectory this
kept all 25 agents at home, so only couples sharing a bedroom ever met.

Pilot rule, same as pilot_tasks: a model answer that cannot be validated after
three attempts raises PilotScheduleError. It is never replaced by a default.
"""
import json
import re

from pilot_tasks import request_contract

MAX_FIELD = 80
MAX_REASONING = 600


def load_json_object(raw):
    """Parse one JSON object; tolerate a single Markdown code fence only."""
    text = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _fold(text):
    text = text.strip().strip("{}[]\"'`").strip().rstrip(".")
    return " ".join(text.split()).casefold()


def match_option(value, options):
    """Map a model answer to the exact option string, or reject it."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Answer must be a nonempty string")
    folded = _fold(value)
    hits = [o for o in options if _fold(o) == folded]
    if len(hits) != 1:
        raise ValueError(f"Answer {value.strip()[:80]!r} is not exactly one of the options")
    return hits[0]


def parse_choice(raw, options, field):
    value = load_json_object(raw)
    if set(value) != {field}:
        raise ValueError(f"Expected exactly the field {field!r}")
    return match_option(value[field], options)


def short_text(value, name):
    if (not isinstance(value, str) or not 1 <= len(value.strip()) <= MAX_FIELD
            or "\n" in value or "\r" in value or not any(c.isalnum() for c in value)):
        raise ValueError(f"{name} must be a short single-line phrase")
    return value.strip()


def parse_triple(raw):
    value = load_json_object(raw)
    if set(value) != {"predicate", "object"}:
        raise ValueError("Expected exactly predicate and object")
    return short_text(value["predicate"], "predicate"), short_text(value["object"], "object")


def _reasoning_ok(value):
    if "reasoning" in value:
        text = value.pop("reasoning")
        if not isinstance(text, str) or len(text) > MAX_REASONING:
            raise ValueError("reasoning must be a short string")


def parse_yes_no(raw):
    value = load_json_object(raw)
    _reasoning_ok(value)
    if set(value) != {"answer"} or not isinstance(value["answer"], str):
        raise ValueError('Expected "answer" with the value "yes" or "no"')
    answer = value["answer"].strip().lower()
    if answer not in ("yes", "no"):
        raise ValueError('"answer" must be exactly "yes" or "no"')
    return answer


def parse_option_number(raw, allowed):
    value = load_json_object(raw)
    _reasoning_ok(value)
    if set(value) != {"option"}:
        raise ValueError('Expected "option"')
    option = value["option"]
    if isinstance(option, str) and option.strip().isdigit():
        option = int(option.strip())
    if type(option) is not int or option not in allowed:
        raise ValueError(f"option must be one of {sorted(allowed)}")
    return str(option)


# ---- prompt construction -------------------------------------------------
# Each builder keeps the original template text (instructions and few-shot
# examples) and only replaces its completion stub with a JSON instruction.

def _strip_stub(prompt, marker):
    """Cut the trailing completion stub that starts at the last ``marker``."""
    head, sep, _ = prompt.rpartition(marker)
    if not sep:
        raise ValueError(f"Template stub {marker!r} not found")
    return head.rstrip()


def _options_text(options):
    return json.dumps(list(options), ensure_ascii=False)


def choose_from(prompt, options, field, label, stub=None):
    """Ask for one of ``options`` as {field: option}; return the exact option."""
    options = list(options)
    if not options:
        raise ValueError("No options to choose from")
    if len(options) == 1:
        return options[0]  # nothing to decide; not a model failure
    body = _strip_stub(prompt, stub) if stub else prompt.rstrip()
    body += (f'\n\nReturn ONLY a JSON object {{"{field}": "<one option>"}}. The value must be '
             f"exactly one of these options, spelled verbatim: {_options_text(options)}. "
             "No Markdown and no explanation.")
    return request_contract(body, lambda raw: parse_choice(raw, options, field), label)


def choose_sector(prompt, options):
    return choose_from(prompt, options, "area", "Action sector", stub="{")


def choose_arena(prompt, options):
    return choose_from(prompt, options, "area", "Action arena", stub="{")


def choose_object(prompt, options):
    return choose_from(prompt, options, "object", "Action object")


def generate_triple(prompt, label="Event triple"):
    body = _strip_stub(prompt, "Output:")
    body += ('\n\nAnswer only for the final Input above. Return ONLY a JSON object '
             '{"predicate": "<short verb>", "object": "<short object>"}, each value a short '
             "single-line phrase. No Markdown and no explanation.")
    return request_contract(body, parse_triple, label)


def decide_yes_no(prompt, label="Decide to talk"):
    body = _strip_stub(prompt, "Reasoning:")
    body += ('\n\nReturn ONLY a JSON object {"reasoning": "<one or two sentences>", '
             '"answer": "yes"} or {"reasoning": "...", "answer": "no"}. No Markdown.')
    return request_contract(body, parse_yes_no, label)


def decide_option(prompt, allowed=(1, 2), label="Decide to react"):
    body = _strip_stub(prompt, "Reasoning:")
    body += ('\n\nReturn ONLY a JSON object {"reasoning": "<one or two sentences>", '
             f'"option": <one of {sorted(allowed)}>}}. No Markdown.')
    return request_contract(body, lambda raw: parse_option_number(raw, set(allowed)), label)
