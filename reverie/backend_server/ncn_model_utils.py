#!/usr/bin/env python3
"""Model consistency checks for the Network Cognition Experiment."""
import os
import sys

# Model recorded from preflight evidence (.env + llm_calls.jsonl).
PREFLIGHT_MODEL = "gpt-4o-mini"


def verify_model_lock():
    """Abort if OPENAI_MODEL does not match the preflight model."""
    from persona.prompt_template.gpt_structure import DEFAULT_CHAT_MODEL

    if DEFAULT_CHAT_MODEL != PREFLIGHT_MODEL:
        print(
            f"ERROR: OPENAI_MODEL={DEFAULT_CHAT_MODEL!r} does not match "
            f"preflight model {PREFLIGHT_MODEL!r}.\n"
            "Set OPENAI_MODEL in .env to match preflight before running."
        )
        sys.exit(1)
    print(f"  Model lock OK: {DEFAULT_CHAT_MODEL}")
    return DEFAULT_CHAT_MODEL


def load_dotenv():
    """Load project-root .env if present (does not override existing vars)."""
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    env_path = os.path.join(root, ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)
