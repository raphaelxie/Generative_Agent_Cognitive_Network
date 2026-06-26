#!/usr/bin/env python3
"""
Calibrate the community_group NCN instrument on preflight personas.

Runs only the community_group prompt (~25 LLM calls) and reports fail-safe
rate and group-count distribution.

Usage (from reverie/backend_server/):
  python calibrate_community_instrument.py
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from ncn_model_utils import load_dotenv, verify_model_lock
from perception_survey import _retrieve_with_diagnostics, _snapshot_last_accessed, _restore_last_accessed
from persona.prompt_template.run_gpt_prompt import run_gpt_prompt_ncn_community_group
from reverie import ReverieServer

PREFLIGHT_SIM = "preflight_the_ville_n25-1"
TARGET_FAIL_SAFE_RATE = 1 / 25


def main():
    load_dotenv()
    verify_model_lock()

    print(f"Loading {PREFLIGHT_SIM} ...")
    rs = ReverieServer(PREFLIGHT_SIM, PREFLIGHT_SIM)
    personas = rs.personas
    persona_names = sorted(personas.keys())
    n = len(persona_names)

    snap = _snapshot_last_accessed(personas)
    fail_safe = 0
    group_counts = []

    try:
        focal = "social connections and interactions in the community"
        for i, name in enumerate(persona_names, start=1):
            persona = personas[name]
            statements, _ = _retrieve_with_diagnostics(
                persona, focal, query_target_person=None)
            groups, meta = run_gpt_prompt_ncn_community_group(
                persona, statements, persona_names)
            is_fs = meta[0] is None
            if is_fs:
                fail_safe += 1
            group_counts.append(len(groups))
            print(f"  [{i:2d}/{n}] {name}: "
                  f"{len(groups)} groups"
                  f"{'  FAIL-SAFE' if is_fs else ''}")
    finally:
        _restore_last_accessed(personas, snap)

    rate = fail_safe / n
    print(f"\nFail-safe: {fail_safe}/{n} ({rate:.1%})")
    print(f"Group counts: min={min(group_counts)}, "
          f"max={max(group_counts)}, "
          f"mean={sum(group_counts)/len(group_counts):.1f}")

    if rate > TARGET_FAIL_SAFE_RATE:
        print(f"\nWARNING: fail-safe rate exceeds target "
              f"({TARGET_FAIL_SAFE_RATE:.1%}). "
              "Iterate on prompt/parsing before formal run.")
        sys.exit(1)

    print("\nCalibration PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
