#!/usr/bin/env python3
"""
Run n=15 pre-flight measurement batch after burn-in:
  survey pre + shock isolate-hub + unshock + shock isolate-broker + unshock

Usage (from this directory, after burn-in and optional `save` in reverie.py):
  python preflight_n15_batch.py

Requires the simulation folder:
  environment/frontend_server/storage/preflight_the_ville_n15-1
with reverie/meta.json step/time matching the state you want to survey.
"""
import os
import sys

_backend_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_backend_dir)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from reverie import ReverieServer  # noqa: E402


SIM_CODE = "preflight_the_ville_n15-1"


def main():
    print(f"Loading ReverieServer({SIM_CODE!r}, {SIM_CODE!r}) ...")
    rs = ReverieServer(SIM_CODE, SIM_CODE)
    print("Running preflight measurement batch (survey pre + hub/broker shocks)...")
    rs.run_preflight_measurement_batch(wave_id="pre")
    print("Done. Save the sim from reverie.py with `fin` if you still have a session open,")
    print("or start reverie.py and use `save` to persist persona memory + meta.")


if __name__ == "__main__":
    main()
