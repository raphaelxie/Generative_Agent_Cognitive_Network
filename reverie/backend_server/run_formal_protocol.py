#!/usr/bin/env python3
"""
Phase B: Formal pre/post shock protocol — headless execution.

Forks from prepost_n15_calibration-1 to formal_n15_full_instruments-1,
runs the full protocol:
  1. Pre-shock survey at step 1800
  2. Advance 600 steps (to 2400)
  3. Post-advance survey at t2400
  4. Apply hub shock
  5. Advance 600 steps (to 3000)
  6. Post-shock survey at t3000_post_hub
  7. Un-shock
  8. Save

Provides the frontend file exchange via a background thread so the
simulation can advance without the browser.

Usage (from reverie/backend_server/):
  python run_formal_protocol.py
"""
import json
import os
import sys
import threading
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

FORK_FROM = "prepost_n15_calibration-1"
SIM_CODE = "formal_n15_full_instruments-1"
FS_STORAGE = "../../environment/frontend_server/storage"

# ── Headless frontend emulator ─────────────────────────────────────────

class HeadlessFrontend:
    """Background thread that reads movement files and writes env files.

    When the backend writes movement/{step}.json, this thread reads the
    persona movements and writes environment/{step+1}.json with the new
    positions, mimicking what the browser frontend does.
    """

    def __init__(self, sim_folder):
        self.sim_folder = sim_folder
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self):
        last_step_processed = -1
        while not self._stop.is_set():
            move_dir = os.path.join(self.sim_folder, "movement")
            env_dir = os.path.join(self.sim_folder, "environment")

            if not os.path.isdir(move_dir):
                time.sleep(0.05)
                continue

            move_files = []
            for fn in os.listdir(move_dir):
                if fn.endswith(".json"):
                    try:
                        step = int(fn.replace(".json", ""))
                        move_files.append(step)
                    except ValueError:
                        pass

            for step in sorted(move_files):
                if step <= last_step_processed:
                    continue
                next_env = os.path.join(env_dir, f"{step + 1}.json")
                if os.path.exists(next_env):
                    last_step_processed = step
                    continue

                move_file = os.path.join(move_dir, f"{step}.json")
                try:
                    with open(move_file) as f:
                        data = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    continue

                persona_data = data.get("persona", {})
                env_out = {}
                for name, info in persona_data.items():
                    mv = info.get("movement", [0, 0])
                    env_out[name] = {"x": mv[0], "y": mv[1]}

                if not env_out:
                    continue

                os.makedirs(env_dir, exist_ok=True)
                with open(next_env, "w") as f:
                    json.dump(env_out, f, indent=2)

                last_step_processed = step

            time.sleep(0.02)


# ── Main protocol ──────────────────────────────────────────────────────

def main():
    from reverie import ReverieServer
    from perception_survey import run_perception_survey

    sim_folder = os.path.abspath(f"{FS_STORAGE}/{SIM_CODE}")

    if os.path.exists(sim_folder):
        print(f"WARNING: {SIM_CODE} already exists. Remove it first to re-run.")
        print(f"  rm -rf {sim_folder}")
        sys.exit(1)

    print(f"Forking: {FORK_FROM} -> {SIM_CODE}")
    rs = ReverieServer(FORK_FROM, SIM_CODE)
    print(f"  Loaded: step={rs.step}, n_personas={len(rs.personas)}")

    survey_dir = os.path.join(sim_folder, "survey")

    frontend = HeadlessFrontend(sim_folder)
    frontend.start()
    print("  Headless frontend started.")

    try:
        # Step 1: Pre-shock survey at step 1800
        print(f"\n{'='*60}")
        print(f"  STEP 1: Pre-shock survey (step {rs.step})")
        print(f"{'='*60}")
        path = run_perception_survey(
            rs.personas, rs.sim_code, rs.step, rs.curr_time,
            survey_dir, wave_id="1800")
        print(f"  -> {os.path.basename(path)}")

        # Step 2: Advance 600 steps
        print(f"\n{'='*60}")
        print(f"  STEP 2: Advance 600 steps ({rs.step} -> {rs.step + 600})")
        print(f"{'='*60}")
        rs.start_server(600)
        print(f"  -> Now at step {rs.step}")

        # Step 3: Post-advance survey
        print(f"\n{'='*60}")
        print(f"  STEP 3: Post-advance survey (step {rs.step})")
        print(f"{'='*60}")
        path = run_perception_survey(
            rs.personas, rs.sim_code, rs.step, rs.curr_time,
            survey_dir, wave_id="t2400")
        print(f"  -> {os.path.basename(path)}")

        # Step 4: Apply hub shock
        print(f"\n{'='*60}")
        print(f"  STEP 4: Apply hub shock")
        print(f"{'='*60}")
        msg = rs._try_shock_isolate_command("shock isolate-hub", sim_folder)
        print(f"  -> {msg}")

        # Step 5: Advance 600 more steps
        print(f"\n{'='*60}")
        print(f"  STEP 5: Advance 600 steps ({rs.step} -> {rs.step + 600})")
        print(f"{'='*60}")
        rs.start_server(600)
        print(f"  -> Now at step {rs.step}")

        # Step 6: Post-shock survey
        print(f"\n{'='*60}")
        print(f"  STEP 6: Post-shock survey (step {rs.step})")
        print(f"{'='*60}")
        path = run_perception_survey(
            rs.personas, rs.sim_code, rs.step, rs.curr_time,
            survey_dir, wave_id="t3000_post_hub")
        print(f"  -> {os.path.basename(path)}")

        # Step 7: Un-shock
        print(f"\n{'='*60}")
        print(f"  STEP 7: Unshock")
        print(f"{'='*60}")
        msg = rs._try_unshock_command(sim_folder)
        print(f"  -> {msg}")

        # Step 8: Save
        print(f"\n{'='*60}")
        print(f"  STEP 8: Save")
        print(f"{'='*60}")
        rs.save()
        print(f"  -> Saved to {sim_folder}")

    finally:
        frontend.stop()

    print(f"\n{'='*60}")
    print(f"  FORMAL PROTOCOL COMPLETE")
    print(f"{'='*60}")
    print(f"  Sim: {SIM_CODE}")
    print(f"  Final step: {rs.step}")
    print(f"  Survey dir: {survey_dir}")
    print(f"\n  Next: run the analysis pipeline:")
    print(f"    python analyze_survey.py {survey_dir}")
    print(f"    python survey_network_summary.py {survey_dir}")
    print(f"    python shock_prepost_audit.py {survey_dir}")
    print(f"    python figures/generate_figures.py {survey_dir}")


if __name__ == "__main__":
    main()
