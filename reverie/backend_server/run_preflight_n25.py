#!/usr/bin/env python3
"""
Headless n=25 pre-flight: fork, burn-in, survey, background truth, analysis.

Forks base_the_ville_n25 -> preflight_the_ville_n25-1, advances 1800 steps
with a headless frontend emulator, runs survey pre, writes all three ground
truth layers, and runs analyze_survey.py.

Usage (from reverie/backend_server/):
  python run_preflight_n25.py
"""
import json
import os
import subprocess
import sys
import threading
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

FORK_FROM = "base_the_ville_n25"
SIM_CODE = "preflight_the_ville_n25-1"
BURN_IN_STEPS = 1800
FS_STORAGE = "../../environment/frontend_server/storage"


class HeadlessFrontend:
    """Background thread: movement/{step}.json -> environment/{step+1}.json."""

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


def _run_script(script_name, *args):
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, script_name), *args]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)


def main():
    from reverie import ReverieServer
    from perception_survey import run_perception_survey

    sim_folder = os.path.abspath(f"{FS_STORAGE}/{SIM_CODE}")
    survey_dir = os.path.join(sim_folder, "survey")

    if os.path.exists(sim_folder):
        meta_path = os.path.join(sim_folder, "reverie", "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            step = meta.get("step", 0)
            print(f"Resuming existing sim {SIM_CODE} at step {step}")
            rs = ReverieServer(SIM_CODE, SIM_CODE)
        else:
            print(f"ERROR: {sim_folder} exists but has no reverie/meta.json")
            sys.exit(1)
    else:
        print(f"Forking: {FORK_FROM} -> {SIM_CODE}")
        rs = ReverieServer(FORK_FROM, SIM_CODE)
        print(f"  Loaded: step={rs.step}, n_personas={len(rs.personas)}")

    frontend = HeadlessFrontend(sim_folder)
    frontend.start()
    print("  Headless frontend started.")

    try:
        if rs.step < BURN_IN_STEPS:
            remaining = BURN_IN_STEPS - rs.step
            print(f"\n{'=' * 60}")
            print(f"  BURN-IN: {remaining} steps ({rs.step} -> {BURN_IN_STEPS})")
            print(f"{'=' * 60}")
            rs.start_server(remaining)
            print(f"  -> Now at step {rs.step}")
            try:
                rs.save()
                print("  -> Saved burn-in state")
            except AttributeError as exc:
                if "strftime" in str(exc):
                    print(f"  WARNING: skipped rs.save() after burn-in ({exc})")
                else:
                    raise

        pre_csv = os.path.join(survey_dir, "perception_survey_pre.csv")
        if not os.path.isfile(pre_csv):
            print(f"\n{'=' * 60}")
            print(f"  SURVEY: pre (step {rs.step})")
            print(f"{'=' * 60}")
            path = run_perception_survey(
                rs.personas, rs.sim_code, rs.step, rs.curr_time,
                survey_dir, wave_id="pre")
            print(f"  -> {os.path.basename(path)}")
            try:
                rs.save()
            except AttributeError as exc:
                if "strftime" in str(exc):
                    print(f"  WARNING: skipped rs.save() after survey ({exc})")
                else:
                    raise
        else:
            print(f"\n  Survey already exists: {pre_csv}")

        bg_path = os.path.join(survey_dir, "ground_truth", "background_social_edges.csv")
        if not os.path.isfile(bg_path):
            print(f"\n{'=' * 60}")
            print("  BACKGROUND TRUTH: background_social_edges.csv")
            print(f"{'=' * 60}")
            _run_script("bootstrap_n25_ground_truth_layers.py", SIM_CODE)
        else:
            print(f"\n  Background truth already exists: {bg_path}")

        metrics_path = os.path.join(
            survey_dir, "analysis_micro_tie_metrics_by_truth.csv")
        if not os.path.isfile(metrics_path):
            print(f"\n{'=' * 60}")
            print("  ANALYSIS: analyze_survey.py")
            print(f"{'=' * 60}")
            _run_script("analyze_survey.py", survey_dir)
        else:
            print(f"\n  Analysis already exists: {metrics_path}")

    finally:
        frontend.stop()

    print(f"\n{'=' * 60}")
    print("  N=25 PRE-FLIGHT COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Sim: {SIM_CODE}")
    print(f"  Final step: {rs.step}")
    print(f"  Survey dir: {survey_dir}")
    print("\n  Verify:")
    print(f"    python verify_preflight_n25.py {survey_dir}")


if __name__ == "__main__":
    main()
