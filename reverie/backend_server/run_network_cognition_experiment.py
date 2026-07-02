#!/usr/bin/env python3
"""
Network Cognition Experiment — between-branch difference-in-differences.

Forks preflight_the_ville_n25-1 into a common baseline, runs a pre-shock
survey, then forks into control / hub / broker branches. Each branch
advances post-shock and runs a post survey.

Usage (from reverie/backend_server/):
  PILOT=1 caffeinate -i python run_network_cognition_experiment.py
  caffeinate -i python run_network_cognition_experiment.py

Long-horizon extension (resume existing branches, advance to step 4200,
add a `post_long` survey wave, and build a separate long-horizon DiD):
  EXTEND=1 caffeinate -i python run_network_cognition_experiment.py
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

from ncn_model_utils import load_dotenv, verify_model_lock, PREFLIGHT_MODEL

FS_STORAGE = "../../environment/frontend_server/storage"
PILOT = os.environ.get("PILOT", "0") == "1"
SIM_SUFFIX = "_pilot" if PILOT else "_full"
S_PRE = 60 if PILOT else 300
N_POST = 90 if PILOT else 900
SEC_PER_STEP = 10

# Long-horizon extension: resume existing branches and push the post-shock
# measurement out to shock_step + N_POST_LONG (full: 2100 -> step 4200).
EXTEND = os.environ.get("EXTEND", "0") == "1"
N_POST_LONG = int(os.environ.get("N_POST_LONG", "180" if PILOT else "2100"))
WAVE_LONG = "post_long"
# Keep the windowed-truth window identical to the short `post` wave so the
# saturation-corrected `observed_interaction_recent` layer stays comparable.
LONG_RECENT_WINDOW_MIN = N_POST * SEC_PER_STEP // 60
# start_server() only persists when it returns, so advance in chunks and save
# between them. A crash then costs at most CHECKPOINT_STEPS of redo, not the
# whole multi-hour advance.
CHECKPOINT_STEPS = int(os.environ.get("CHECKPOINT_STEPS", "300"))

FORK_FROM = "preflight_the_ville_n25-1"
BASELINE_SIM = f"ncn_baseline_n25{SIM_SUFFIX}"
BRANCHES = {
    "control": f"ncn_control_n25{SIM_SUFFIX}",
    "hub": f"ncn_hub_n25{SIM_SUFFIX}",
    "broker": f"ncn_broker_n25{SIM_SUFFIX}",
}


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
                        move_files.append(int(fn.replace(".json", "")))
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


def _shock_targets(rs, sim_folder):
    from ground_truth_log import (
        build_ground_truth,
        highest_degree_agent,
        highest_betweenness_agent,
    )

    _, edge_rows = build_ground_truth(
        rs.personas, rs.sim_code, rs.step, rs.curr_time)
    hub_name, hub_deg = highest_degree_agent(edge_rows)
    broker_name, broker_bc = highest_betweenness_agent(edge_rows)
    targets = {
        "step": rs.step,
        "hub_agent": hub_name,
        "hub_degree": hub_deg,
        "broker_agent": broker_name,
        "broker_betweenness": broker_bc,
        "openai_model": PREFLIGHT_MODEL,
    }
    survey_dir = os.path.join(sim_folder, "survey")
    os.makedirs(survey_dir, exist_ok=True)
    path = os.path.join(survey_dir, "shock_targets.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(targets, f, indent=2)
    print(f"  Shock targets: hub={hub_name} (deg={hub_deg}), "
          f"broker={broker_name} (bc={broker_bc:.4f})")
    if hub_name == broker_name:
        print("  WARNING: hub and broker are the same agent.")
    return targets


def _ensure_baseline():
    from reverie import ReverieServer
    from perception_survey import run_perception_survey

    sim_folder = os.path.abspath(f"{FS_STORAGE}/{BASELINE_SIM}")
    survey_dir = os.path.join(sim_folder, "survey")
    shock_step = 1800 + S_PRE

    if os.path.exists(sim_folder):
        print(f"Resuming baseline {BASELINE_SIM}")
        rs = ReverieServer(BASELINE_SIM, BASELINE_SIM)
    else:
        print(f"Forking {FORK_FROM} -> {BASELINE_SIM}")
        rs = ReverieServer(FORK_FROM, BASELINE_SIM)

    frontend = HeadlessFrontend(sim_folder)
    frontend.start()
    try:
        if rs.step < shock_step:
            remaining = shock_step - rs.step
            print(f"\n{'=' * 60}")
            print(f"  BASELINE ADVANCE: {remaining} steps "
                  f"({rs.step} -> {shock_step})")
            print(f"{'=' * 60}")
            rs.start_server(remaining)
            rs.save()
            print(f"  -> Now at step {rs.step}")

        pre_csv = os.path.join(survey_dir, "perception_survey_pre.csv")
        if not os.path.isfile(pre_csv):
            print(f"\n{'=' * 60}")
            print(f"  PRE-SHOCK SURVEY @ step {rs.step}")
            print(f"{'=' * 60}")
            run_perception_survey(
                rs.personas, rs.sim_code, rs.step, rs.curr_time,
                survey_dir, wave_id="pre",
                recent_window_minutes=S_PRE * SEC_PER_STEP // 60)
            rs.save()
        else:
            print(f"  Pre survey exists: {pre_csv}")

        _shock_targets(rs, sim_folder)

        bg_path = os.path.join(
            survey_dir, "ground_truth", "background_social_edges.csv")
        if not os.path.isfile(bg_path):
            _run_script("bootstrap_n25_ground_truth_layers.py", BASELINE_SIM)
    finally:
        frontend.stop()

    return shock_step


def _run_branch(treatment, branch_code, shock_step):
    from reverie import ReverieServer
    from perception_survey import run_perception_survey

    sim_folder = os.path.abspath(f"{FS_STORAGE}/{branch_code}")
    survey_dir = os.path.join(sim_folder, "survey")
    post_csv = os.path.join(survey_dir, "perception_survey_post.csv")

    if os.path.isfile(post_csv):
        print(f"\n  Branch {branch_code}: post survey already done, skipping.")
        return

    if os.path.exists(sim_folder):
        print(f"\n  Resuming branch {branch_code}")
        rs = ReverieServer(branch_code, branch_code)
    else:
        print(f"\n  Forking {BASELINE_SIM} -> {branch_code}")
        rs = ReverieServer(BASELINE_SIM, branch_code)

    os.environ["GA_RUN_TAG"] = branch_code
    frontend = HeadlessFrontend(sim_folder)
    frontend.start()
    try:
        if treatment == "hub":
            msg = rs._try_shock_isolate_command(
                "shock isolate-hub", sim_folder)
            print(f"  Shock: {msg.strip()}")
        elif treatment == "broker":
            msg = rs._try_shock_isolate_command(
                "shock isolate-broker", sim_folder)
            print(f"  Shock: {msg.strip()}")

        target_step = shock_step + N_POST
        if rs.step < target_step:
            remaining = target_step - rs.step
            print(f"  Advancing {remaining} steps "
                  f"({rs.step} -> {target_step})")
            rs.start_server(remaining)
            rs.save()

        print(f"  POST-SHOCK SURVEY @ step {rs.step}")
        window_min = N_POST * SEC_PER_STEP // 60
        run_perception_survey(
            rs.personas, rs.sim_code, rs.step, rs.curr_time,
            survey_dir, wave_id="post",
            recent_window_minutes=window_min)
        rs.save()

        _run_script("analyze_survey.py", survey_dir)
    finally:
        frontend.stop()


def _read_shock_targets():
    path = os.path.abspath(
        f"{FS_STORAGE}/{BASELINE_SIM}/survey/shock_targets.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _advance_with_checkpoints(rs, target_step):
    """Advance to target_step, saving every CHECKPOINT_STEPS so a crash only
    loses the current chunk (resume picks up from the last saved step)."""
    while rs.step < target_step:
        chunk = min(CHECKPOINT_STEPS, target_step - rs.step)
        print(f"  Advancing {chunk} steps "
              f"({rs.step} -> {rs.step + chunk}) [target {target_step}]")
        rs.start_server(chunk)
        rs.save()
        print(f"  -> checkpoint saved at step {rs.step}")


def _extend_branch(treatment, branch_code, shock_step, targets):
    from reverie import ReverieServer
    from perception_survey import run_perception_survey

    sim_folder = os.path.abspath(f"{FS_STORAGE}/{branch_code}")
    survey_dir = os.path.join(sim_folder, "survey")
    long_csv = os.path.join(survey_dir, f"perception_survey_{WAVE_LONG}.csv")

    if not os.path.exists(sim_folder):
        print(f"  ERROR: branch {branch_code} not found; "
              f"run the base experiment first. Skipping.")
        return

    if os.path.isfile(long_csv):
        print(f"  Branch {branch_code}: {WAVE_LONG} survey present; "
              f"refreshing analysis only.")
        _run_script("analyze_survey.py", survey_dir)
        return

    print(f"\n  Resuming branch {branch_code} for extension")
    rs = ReverieServer(branch_code, branch_code)

    os.environ["GA_RUN_TAG"] = branch_code
    frontend = HeadlessFrontend(sim_folder)
    frontend.start()
    try:
        # Isolation is an in-memory flag and is NOT persisted across save /
        # resume, so re-apply the SAME treatment to the SAME agent that was
        # originally shocked (explicit name -> deterministic target).
        if treatment == "hub":
            cmd = f"shock isolate-hub {targets['hub_agent']}"
            msg = rs._try_shock_isolate_command(cmd, sim_folder)
            print(f"  Re-shock: {msg.strip()}")
        elif treatment == "broker":
            cmd = f"shock isolate-broker {targets['broker_agent']}"
            msg = rs._try_shock_isolate_command(cmd, sim_folder)
            print(f"  Re-shock: {msg.strip()}")

        target_step = shock_step + N_POST_LONG
        _advance_with_checkpoints(rs, target_step)

        print(f"  {WAVE_LONG.upper()} SURVEY @ step {rs.step}")
        run_perception_survey(
            rs.personas, rs.sim_code, rs.step, rs.curr_time,
            survey_dir, wave_id=WAVE_LONG,
            recent_window_minutes=LONG_RECENT_WINDOW_MIN)
        rs.save()

        _run_script("analyze_survey.py", survey_dir)
    finally:
        frontend.stop()


def _run_extension(model):
    shock_step = 1800 + S_PRE
    target_step = shock_step + N_POST_LONG
    targets = _read_shock_targets()

    print(f"\n{'=' * 60}")
    print("  NETWORK COGNITION EXPERIMENT -- LONG-HORIZON EXTENSION")
    print(f"  Mode: {'PILOT' if PILOT else 'FULL'}")
    print(f"  Model: {model}")
    print(f"  shock_step={shock_step}, target_step={target_step} "
          f"(N_POST_LONG={N_POST_LONG})")
    print(f"  recent window: {LONG_RECENT_WINDOW_MIN} min "
          f"(matched to short `post` wave)")
    print(f"  hub={targets.get('hub_agent')}, "
          f"broker={targets.get('broker_agent')}")
    print(f"{'=' * 60}")

    for treatment, branch_code in BRANCHES.items():
        print(f"\n{'=' * 60}")
        print(f"  EXTEND BRANCH: {treatment} ({branch_code})")
        print(f"{'=' * 60}")
        _extend_branch(treatment, branch_code, shock_step, targets)

    os.environ["NCN_SUFFIX"] = SIM_SUFFIX
    os.environ["NCN_POST_WAVE"] = WAVE_LONG
    _run_script("ncn_did_summary.py")

    print(f"\n{'=' * 60}")
    print("  LONG-HORIZON EXTENSION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Branches advanced to step {target_step}, "
          f"wave `{WAVE_LONG}` surveyed.")
    print(f"  Results: {BASELINE_SIM}/survey/ncn_did_summary_{WAVE_LONG}.csv")


def main():
    load_dotenv()
    model = verify_model_lock()

    if EXTEND:
        _run_extension(model)
        return

    print(f"\n{'=' * 60}")
    print("  NETWORK COGNITION EXPERIMENT")
    print(f"  Mode: {'PILOT' if PILOT else 'FULL'}")
    print(f"  Model: {model}")
    print(f"  S_PRE={S_PRE}, N_POST={N_POST}")
    print(f"{'=' * 60}")

    shock_step = _ensure_baseline()

    for treatment, branch_code in BRANCHES.items():
        print(f"\n{'=' * 60}")
        print(f"  BRANCH: {treatment} ({branch_code})")
        print(f"{'=' * 60}")
        _run_branch(treatment, branch_code, shock_step)

    os.environ["NCN_SUFFIX"] = SIM_SUFFIX
    _run_script("ncn_did_summary.py")

    print(f"\n{'=' * 60}")
    print("  EXPERIMENT COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Baseline: {BASELINE_SIM} @ step {shock_step}")
    for branch_code in BRANCHES.values():
        print(f"  Branch:   {branch_code}")
    print("\n  Results: survey/ncn_did_summary.csv")


if __name__ == "__main__":
    main()
