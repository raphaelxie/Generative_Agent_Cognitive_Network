#!/usr/bin/env python3
"""
Identification-first NCN experiment (Window 1: advance forks + noise floor).

Forks the saved pre-shock baseline (step 2100) into replicated treatment arms,
advances each fork post-shock, logs network structure at checkpoints, and
optionally runs K measurement-noise surveys on the frozen baseline state.

Window 1 (default): simulation + noise floor if time remains.
Window 2 (ID_WINDOW=2): post-shock surveys + analysis on saved fork states.

Usage (from reverie/backend_server/):
  ID_WINDOW=1 caffeinate -i python run_identification_experiment.py
  ID_WINDOW=2 caffeinate -i python run_identification_experiment.py
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from ncn_model_utils import load_dotenv, verify_model_lock, PREFLIGHT_MODEL
from run_network_cognition_experiment import HeadlessFrontend

FS_STORAGE = "../../environment/frontend_server/storage"
BASELINE_SIM = "ncn_baseline_n25_full"
SHOCK_STEP = 2100
N_POST = int(os.environ.get("ID_N_POST", "1200"))
TARGET_STEP = SHOCK_STEP + N_POST
SEC_PER_STEP = 10
CHECKPOINT_STEPS = int(os.environ.get("CHECKPOINT_STEPS", "300"))
RECENT_WINDOW_MIN = N_POST * SEC_PER_STEP // 60

ID_WINDOW = int(os.environ.get("ID_WINDOW", "1"))
NOISE_REPS = int(os.environ.get("ID_NOISE_REPS", "5"))

FORKS = [
    ("control", 1, "ncn_id_control_r1"),
    ("control", 2, "ncn_id_control_r2"),
    ("hub", 1, "ncn_id_hub_r1"),
    ("hub", 2, "ncn_id_hub_r2"),
    ("broker", 1, "ncn_id_broker_r1"),
    ("broker", 2, "ncn_id_broker_r2"),
]

MANIFEST_NAME = "ncn_id_manifest.json"


def _manifest_path():
    return os.path.abspath(
        os.path.join(FS_STORAGE, BASELINE_SIM, "survey", MANIFEST_NAME))


def _load_manifest():
    path = _manifest_path()
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_manifest(manifest):
    path = _manifest_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    manifest["updated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest saved: {path}")


def _init_manifest(targets):
    manifest = {
        "design": "identification_v1",
        "baseline": BASELINE_SIM,
        "shock_step": SHOCK_STEP,
        "target_step": TARGET_STEP,
        "n_post": N_POST,
        "recent_window_minutes": RECENT_WINDOW_MIN,
        "hub_agent": targets.get("hub_agent"),
        "broker_agent": targets.get("broker_agent"),
        "openai_model": PREFLIGHT_MODEL,
        "forks": {},
        "noise_floor": {"k": NOISE_REPS, "completed": []},
        "window_1_sim_complete": False,
        "window_2_surveys_complete": False,
    }
    for treatment, rep, code in FORKS:
        manifest["forks"][code] = {
            "treatment": treatment,
            "replicate": rep,
            "step": None,
            "sim_complete": False,
            "post_survey": False,
        }
    return manifest


def _read_shock_targets():
    path = os.path.abspath(
        os.path.join(FS_STORAGE, BASELINE_SIM, "survey/shock_targets.json"))
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_script(script_name, *args):
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, script_name), *args]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)


def _advance_with_checkpoints(rs, target_step, on_checkpoint=None):
    while rs.step < target_step:
        chunk = min(CHECKPOINT_STEPS, target_step - rs.step)
        print(f"  Advancing {chunk} steps "
              f"({rs.step} -> {rs.step + chunk}) [target {target_step}]")
        rs.start_server(chunk)
        rs.save()
        print(f"  -> checkpoint saved at step {rs.step}")
        if on_checkpoint:
            on_checkpoint(rs)


def _log_network_checkpoint(rs, sim_folder, wave_id=""):
    from ground_truth_log import (
        build_ground_truth,
        graph_scope_summary,
        write_ground_truth_csv,
    )

    survey_dir = os.path.join(sim_folder, "survey")
    gt_dir = os.path.join(survey_dir, "ground_truth")
    write_ground_truth_csv(
        rs.personas, rs.sim_code, rs.step, rs.curr_time,
        gt_dir, recent_window_minutes=RECENT_WINDOW_MIN, wave_id=wave_id)

    _, edge_rows = build_ground_truth(
        rs.personas, rs.sim_code, rs.step, rs.curr_time,
        recent_window_minutes=RECENT_WINDOW_MIN, wave_id=wave_id)

    cum = graph_scope_summary(edge_rows, "count_cumulative")
    rec = graph_scope_summary(edge_rows, "count_recent")

    row = {
        "sim_code": rs.sim_code,
        "step": rs.step,
        "sim_time": rs.curr_time.strftime("%Y-%m-%d %H:%M:%S"),
        "density_cumulative": cum["density"],
        "density_recent": rec["density"],
        "n_edges_cumulative": cum["n_edges"],
        "n_edges_recent": rec["n_edges"],
        "mean_degree_cumulative": cum["mean_degree"],
        "component_count_cumulative": cum["component_count"],
        "largest_component_cumulative": cum["largest_component_size"],
        "bridge_edge_count_cumulative": cum["bridge_edge_count"],
        "hub": cum["hub"],
        "broker": cum["broker"],
        "hub_eq_broker": cum["hub_eq_broker"],
    }

    struct_path = os.path.join(survey_dir, "network_structure_over_time.csv")
    write_header = not os.path.isfile(struct_path)
    with open(struct_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def _apply_shock(rs, treatment, targets, sim_folder):
    if treatment == "control":
        return
    if treatment == "hub":
        cmd = f"shock isolate-hub {targets['hub_agent']}"
    else:
        cmd = f"shock isolate-broker {targets['broker_agent']}"
    msg = rs._try_shock_isolate_command(cmd, sim_folder)
    print(f"  Shock ({treatment}): {msg.strip()}")


def _run_fork_sim(treatment, rep, fork_code, targets, manifest):
    from reverie import ReverieServer

    fork_entry = manifest["forks"][fork_code]
    if fork_entry.get("sim_complete"):
        print(f"  {fork_code}: sim already complete @ step {fork_entry.get('step')}")
        return

    sim_folder = os.path.abspath(os.path.join(FS_STORAGE, fork_code))
    survey_dir = os.path.join(sim_folder, "survey")
    meta_path = os.path.join(sim_folder, "reverie", "meta.json")

    if os.path.isdir(sim_folder) and not os.path.isfile(meta_path):
        print(f"  Removing incomplete fork folder {fork_code}")
        shutil.rmtree(sim_folder)

    if os.path.isfile(meta_path):
        print(f"\n  Resuming fork {fork_code}")
        rs = ReverieServer(fork_code, fork_code)
    else:
        print(f"\n  Forking {BASELINE_SIM} -> {fork_code} "
              f"({treatment} r{rep})")
        rs = ReverieServer(BASELINE_SIM, fork_code)

    os.makedirs(survey_dir, exist_ok=True)

    targets_path = os.path.join(survey_dir, "shock_targets.json")
    with open(targets_path, "w", encoding="utf-8") as f:
        json.dump({**targets, "fork": fork_code, "treatment": treatment,
                   "replicate": rep}, f, indent=2)

    os.environ["GA_RUN_TAG"] = fork_code
    frontend = HeadlessFrontend(sim_folder)
    frontend.start()
    try:
        _apply_shock(rs, treatment, targets, sim_folder)

        if rs.step == SHOCK_STEP:
            _log_network_checkpoint(rs, sim_folder, wave_id="shock")

        def on_checkpoint(server):
            _log_network_checkpoint(server, sim_folder, wave_id="checkpoint")

        _advance_with_checkpoints(rs, TARGET_STEP, on_checkpoint=on_checkpoint)
        _log_network_checkpoint(rs, sim_folder, wave_id="post_shock")

        fork_entry["step"] = rs.step
        fork_entry["sim_complete"] = rs.step >= TARGET_STEP
        _save_manifest(manifest)

        _run_script("survey_network_summary.py", survey_dir)
    finally:
        frontend.stop()


def _run_sim_phase(manifest, targets):
    print(f"\n{'=' * 60}")
    print("  IDENTIFICATION WINDOW — SIM PHASE")
    print(f"  target_step={TARGET_STEP} (+{N_POST} post-shock)")
    print(f"  forks: {len(FORKS)}")
    print(f"{'=' * 60}")

    for treatment, rep, fork_code in FORKS:
        print(f"\n{'=' * 60}")
        print(f"  FORK: {fork_code} ({treatment} r{rep})")
        print(f"{'=' * 60}")
        _run_fork_sim(treatment, rep, fork_code, targets, manifest)

    all_done = all(
        manifest["forks"][c]["sim_complete"] for _, _, c in FORKS)
    manifest["window_1_sim_complete"] = all_done
    _save_manifest(manifest)


def _run_noise_phase(manifest):
    from reverie import ReverieServer
    from perception_survey import run_perception_survey

    completed = set(manifest["noise_floor"].get("completed", []))
    pending = [
        f"noise_r{i:02d}" for i in range(1, NOISE_REPS + 1)
        if f"noise_r{i:02d}" not in completed]

    if not pending:
        print("  Noise floor: all reps complete.")
        return

    sim_folder = os.path.abspath(os.path.join(FS_STORAGE, BASELINE_SIM))
    survey_dir = os.path.join(sim_folder, "survey")
    rs = ReverieServer(BASELINE_SIM, BASELINE_SIM)

    print(f"\n{'=' * 60}")
    print(f"  NOISE FLOOR: {len(pending)} reps @ frozen step {rs.step}")
    print(f"{'=' * 60}")

    for wave_id in pending:
        out_csv = os.path.join(survey_dir, f"perception_survey_{wave_id}.csv")
        if os.path.isfile(out_csv):
            completed.add(wave_id)
            continue

        print(f"\n  Running {wave_id} @ step {rs.step}")
        run_perception_survey(
            rs.personas, rs.sim_code, rs.step, rs.curr_time,
            survey_dir, wave_id=wave_id,
            recent_window_minutes=RECENT_WINDOW_MIN)
        completed.add(wave_id)
        manifest["noise_floor"]["completed"] = sorted(completed)
        _save_manifest(manifest)
        _run_script("analyze_survey.py", survey_dir)


def _run_survey_phase(manifest, targets):
    from reverie import ReverieServer
    from perception_survey import run_perception_survey

    print(f"\n{'=' * 60}")
    print("  IDENTIFICATION WINDOW — SURVEY PHASE")
    print(f"{'=' * 60}")

    for treatment, rep, fork_code in FORKS:
        entry = manifest["forks"][fork_code]
        if not entry.get("sim_complete"):
            print(f"  SKIP {fork_code}: sim not complete")
            continue
        if entry.get("post_survey"):
            print(f"  SKIP {fork_code}: post survey done")
            continue

        sim_folder = os.path.abspath(os.path.join(FS_STORAGE, fork_code))
        survey_dir = os.path.join(sim_folder, "survey")
        post_csv = os.path.join(survey_dir, "perception_survey_post.csv")

        rs = ReverieServer(fork_code, fork_code)
        _apply_shock(rs, treatment, targets, sim_folder)

        print(f"\n  POST SURVEY: {fork_code} @ step {rs.step}")
        run_perception_survey(
            rs.personas, rs.sim_code, rs.step, rs.curr_time,
            survey_dir, wave_id="post",
            recent_window_minutes=RECENT_WINDOW_MIN)
        rs.save()
        entry["post_survey"] = os.path.isfile(post_csv)
        _save_manifest(manifest)
        _run_script("analyze_survey.py", survey_dir)

    all_surveyed = all(
        manifest["forks"][c].get("post_survey") for _, _, c in FORKS
        if manifest["forks"][c].get("sim_complete"))
    manifest["window_2_surveys_complete"] = all_surveyed
    _save_manifest(manifest)


def main():
    load_dotenv()
    model = verify_model_lock()
    targets = _read_shock_targets()

    manifest = _load_manifest()
    if manifest is None:
        manifest = _init_manifest(targets)
        _save_manifest(manifest)

    print(f"\n{'=' * 60}")
    print("  NCN IDENTIFICATION EXPERIMENT")
    print(f"  Window: {ID_WINDOW}")
    print(f"  Model: {model}")
    print(f"  Baseline: {BASELINE_SIM} @ shock_step {SHOCK_STEP}")
    print(f"  Target: {TARGET_STEP} (+{N_POST} steps)")
    print(f"  Hub: {targets['hub_agent']}, Broker: {targets['broker_agent']}")
    print(f"{'=' * 60}")

    if ID_WINDOW == 1:
        _run_sim_phase(manifest, targets)
        _run_noise_phase(manifest)
    elif ID_WINDOW == 2:
        _run_survey_phase(manifest, targets)
    else:
        _run_sim_phase(manifest, targets)
        _run_noise_phase(manifest)
        _run_survey_phase(manifest, targets)

    print(f"\n{'=' * 60}")
    print("  IDENTIFICATION RUN COMPLETE (this window)")
    print(f"  sim_complete: {manifest.get('window_1_sim_complete')}")
    print(f"  noise_reps: {len(manifest['noise_floor'].get('completed', []))}"
          f"/{NOISE_REPS}")
    print(f"  surveys_complete: {manifest.get('window_2_surveys_complete')}")
    print(f"  Manifest: {_manifest_path()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
