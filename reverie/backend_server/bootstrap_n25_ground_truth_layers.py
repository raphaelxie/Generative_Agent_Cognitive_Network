#!/usr/bin/env python3
"""
Bootstrap n=25 ground-truth artifacts for a simulation folder.

Writes Layer 1 (observed_interaction edges at current step) and Layer 2
(background_social_edges.csv) without running a survey. Run analyze_survey.py
after a survey wave exists to materialize Layer 3 (union) and metrics.

Usage (from reverie/backend_server/):
  python bootstrap_n25_ground_truth_layers.py [sim_code]

Default sim_code: preflight_the_ville_n25-1
"""
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

DEFAULT_SIM = "preflight_the_ville_n25-1"
FS_STORAGE = "../../environment/frontend_server/storage"
EXPECTED_PAIRS = 25 * 24 // 2


def main():
    sim_code = sys.argv[1].strip() if len(sys.argv) > 1 else DEFAULT_SIM
    sim_folder = os.path.abspath(f"{FS_STORAGE}/{sim_code}")
    survey_dir = os.path.join(sim_folder, "survey")
    gt_dir = os.path.join(survey_dir, "ground_truth")

    if not os.path.isdir(sim_folder):
        raise SystemExit(f"sim folder not found: {sim_folder}")

    from reverie import ReverieServer
    from ground_truth_log import write_ground_truth_csv

    print(f"Loading {sim_code} ...")
    rs = ReverieServer(sim_code, sim_code)
    n = len(rs.personas)
    if n != 25:
        print(f"WARNING: expected 25 personas, got {n}")

    os.makedirs(gt_dir, exist_ok=True)
    chats_path, edges_path = write_ground_truth_csv(
        rs.personas, rs.sim_code, rs.step, rs.curr_time,
        gt_dir, wave_id="bootstrap")
    print(f"Layer 1: {edges_path}")

    import csv
    with open(edges_path, newline="", encoding="utf-8") as f:
        n_rows = sum(1 for _ in csv.DictReader(f))
    if n_rows != EXPECTED_PAIRS:
        raise SystemExit(f"expected {EXPECTED_PAIRS} edge rows, got {n_rows}")

    bg_script = os.path.join(SCRIPT_DIR, "background_social_truth.py")
    print("Layer 2: running background_social_truth.py ...")
    subprocess.run(
        [sys.executable, bg_script, survey_dir],
        check=True,
        cwd=SCRIPT_DIR,
    )

    bg_path = os.path.join(gt_dir, "background_social_edges.csv")
    with open(bg_path, newline="", encoding="utf-8") as f:
        bg_rows = list(csv.DictReader(f))
    n_pos = sum(int(r.get("tie_background", 0) or 0) for r in bg_rows)
    print(f"Layer 2: {bg_path} ({n_pos} positive / {len(bg_rows)} dyads)")
    print("Layer 3 (background_or_interaction): run analyze_survey.py after survey pre")


if __name__ == "__main__":
    main()
