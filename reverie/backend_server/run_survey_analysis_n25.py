#!/usr/bin/env python3
"""
Run survey pre + analysis for n=25 (assumes burn-in state already saved).

Does not run burn-in. Refreshes Layer 1 via survey, Layer 2 via
background_social_truth if missing, Layer 3 via analyze_survey.py.

Usage:
  python run_survey_analysis_n25.py [sim_code]
"""
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

DEFAULT_SIM = "preflight_the_ville_n25-1"
FS_STORAGE = "../../environment/frontend_server/storage"


def _run(script, *args):
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, script), *args]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)


def main():
    sim_code = sys.argv[1].strip() if len(sys.argv) > 1 else DEFAULT_SIM
    sim_folder = os.path.abspath(f"{FS_STORAGE}/{sim_code}")
    survey_dir = os.path.join(sim_folder, "survey")

    from reverie import ReverieServer
    from perception_survey import run_perception_survey

    rs = ReverieServer(sim_code, sim_code)
    print(f"Loaded {sim_code} at step {rs.step}, n={len(rs.personas)}")

    pre_csv = os.path.join(survey_dir, "perception_survey_pre.csv")
    if not os.path.isfile(pre_csv):
        print("Running survey pre ...")
        path = run_perception_survey(
            rs.personas, rs.sim_code, rs.step, rs.curr_time,
            survey_dir, wave_id="pre")
        print(f"Wrote {path}")
        try:
            rs.save()
        except AttributeError as exc:
            if "strftime" in str(exc):
                print(f"WARNING: skipped rs.save() ({exc})")
            else:
                raise
    else:
        print(f"Survey exists: {pre_csv}")

    bg = os.path.join(survey_dir, "ground_truth", "background_social_edges.csv")
    if not os.path.isfile(bg):
        _run("background_social_truth.py", survey_dir)

    metrics = os.path.join(survey_dir, "analysis_micro_tie_metrics_by_truth.csv")
    if not os.path.isfile(metrics):
        _run("analyze_survey.py", survey_dir)

    print("Done. Verify with:")
    print(f"  python verify_preflight_n25.py {survey_dir}")


if __name__ == "__main__":
    main()
