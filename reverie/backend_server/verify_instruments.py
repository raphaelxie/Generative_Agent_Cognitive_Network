#!/usr/bin/env python3
"""
Phase A: Verification of all perception instruments.

Loads prepost_n15_calibration-1 (step 1800, n=15) in-place (no copy),
runs a single survey wave with all 8 question types, then validates the
output files. After the survey, runs the downstream analysis pipeline
(analyze_survey, survey_network_summary, figures/generate_figures) and
checks their outputs.

This script makes ~960 LLM calls (one full survey wave on n=15).

Usage (from reverie/backend_server/):
  python verify_instruments.py
"""
import csv
import json
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

SIM_CODE = "prepost_n15_calibration-1"
WAVE_ID = "verify_all"
N_PERSONAS = 15
SURVEY_REL = f"../../environment/frontend_server/storage/{SIM_CODE}/survey"
SURVEY_DIR = os.path.abspath(SURVEY_REL)

EXPECTED_QUESTION_TYPES = {
    "micro_tie",
    "micro_tie_interaction",
    "micro_tie_social",
    "micro_tie_group",
    "centrality_rank",
    "bridge_rank",
    "self_position",
    "community_group",
    "community_group_n_groups",
}

MICRO_TIE_TYPES = {
    "micro_tie", "micro_tie_interaction", "micro_tie_social", "micro_tie_group"
}

EXPECTED_MICRO_ROWS_PER_TYPE = N_PERSONAS * N_PERSONAS * (N_PERSONAS - 1)
EXPECTED_RANK_ROWS = N_PERSONAS * N_PERSONAS
EXPECTED_SELF_ROWS = N_PERSONAS
MAX_FAIL_SAFE_RATE = 0.20

results = []


def check(condition, description):
    status = "PASS" if condition else "FAIL"
    results.append((status, description))
    print(f"  [{status}] {description}")
    return condition


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Phase A1: Import verification ─────────────────────────────────────

section("A1: Import Verification")

try:
    sys.path.insert(0, SCRIPT_DIR)
    from perception_survey import run_perception_survey
    check(True, "perception_survey imports OK")
except Exception as e:
    check(False, f"perception_survey import failed: {e}")

try:
    from analyze_survey import main as analyze_main
    check(True, "analyze_survey imports OK")
except Exception as e:
    check(False, f"analyze_survey import failed: {e}")

try:
    from survey_network_summary import main as network_summary_main
    check(True, "survey_network_summary imports OK")
except Exception as e:
    check(False, f"survey_network_summary import failed: {e}")

try:
    sys.path.insert(0, os.path.join(SCRIPT_DIR, "figures"))
    from generate_figures import main as figures_main
    check(True, "figures/generate_figures imports OK")
except Exception as e:
    check(False, f"figures/generate_figures import failed: {e}")

# ── Phase A2: Load sim and run survey ─────────────────────────────────

section("A2: Survey Execution")

print(f"\n  Loading ReverieServer (in-place, no copy): {SIM_CODE}")
print(f"  This will make ~960 LLM calls. Estimated time: 10-20 min.\n")

try:
    from reverie import ReverieServer
    rs = ReverieServer(SIM_CODE, SIM_CODE)
    check(True, f"ReverieServer loaded: step={rs.step}, n_personas={len(rs.personas)}")
    check(rs.step == 1800, f"Step is 1800 (got {rs.step})")
    check(len(rs.personas) == N_PERSONAS, f"N personas = {N_PERSONAS} (got {len(rs.personas)})")
except Exception as e:
    check(False, f"ReverieServer load failed: {e}")
    print("\nCannot proceed without ReverieServer. Exiting.")
    sys.exit(1)

start_time = time.time()
try:
    survey_path = run_perception_survey(
        rs.personas, rs.sim_code, rs.step, rs.curr_time,
        SURVEY_DIR, wave_id=WAVE_ID)
    elapsed = time.time() - start_time
    check(True, f"Survey completed in {elapsed:.1f}s -> {os.path.basename(survey_path)}")
except Exception as e:
    elapsed = time.time() - start_time
    check(False, f"Survey failed after {elapsed:.1f}s: {e}")
    import traceback
    traceback.print_exc()
    print("\nCannot proceed without survey output. Exiting.")
    sys.exit(1)

# ── Phase A3: Validate survey outputs ─────────────────────────────────

section("A3: Survey Output Validation")

survey_csv = os.path.join(SURVEY_DIR, f"perception_survey_{WAVE_ID}.csv")
meta_json = os.path.join(SURVEY_DIR, f"perception_survey_{WAVE_ID}_meta.json")
diag_jsonl = os.path.join(SURVEY_DIR, f"retrieval_diagnostics_{WAVE_ID}.jsonl")

check(os.path.isfile(survey_csv), f"Survey CSV exists: {os.path.basename(survey_csv)}")
check(os.path.isfile(meta_json), f"Meta JSON exists: {os.path.basename(meta_json)}")
check(os.path.isfile(diag_jsonl), f"Diagnostics JSONL exists: {os.path.basename(diag_jsonl)}")

with open(survey_csv, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

question_types_found = {r["question_type"] for r in rows}
for qt in EXPECTED_QUESTION_TYPES:
    check(qt in question_types_found, f"Question type present: {qt}")

missing = EXPECTED_QUESTION_TYPES - question_types_found
if missing:
    print(f"  WARNING: Missing question types: {missing}")

for qt in MICRO_TIE_TYPES:
    qt_rows = [r for r in rows if r["question_type"] == qt]
    n = len(qt_rows)
    check(n == EXPECTED_MICRO_ROWS_PER_TYPE,
          f"{qt}: {n} rows (expected {EXPECTED_MICRO_ROWS_PER_TYPE})")

rank_types = {"centrality_rank", "bridge_rank"}
for qt in rank_types:
    qt_rows = [r for r in rows if r["question_type"] == qt]
    n = len(qt_rows)
    check(n == EXPECTED_RANK_ROWS, f"{qt}: {n} rows (expected {EXPECTED_RANK_ROWS})")

self_rows = [r for r in rows if r["question_type"] == "self_position"]
check(len(self_rows) == EXPECTED_SELF_ROWS,
      f"self_position: {len(self_rows)} rows (expected {EXPECTED_SELF_ROWS})")

cg_rows = [r for r in rows if r["question_type"] == "community_group"]
cg_n_rows = [r for r in rows if r["question_type"] == "community_group_n_groups"]
check(len(cg_rows) > 0, f"community_group: {len(cg_rows)} rows")
check(len(cg_n_rows) == N_PERSONAS,
      f"community_group_n_groups: {len(cg_n_rows)} rows (expected {N_PERSONAS})")

n_groups_values = [int(r["value"]) for r in cg_n_rows if r["value"].isdigit()]
if n_groups_values:
    check(max(n_groups_values) > 1,
          f"Community groups variable: max groups = {max(n_groups_values)}")
    check(len(set(n_groups_values)) > 1,
          f"Community group counts vary across respondents: {sorted(set(n_groups_values))}")

for qt in EXPECTED_QUESTION_TYPES - {"community_group_n_groups"}:
    qt_rows = [r for r in rows if r["question_type"] == qt]
    if qt_rows:
        n_fs = sum(1 for r in qt_rows if r["is_fail_safe"] == "1")
        rate = n_fs / len(qt_rows)
        check(rate <= MAX_FAIL_SAFE_RATE,
              f"{qt} fail-safe rate: {rate:.1%} ({n_fs}/{len(qt_rows)})")

gt_dir = os.path.join(SURVEY_DIR, "ground_truth")
check(os.path.isdir(gt_dir), "ground_truth/ directory exists")
gt_edges = os.path.join(gt_dir, "ground_truth_edges_1800.csv")
gt_chats = os.path.join(gt_dir, "ground_truth_chats_1800.csv")
check(os.path.isfile(gt_edges), "ground_truth_edges_1800.csv exists")
check(os.path.isfile(gt_chats), "ground_truth_chats_1800.csv exists")

with open(meta_json, encoding="utf-8") as f:
    meta = json.load(f)
check(meta.get("n_retrieval_calls", 0) > 0,
      f"Retrieval calls logged: {meta.get('n_retrieval_calls')}")

# ── Phase A4: Downstream analysis ─────────────────────────────────────

section("A4: Downstream Analysis Pipeline")

print("  Running analyze_survey.py ...")
r = subprocess.run(
    [sys.executable, "analyze_survey.py", SURVEY_DIR],
    capture_output=True, text=True, cwd=SCRIPT_DIR)
check(r.returncode == 0, f"analyze_survey.py exit code = {r.returncode}")
if r.returncode != 0:
    print(f"  STDERR: {r.stderr[:500]}")

print("  Running survey_network_summary.py ...")
r = subprocess.run(
    [sys.executable, "survey_network_summary.py", SURVEY_DIR],
    capture_output=True, text=True, cwd=SCRIPT_DIR)
check(r.returncode == 0, f"survey_network_summary.py exit code = {r.returncode}")
if r.returncode != 0:
    print(f"  STDERR: {r.stderr[:500]}")

print("  Running figures/generate_figures.py ...")
r = subprocess.run(
    [sys.executable, "figures/generate_figures.py", SURVEY_DIR],
    capture_output=True, text=True, cwd=SCRIPT_DIR)
check(r.returncode == 0, f"figures/generate_figures.py exit code = {r.returncode}")
if r.returncode != 0:
    print(f"  STDERR: {r.stderr[:500]}")

# ── Phase A5: Validate analysis outputs ───────────────────────────────

section("A5: Analysis Output Validation")

by_truth = os.path.join(SURVEY_DIR, "analysis_micro_tie_metrics_by_truth.csv")
by_construct = os.path.join(SURVEY_DIR, "analysis_micro_tie_by_construct.csv")
bridge_metrics = os.path.join(SURVEY_DIR, "analysis_bridge_rank_metrics.csv")
community_metrics = os.path.join(SURVEY_DIR, "analysis_community_group_metrics.csv")
network_summary = os.path.join(SURVEY_DIR, "network_summary_over_time.csv")

check(os.path.isfile(by_truth), "analysis_micro_tie_metrics_by_truth.csv exists")
check(os.path.isfile(by_construct), "analysis_micro_tie_by_construct.csv exists")
check(os.path.isfile(bridge_metrics), "analysis_bridge_rank_metrics.csv exists")
check(os.path.isfile(community_metrics), "analysis_community_group_metrics.csv exists")
check(os.path.isfile(network_summary), "network_summary_over_time.csv exists")

if os.path.isfile(network_summary):
    with open(network_summary, newline="", encoding="utf-8") as f:
        ns_rows = list(csv.DictReader(f))
    ns_cols = set(ns_rows[0].keys()) if ns_rows else set()
    new_cols = {"transitivity", "avg_clustering", "modularity_louvain",
                "n_communities_louvain"}
    for col in new_cols:
        check(col in ns_cols, f"network_summary has column: {col}")

if os.path.isfile(community_metrics):
    with open(community_metrics, newline="", encoding="utf-8") as f:
        cm_rows = list(csv.DictReader(f))
    check(len(cm_rows) > 0, f"community_group_metrics has {len(cm_rows)} rows")
    cm_cols = set(cm_rows[0].keys()) if cm_rows else set()
    for metric in ("nmi", "ari"):
        check(any(metric in c.lower() for c in cm_cols),
              f"community_group_metrics has {metric.upper()} column")

if os.path.isfile(by_construct):
    with open(by_construct, newline="", encoding="utf-8") as f:
        bc_rows = list(csv.DictReader(f))
    check(len(bc_rows) > 0, f"micro_tie_by_construct has {len(bc_rows)} rows")
    constructs_found = {r.get("construct") or r.get("question_type", "")
                        for r in bc_rows}
    for c in ("micro_tie_interaction", "micro_tie_social", "micro_tie_group"):
        check(c in constructs_found, f"by_construct includes: {c}")

gt_communities = os.path.join(gt_dir, "ground_truth_communities_1800.csv")
check(os.path.isfile(gt_communities), "ground_truth_communities_1800.csv exists")

figures_dir = os.path.join(SURVEY_DIR, "figures")
if os.path.isdir(figures_dir):
    fig_files = [f for f in os.listdir(figures_dir)
                 if f.endswith(".pdf") or f.endswith(".png")]
    check(len(fig_files) > 0, f"Figures generated: {len(fig_files)} files")
else:
    check(False, "figures/ directory exists")

# ── Summary ───────────────────────────────────────────────────────────

section("SUMMARY")

n_pass = sum(1 for s, _ in results if s == "PASS")
n_fail = sum(1 for s, _ in results if s == "FAIL")
total = len(results)

print(f"\n  {n_pass}/{total} checks passed, {n_fail} failed.\n")

if n_fail > 0:
    print("  Failed checks:")
    for s, desc in results:
        if s == "FAIL":
            print(f"    - {desc}")
    print()

sys.exit(0 if n_fail == 0 else 1)
