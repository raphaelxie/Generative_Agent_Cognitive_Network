#!/usr/bin/env python3
"""
Verify n=15 pre-flight artifacts and analysis outputs (plan pass / not-ready).

Usage:
  python verify_preflight_n15.py <survey_dir>

Example:
  python verify_preflight_n15.py ../../environment/frontend_server/storage/preflight_the_ville_n15-1/survey
"""
import csv
import json
import os
import sys

EXPECTED_PAIRS = 15 * 14 // 2  # 105
EXPECTED_RESPONDENTS = 15
EXPECTED_RETRIEVAL_LINES = EXPECTED_RESPONDENTS * (
    15 + 1
)  # 15 micro-tie focal retrievals + 1 social per respondent

# Directed micro-tie survey rows per wave: each respondent, each j, each k!=j
EXPECTED_MICRO_TIE_ROWS = EXPECTED_RESPONDENTS * EXPECTED_RESPONDENTS * (
    EXPECTED_RESPONDENTS - 1
)


def fail(msg):
    print(f"NOT READY: {msg}")
    sys.exit(1)


def ok(msg):
    print(f"PASS: {msg}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_preflight_n15.py <survey_dir>")
        sys.exit(2)
    survey_dir = os.path.abspath(sys.argv[1].strip())
    if not os.path.isdir(survey_dir):
        fail(f"not a directory: {survey_dir}")

    pre_csv = os.path.join(survey_dir, "perception_survey_pre.csv")
    pre_meta = os.path.join(survey_dir, "perception_survey_pre_meta.json")
    diag = os.path.join(survey_dir, "retrieval_diagnostics_pre.jsonl")
    shock = os.path.join(survey_dir, "shock_log.jsonl")
    metrics = os.path.join(survey_dir, "analysis_micro_tie_metrics.csv")
    gt_dir = os.path.join(survey_dir, "ground_truth")

    for p in (pre_csv, pre_meta, diag):
        if not os.path.isfile(p):
            fail(f"missing file: {p}")

    with open(pre_meta, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("n_personas") != EXPECTED_RESPONDENTS:
        fail(f"meta n_personas={meta.get('n_personas')!r}, expected {EXPECTED_RESPONDENTS}")
    n_ret = meta.get("n_retrieval_calls")
    if n_ret != EXPECTED_RETRIEVAL_LINES:
        fail(
            f"meta n_retrieval_calls={n_ret!r}, expected {EXPECTED_RETRIEVAL_LINES} "
            f"(15*(15+1) per preflight design)"
        )

    with open(pre_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        fail("perception_survey_pre.csv is empty")
    respondents = {r["respondent"] for r in rows}
    if len(respondents) < EXPECTED_RESPONDENTS:
        fail(f"expected {EXPECTED_RESPONDENTS} respondents, got {len(respondents)}")

    with open(diag, encoding="utf-8") as f:
        diag_lines = [ln for ln in f if ln.strip()]
    if len(diag_lines) != EXPECTED_RETRIEVAL_LINES:
        fail(
            f"retrieval_diagnostics_pre.jsonl has {len(diag_lines)} lines, "
            f"expected {EXPECTED_RETRIEVAL_LINES}"
        )
    sample = json.loads(diag_lines[0])
    for key in ("focal_query", "n_candidates", "n_returned", "nodes"):
        if key not in sample:
            fail(f"diagnostics line missing key {key!r}")

    # Ground truth edges
    step = rows[0].get("step", "").strip()
    edges_path = os.path.join(gt_dir, f"ground_truth_edges_{step}.csv")
    if not os.path.isfile(edges_path):
        fail(f"missing {edges_path}")
    with open(edges_path, newline="", encoding="utf-8") as f:
        erows = list(csv.DictReader(f))
    if len(erows) != EXPECTED_PAIRS:
        fail(f"ground_truth_edges has {len(erows)} rows, expected {EXPECTED_PAIRS}")
    if not any(int(r.get("tie_cumulative", 0) or 0) == 1 for r in erows):
        fail("no pair with tie_cumulative==1 (burn-in likely too short)")

    # Shock log: hub + broker treatments
    if not os.path.isfile(shock):
        fail(f"missing {shock}")
    shock_entries = []
    with open(shock, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            shock_entries.append(json.loads(line))
    types = {e.get("treatment_type") for e in shock_entries}
    if "hub_removal" not in types or "broker_removal" not in types:
        fail(f"shock_log missing hub_removal or broker_removal, got types={types!r}")
    hub_e = next(e for e in shock_entries if e.get("treatment_type") == "hub_removal")
    br_e = next(e for e in shock_entries if e.get("treatment_type") == "broker_removal")
    for label, e in (("hub", hub_e), ("broker", br_e)):
        if not e.get("all_degrees") or not e.get("all_betweenness"):
            fail(f"shock entry {label!r} missing all_degrees or all_betweenness")
    hub_target = hub_e.get("target_agent")
    broker_target = br_e.get("target_agent")
    bc_vals = [float(v) for v in hub_e["all_betweenness"].values()]
    if max(bc_vals) == 0.0:
        fail("all betweenness scores are 0 (cannot distinguish broker structure)")
    # Pass if different targets OR graph could support divergence
    deg_set = set(hub_e["all_degrees"].values())
    pos_bc = sum(1 for v in bc_vals if v > 0.0)
    if hub_target == broker_target and not (len(deg_set) >= 2 and pos_bc >= 2):
        fail(
            "hub and broker selected same agent and graph shows insufficient "
            "degree/betweenness diversity for a meaningful pilot"
        )

    # analysis_micro_tie_metrics.csv (run analyze_survey.py first)
    if not os.path.isfile(metrics):
        fail(
            f"missing {metrics} — run: python analyze_survey.py {survey_dir!r}"
        )
    with open(metrics, newline="", encoding="utf-8") as f:
        mrows = list(csv.DictReader(f))
    pre_metrics = [r for r in mrows if r.get("wave_id") == "pre"]
    if len(pre_metrics) != EXPECTED_RESPONDENTS + 1:
        fail(
            f"expected {EXPECTED_RESPONDENTS + 1} pre-wave metric rows "
            f"(incl. __overall__), got {len(pre_metrics)}"
        )
    overall = next((r for r in pre_metrics if r["respondent"] == "__overall__"), None)
    if not overall:
        fail("missing __overall__ row in analysis_micro_tie_metrics.csv")
    tp = int(overall["tp"])
    fp = int(overall["fp"])
    fn = int(overall["fn"])
    tn = int(overall["tn"])
    n_pairs = int(overall["n_pairs_evaluated"])
    if tp + fp + fn + tn != n_pairs:
        fail(
            f"invariant tp+fp+fn+tn != n_pairs_evaluated: "
            f"{tp}+{fp}+{fn}+{tn} != {n_pairs}"
        )
    n_fs = int(overall["n_fail_safe"])
    denom = EXPECTED_MICRO_TIE_ROWS
    fs_rate = n_fs / denom
    if fs_rate > 0.10:
        fail(f"fail-safe rate {fs_rate:.3f} > 0.10 threshold (n_fail_safe={n_fs})")

    ok("all pre-flight checks passed — ready to plan full n=15 pilot.")
    print(f"  survey_dir={survey_dir}")
    print(f"  step={step}  hub_target={hub_target!r}  broker_target={broker_target!r}")
    print(f"  __overall__ pair_accuracy={overall.get('pair_accuracy')!r}")


if __name__ == "__main__":
    main()
