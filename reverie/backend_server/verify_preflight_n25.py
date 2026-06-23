#!/usr/bin/env python3
"""
Verify n=25 pre-flight artifacts and three ground-truth layers.

Usage:
  python verify_preflight_n25.py <survey_dir>

Example:
  python verify_preflight_n25.py ../../environment/frontend_server/storage/preflight_the_ville_n25-1/survey
"""
import csv
import json
import os
import sys

EXPECTED_RESPONDENTS = 25
EXPECTED_PAIRS = EXPECTED_RESPONDENTS * (EXPECTED_RESPONDENTS - 1) // 2  # 300
# 4 micro-tie instruments x n focal retrievals + 1 NCN social retrieval per respondent
EXPECTED_RETRIEVAL_LINES = EXPECTED_RESPONDENTS * (4 * EXPECTED_RESPONDENTS + 1)
EXPECTED_MICRO_TIE_ROWS_PER_INSTRUMENT = (
    EXPECTED_RESPONDENTS * EXPECTED_RESPONDENTS * (EXPECTED_RESPONDENTS - 1)
)


def fail(msg):
    print(f"NOT READY: {msg}")
    sys.exit(1)


def ok(msg):
    print(f"PASS: {msg}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_preflight_n25.py <survey_dir> [--layers-only]")
        sys.exit(2)
    args = [a for a in sys.argv[1:] if a.startswith("-")]
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    layers_only = "--layers-only" in args
    survey_dir = os.path.abspath(positional[0].strip())
    if not os.path.isdir(survey_dir):
        fail(f"not a directory: {survey_dir}")

    pre_csv = os.path.join(survey_dir, "perception_survey_pre.csv")
    pre_meta = os.path.join(survey_dir, "perception_survey_pre_meta.json")
    diag = os.path.join(survey_dir, "retrieval_diagnostics_pre.jsonl")
    shock = os.path.join(survey_dir, "shock_log.jsonl")
    metrics = os.path.join(survey_dir, "analysis_micro_tie_metrics.csv")
    metrics_by_truth = os.path.join(
        survey_dir, "analysis_micro_tie_metrics_by_truth.csv")
    metrics_by_construct = os.path.join(
        survey_dir, "analysis_micro_tie_by_construct.csv")
    gt_dir = os.path.join(survey_dir, "ground_truth")
    background_path = os.path.join(gt_dir, "background_social_edges.csv")

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
            f"(25*(4*25+1) per full-instrument design)"
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

    step = rows[0].get("step", "").strip()
    edges_path = os.path.join(gt_dir, f"ground_truth_edges_{step}.csv")
    if not os.path.isfile(edges_path):
        fail(f"missing Layer 1 file: {edges_path}")
    with open(edges_path, newline="", encoding="utf-8") as f:
        erows = list(csv.DictReader(f))
    if len(erows) != EXPECTED_PAIRS:
        fail(f"ground_truth_edges has {len(erows)} rows, expected {EXPECTED_PAIRS}")
    if not any(int(r.get("tie_cumulative", 0) or 0) == 1 for r in erows):
        if layers_only:
            print("  (no observed chats yet — expected before burn-in completes)")
        else:
            fail("no pair with tie_cumulative==1 (burn-in likely too short)")

    if not os.path.isfile(background_path):
        fail(f"missing Layer 2 file: {background_path}")
    with open(background_path, newline="", encoding="utf-8") as f:
        bg_rows = list(csv.DictReader(f))
    if len(bg_rows) != EXPECTED_PAIRS:
        fail(f"background_social_edges has {len(bg_rows)} rows, expected {EXPECTED_PAIRS}")
    if not any(int(r.get("tie_background", 0) or 0) == 1 for r in bg_rows):
        fail("no pair with tie_background==1 (background truth extraction failed)")

    if not os.path.isfile(metrics_by_truth):
        fail(
            f"missing {metrics_by_truth} — run: python analyze_survey.py {survey_dir!r}"
        )
    with open(metrics_by_truth, newline="", encoding="utf-8") as f:
        truth_rows = list(csv.DictReader(f))
    truth_layers = {r.get("truth_layer") for r in truth_rows if r.get("wave_id") == "pre"}
    expected_layers = {
        "observed_interaction",
        "background_social_tie",
        "background_or_interaction",
    }
    missing_layers = expected_layers - truth_layers
    if missing_layers:
        fail(f"analysis_micro_tie_metrics_by_truth missing layers: {sorted(missing_layers)}")

    if not os.path.isfile(metrics_by_construct):
        fail(f"missing {metrics_by_construct}")

    if not os.path.isfile(metrics):
        fail(f"missing {metrics}")

    if not os.path.isfile(shock):
        print(f"  (no shock_log.jsonl — optional for ground-truth-only setup)")
    else:
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

    ok("all n=25 pre-flight checks passed — three ground truth layers ready.")
    print(f"  survey_dir={survey_dir}")
    print(f"  step={step}")
    print(f"  Layer 1 (observed_interaction): {edges_path}")
    print(f"  Layer 2 (background_social_tie): {background_path}")
    print(f"  Layer 3 (background_or_interaction): derived in {metrics_by_truth}")
    n_pos_bg = sum(int(r.get("tie_background", 0) or 0) for r in bg_rows)
    n_pos_obs = sum(int(r.get("tie_cumulative", 0) or 0) for r in erows)
    print(f"  positive background ties: {n_pos_bg}/{EXPECTED_PAIRS}")
    print(f"  positive observed ties: {n_pos_obs}/{EXPECTED_PAIRS}")


if __name__ == "__main__":
    main()
