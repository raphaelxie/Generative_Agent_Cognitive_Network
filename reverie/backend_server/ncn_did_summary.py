#!/usr/bin/env python3
"""
Difference-in-differences summary across NCN experiment branches.

Reads pre/post metrics from control, hub, and broker branch survey dirs
and writes ncn_did_summary.csv.

Usage (from reverie/backend_server/):
  python ncn_did_summary.py
"""
import csv
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FS_STORAGE = os.path.join(SCRIPT_DIR, "../../environment/frontend_server/storage")

BRANCHES = {
    "control": "ncn_control_n25",
    "hub": "ncn_hub_n25",
    "broker": "ncn_broker_n25",
}
BASELINE = "ncn_baseline_n25"

if os.environ.get("NCN_SUFFIX"):
    suffix = os.environ["NCN_SUFFIX"]
    BASELINE = f"ncn_baseline_n25{suffix}"
    BRANCHES = {
        "control": f"ncn_control_n25{suffix}",
        "hub": f"ncn_hub_n25{suffix}",
        "broker": f"ncn_broker_n25{suffix}",
    }

METRIC_SOURCES = [
    ("micro_tie", "analysis_micro_tie_metrics_by_truth.csv",
     "pair_accuracy", "truth_layer"),
    ("bridge_rank", "analysis_bridge_rank_metrics.csv",
     "bridge_mean_abs_rank_error", "truth_definition"),
    ("bridge_top1", "analysis_bridge_rank_metrics.csv",
     "bridge_top1_hit", "truth_definition"),
    ("community_nmi", "analysis_community_group_metrics.csv",
     "nmi", "truth_definition"),
]


def _load_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _overall_val(rows, metric_col, filter_col=None, filter_val=None,
                 wave_id="pre"):
    for r in rows:
        if r.get("respondent") != "__overall__":
            continue
        if r.get("wave_id") != wave_id:
            continue
        if filter_col and filter_val and r.get(filter_col) != filter_val:
            continue
        try:
            return float(r[metric_col])
        except (KeyError, ValueError, TypeError):
            return None
    return None


def _delta(pre, post):
    if pre is None or post is None:
        return None
    return post - pre


def _did(delta_treat, delta_ctrl):
    if delta_treat is None or delta_ctrl is None:
        return None
    return delta_treat - delta_ctrl


def _shocked_agent_rank(survey_dir, wave_id, shocked_agent):
    """Mean perceived bridge rank of the shocked agent in a wave."""
    path = os.path.join(survey_dir, f"perception_survey_{wave_id}.csv")
    if not os.path.isfile(path) or not shocked_agent:
        return None
    ranks = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("question_type") == "bridge_rank"
                    and row.get("target") == shocked_agent
                    and row.get("is_fail_safe") != "1"):
                try:
                    ranks.append(int(row["value"]))
                except ValueError:
                    pass
    return sum(ranks) / len(ranks) if ranks else None


def main():
    baseline_dir = os.path.join(FS_STORAGE, BASELINE, "survey")
    targets_path = os.path.join(baseline_dir, "shock_targets.json")
    shocked_hub = shocked_broker = None
    if os.path.isfile(targets_path):
        with open(targets_path, encoding="utf-8") as f:
            targets = json.load(f)
        shocked_hub = targets.get("hub_agent")
        shocked_broker = targets.get("broker_agent")

    records = []

    for label, filename, metric_col, filter_col in METRIC_SOURCES:
        for filter_val in (
            "observed_interaction",
            "observed_interaction_recent",
            "observed_interaction_cumulative_betweenness_rank",
            "observed_interaction_recent_betweenness_rank",
            "observed_interaction_louvain_communities",
            "observed_interaction_recent_louvain_communities",
        ):
            ctrl_dir = os.path.join(FS_STORAGE, BRANCHES["control"], "survey")
            hub_dir = os.path.join(FS_STORAGE, BRANCHES["hub"], "survey")
            broker_dir = os.path.join(FS_STORAGE, BRANCHES["broker"], "survey")

            ctrl_rows = _load_csv(os.path.join(ctrl_dir, filename))
            hub_rows = _load_csv(os.path.join(hub_dir, filename))
            broker_rows = _load_csv(os.path.join(broker_dir, filename))

            if not ctrl_rows and not hub_rows and not broker_rows:
                continue

            ctrl_pre = _overall_val(ctrl_rows, metric_col, filter_col,
                                    filter_val, "pre")
            ctrl_post = _overall_val(ctrl_rows, metric_col, filter_col,
                                     filter_val, "post")
            hub_pre = _overall_val(hub_rows, metric_col, filter_col,
                                   filter_val, "pre")
            hub_post = _overall_val(hub_rows, metric_col, filter_col,
                                    filter_val, "post")
            broker_pre = _overall_val(broker_rows, metric_col, filter_col,
                                      filter_val, "pre")
            broker_post = _overall_val(broker_rows, metric_col, filter_col,
                                       filter_val, "post")

            # Branches inherit baseline pre; use baseline pre if branch pre missing
            if hub_pre is None:
                hub_pre = _overall_val(
                    _load_csv(os.path.join(baseline_dir, filename)),
                    metric_col, filter_col, filter_val, "pre")
            if broker_pre is None:
                broker_pre = hub_pre

            d_ctrl = _delta(ctrl_pre, ctrl_post)
            d_hub = _delta(hub_pre, hub_post)
            d_broker = _delta(broker_pre, broker_post)

            records.append({
                "instrument": label,
                "truth_layer": filter_val,
                "metric": metric_col,
                "control_pre": ctrl_pre,
                "control_post": ctrl_post,
                "control_delta": d_ctrl,
                "hub_pre": hub_pre,
                "hub_post": hub_post,
                "hub_delta": d_hub,
                "hub_did": _did(d_hub, d_ctrl),
                "broker_pre": broker_pre,
                "broker_post": broker_post,
                "broker_delta": d_broker,
                "broker_did": _did(d_broker, d_ctrl),
            })

    # Targeted: perceived rank of shocked agents
    for treatment, agent, branch_key in (
        ("hub", shocked_hub, "hub"),
        ("broker", shocked_broker, "broker"),
    ):
        if not agent:
            continue
        ctrl_dir = os.path.join(FS_STORAGE, BRANCHES["control"], "survey")
        treat_dir = os.path.join(FS_STORAGE, BRANCHES[branch_key], "survey")
        pre = _shocked_agent_rank(baseline_dir, "pre", agent)
        ctrl_post = _shocked_agent_rank(ctrl_dir, "post", agent)
        treat_post = _shocked_agent_rank(treat_dir, "post", agent)
        records.append({
            "instrument": f"shocked_agent_rank_{treatment}",
            "truth_layer": agent,
            "metric": "mean_perceived_bridge_rank",
            "control_pre": pre,
            "control_post": ctrl_post,
            "control_delta": _delta(pre, ctrl_post),
            "hub_pre": pre if treatment == "hub" else None,
            "hub_post": treat_post if treatment == "hub" else None,
            "hub_delta": _delta(pre, treat_post) if treatment == "hub" else None,
            "hub_did": None,
            "broker_pre": pre if treatment == "broker" else None,
            "broker_post": treat_post if treatment == "broker" else None,
            "broker_delta": (
                _delta(pre, treat_post) if treatment == "broker" else None),
            "broker_did": None,
        })

    if not records:
        print("No branch metrics found. Run the experiment first.")
        sys.exit(1)

    out_path = os.path.join(baseline_dir, "ncn_did_summary.csv")
    fieldnames = list(records[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)

    print(f"Wrote {out_path} ({len(records)} rows)")

    md_path = os.path.join(baseline_dir, "NCN_EXPERIMENT.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Network Cognition Experiment (n=25)\n\n")
        f.write("## Design\n\n")
        f.write("- Between-branch difference-in-differences\n")
        f.write("- Treatments: control (no shock), hub removal, broker removal\n")
        f.write(f"- Model: gpt-4o-mini (locked)\n\n")
        if os.path.isfile(targets_path):
            with open(targets_path, encoding="utf-8") as tf:
                f.write(f"## Shock targets\n\n```json\n{tf.read()}```\n\n")
        f.write("## DiD summary\n\n")
        f.write(f"See `{os.path.basename(out_path)}`.\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
