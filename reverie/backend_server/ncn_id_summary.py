#!/usr/bin/env python3
"""
Identification experiment summary: noise band + replicated DiD.

Reads analysis CSVs from the 6 identification forks and the baseline
noise-floor reps (noise_r01..noise_r05 @ step 2100), then writes:
  - ncn_id_noise_band.csv
  - ncn_id_replicate_deltas.csv
  - ncn_id_did_summary.csv
  - NCN_IDENTIFICATION.md

Usage (from reverie/backend_server/):
  python ncn_id_summary.py
"""
import csv
import json
import math
import os
import statistics
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FS_STORAGE = os.path.join(SCRIPT_DIR, "../../environment/frontend_server/storage")
BASELINE = "ncn_baseline_n25_full"
PRE_WAVE = "pre"
POST_WAVE = "post"

FORKS = [
    ("control", 1, "ncn_id_control_r1"),
    ("control", 2, "ncn_id_control_r2"),
    ("hub", 1, "ncn_id_hub_r1"),
    ("hub", 2, "ncn_id_hub_r2"),
    ("broker", 1, "ncn_id_broker_r1"),
    ("broker", 2, "ncn_id_broker_r2"),
]

TRUTH_LAYERS = (
    "observed_interaction",
    "observed_interaction_recent",
    "observed_interaction_cumulative_betweenness_rank",
    "observed_interaction_recent_betweenness_rank",
    "observed_interaction_louvain_communities",
    "observed_interaction_recent_louvain_communities",
)

METRIC_SOURCES = [
    ("micro_tie", "analysis_micro_tie_metrics_by_truth.csv",
     "pair_accuracy", "truth_layer"),
    ("micro_tie_fpr", "analysis_micro_tie_metrics_by_truth.csv",
     "fpr", "truth_layer"),
    ("micro_tie_fnr", "analysis_micro_tie_metrics_by_truth.csv",
     "fnr", "truth_layer"),
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


def _overall_val(rows, metric_col, filter_col, filter_val, wave_id):
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


def _noise_waves(manifest):
    waves = manifest.get("noise_floor", {}).get("completed", [])
    return sorted(w for w in waves if w.startswith("noise_r"))


def _noise_stats(values):
    if not values:
        return None, None, None, None
    if len(values) == 1:
        return values[0], 0.0, values[0], values[0]
    return (
        statistics.mean(values),
        statistics.stdev(values),
        min(values),
        max(values),
    )


def _shocked_agent_rank(survey_dir, wave_id, agent):
    path = os.path.join(survey_dir, f"perception_survey_{wave_id}.csv")
    if not os.path.isfile(path) or not agent:
        return None
    ranks = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("question_type") == "bridge_rank"
                    and row.get("target") == agent
                    and row.get("is_fail_safe") != "1"):
                try:
                    ranks.append(int(row["value"]))
                except ValueError:
                    pass
    return sum(ranks) / len(ranks) if ranks else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def main():
    baseline_dir = os.path.join(FS_STORAGE, BASELINE, "survey")
    manifest_path = os.path.join(baseline_dir, "ncn_id_manifest.json")
    if not os.path.isfile(manifest_path):
        print("Missing ncn_id_manifest.json — run identification experiment first.")
        sys.exit(1)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    noise_waves = _noise_waves(manifest)
    hub_agent = manifest.get("hub_agent")
    broker_agent = manifest.get("broker_agent")

    noise_rows = []
    delta_rows = []
    did_rows = []

    # ── metric-level noise band + replicate deltas ─────────────────────
    for label, filename, metric_col, filter_col in METRIC_SOURCES:
        baseline_rows = _load_csv(os.path.join(baseline_dir, filename))
        for truth in TRUTH_LAYERS:
            pre_val = _overall_val(
                baseline_rows, metric_col, filter_col, truth, PRE_WAVE)

            noise_vals = []
            for wave in noise_waves:
                v = _overall_val(
                    baseline_rows, metric_col, filter_col, truth, wave)
                if v is not None:
                    noise_vals.append(v)

            n_mean, n_std, n_min, n_max = _noise_stats(noise_vals)
            noise_rows.append({
                "instrument": label,
                "truth_layer": truth,
                "metric": metric_col,
                "pre_value": pre_val,
                "noise_n": len(noise_vals),
                "noise_mean": n_mean,
                "noise_std": n_std,
                "noise_min": n_min,
                "noise_max": n_max,
                "noise_range": (n_max - n_min) if n_min is not None else None,
            })

            by_treatment = {"control": {}, "hub": {}, "broker": {}}
            for treatment, rep, code in FORKS:
                rows = _load_csv(
                    os.path.join(FS_STORAGE, code, "survey", filename))
                if not rows:
                    rows = baseline_rows
                post_val = _overall_val(
                    rows, metric_col, filter_col, truth, POST_WAVE)
                fork_pre = _overall_val(
                    rows, metric_col, filter_col, truth, PRE_WAVE)
                if fork_pre is None:
                    fork_pre = pre_val
                delta = (post_val - fork_pre) if (
                    post_val is not None and fork_pre is not None) else None
                delta_rows.append({
                    "fork": code,
                    "treatment": treatment,
                    "replicate": rep,
                    "instrument": label,
                    "truth_layer": truth,
                    "metric": metric_col,
                    "pre": fork_pre,
                    "post": post_val,
                    "delta": delta,
                })
                by_treatment[treatment][rep] = delta

            ctrl_deltas = [v for v in by_treatment["control"].values()
                           if v is not None]
            hub_deltas = [v for v in by_treatment["hub"].values()
                          if v is not None]
            broker_deltas = [v for v in by_treatment["broker"].values()
                             if v is not None]

            ctrl_mean = _mean(ctrl_deltas)
            hub_mean = _mean(hub_deltas)
            broker_mean = _mean(broker_deltas)
            hub_did = (hub_mean - ctrl_mean) if (
                hub_mean is not None and ctrl_mean is not None) else None
            broker_did = (broker_mean - ctrl_mean) if (
                broker_mean is not None and ctrl_mean is not None) else None

            paired_hub = []
            paired_broker = []
            for rep in (1, 2):
                cd = by_treatment["control"].get(rep)
                hd = by_treatment["hub"].get(rep)
                bd = by_treatment["broker"].get(rep)
                if cd is not None and hd is not None:
                    paired_hub.append(hd - cd)
                if cd is not None and bd is not None:
                    paired_broker.append(bd - cd)

            threshold = (2.0 * n_std) if n_std is not None else None

            did_rows.append({
                "instrument": label,
                "truth_layer": truth,
                "metric": metric_col,
                "control_delta_mean": ctrl_mean,
                "control_delta_r1": by_treatment["control"].get(1),
                "control_delta_r2": by_treatment["control"].get(2),
                "hub_delta_mean": hub_mean,
                "hub_delta_r1": by_treatment["hub"].get(1),
                "hub_delta_r2": by_treatment["hub"].get(2),
                "hub_did_mean": hub_did,
                "hub_did_r1": paired_hub[0] if len(paired_hub) > 0 else None,
                "hub_did_r2": paired_hub[1] if len(paired_hub) > 1 else None,
                "broker_delta_mean": broker_mean,
                "broker_delta_r1": by_treatment["broker"].get(1),
                "broker_delta_r2": by_treatment["broker"].get(2),
                "broker_did_mean": broker_did,
                "broker_did_r1": (
                    paired_broker[0] if len(paired_broker) > 0 else None),
                "broker_did_r2": (
                    paired_broker[1] if len(paired_broker) > 1 else None),
                "noise_std": n_std,
                "noise_range": (n_max - n_min) if n_min is not None else None,
                "hub_did_exceeds_2x_noise": (
                    int(abs(hub_did) > threshold)
                    if hub_did is not None and threshold else None),
                "broker_did_exceeds_2x_noise": (
                    int(abs(broker_did) > threshold)
                    if broker_did is not None and threshold else None),
                "control_delta_exceeds_2x_noise": (
                    int(abs(ctrl_mean) > threshold)
                    if ctrl_mean is not None and threshold else None),
            })

    # ── shocked-agent perceived bridge rank ────────────────────────────
    for treatment, agent in (("hub", hub_agent), ("broker", broker_agent)):
        if not agent:
            continue
        pre_rank = _shocked_agent_rank(baseline_dir, PRE_WAVE, agent)
        noise_vals = [
            _shocked_agent_rank(baseline_dir, w, agent) for w in noise_waves]
        noise_vals = [v for v in noise_vals if v is not None]
        n_mean, n_std, n_min, n_max = _noise_stats(noise_vals)
        noise_rows.append({
            "instrument": f"shocked_agent_rank_{treatment}",
            "truth_layer": agent,
            "metric": "mean_perceived_bridge_rank",
            "pre_value": pre_rank,
            "noise_n": len(noise_vals),
            "noise_mean": n_mean,
            "noise_std": n_std,
            "noise_min": n_min,
            "noise_max": n_max,
            "noise_range": (n_max - n_min) if n_min is not None else None,
        })

        by_treatment = {"control": {}, "hub": {}, "broker": {}}
        for trt, rep, code in FORKS:
            survey_dir = os.path.join(FS_STORAGE, code, "survey")
            post_rank = _shocked_agent_rank(survey_dir, POST_WAVE, agent)
            fork_pre = pre_rank
            delta = (post_rank - fork_pre) if (
                post_rank is not None and fork_pre is not None) else None
            delta_rows.append({
                "fork": code,
                "treatment": trt,
                "replicate": rep,
                "instrument": f"shocked_agent_rank_{treatment}",
                "truth_layer": agent,
                "metric": "mean_perceived_bridge_rank",
                "pre": fork_pre,
                "post": post_rank,
                "delta": delta,
            })
            if trt == treatment or trt == "control":
                by_treatment[trt][rep] = delta

        ctrl_mean = _mean([v for v in by_treatment["control"].values()
                           if v is not None])
        treat_mean = _mean([v for v in by_treatment[treatment].values()
                            if v is not None])
        did_val = (treat_mean - ctrl_mean) if (
            treat_mean is not None and ctrl_mean is not None) else None
        threshold = (2.0 * n_std) if n_std is not None else None

        paired = []
        for rep in (1, 2):
            cd = by_treatment["control"].get(rep)
            td = by_treatment[treatment].get(rep)
            if cd is not None and td is not None:
                paired.append(td - cd)

        did_rows.append({
            "instrument": f"shocked_agent_rank_{treatment}",
            "truth_layer": agent,
            "metric": "mean_perceived_bridge_rank",
            "control_delta_mean": ctrl_mean,
            "control_delta_r1": by_treatment["control"].get(1),
            "control_delta_r2": by_treatment["control"].get(2),
            "hub_delta_mean": treat_mean if treatment == "hub" else None,
            "hub_delta_r1": (
                by_treatment["hub"].get(1) if treatment == "hub" else None),
            "hub_delta_r2": (
                by_treatment["hub"].get(2) if treatment == "hub" else None),
            "hub_did_mean": did_val if treatment == "hub" else None,
            "hub_did_r1": paired[0] if treatment == "hub" and paired else None,
            "hub_did_r2": (
                paired[1] if treatment == "hub" and len(paired) > 1 else None),
            "broker_delta_mean": treat_mean if treatment == "broker" else None,
            "broker_delta_r1": (
                by_treatment["broker"].get(1) if treatment == "broker"
                else None),
            "broker_delta_r2": (
                by_treatment["broker"].get(2) if treatment == "broker"
                else None),
            "broker_did_mean": did_val if treatment == "broker" else None,
            "broker_did_r1": (
                paired[0] if treatment == "broker" and paired else None),
            "broker_did_r2": (
                paired[1] if treatment == "broker" and len(paired) > 1
                else None),
            "noise_std": n_std,
            "noise_range": (n_max - n_min) if n_min is not None else None,
            "hub_did_exceeds_2x_noise": (
                int(abs(did_val) > threshold)
                if treatment == "hub" and did_val is not None and threshold
                else None),
            "broker_did_exceeds_2x_noise": (
                int(abs(did_val) > threshold)
                if treatment == "broker" and did_val is not None and threshold
                else None),
            "control_delta_exceeds_2x_noise": (
                int(abs(ctrl_mean) > threshold)
                if ctrl_mean is not None and threshold else None),
        })

    def _write_csv(name, rows):
        if not rows:
            return None
        path = os.path.join(baseline_dir, name)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {path} ({len(rows)} rows)")
        return path

    _write_csv("ncn_id_noise_band.csv", noise_rows)
    _write_csv("ncn_id_replicate_deltas.csv", delta_rows)
    did_path = _write_csv("ncn_id_did_summary.csv", did_rows)

    # ── markdown summary ───────────────────────────────────────────────
    md_path = os.path.join(baseline_dir, "NCN_IDENTIFICATION.md")
    primary_truth = "observed_interaction_recent"
    highlights = [
        r for r in did_rows
        if r["truth_layer"] in (
            primary_truth,
            "observed_interaction_recent_betweenness_rank",
            "observed_interaction_recent_louvain_communities",
        )
        and r["instrument"] in (
            "micro_tie", "bridge_rank", "community_nmi",
            "shocked_agent_rank_hub", "shocked_agent_rank_broker",
        )
    ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# NCN Identification Experiment (n=25, R=2 per arm)\n\n")
        f.write("## Design\n\n")
        f.write("- Pre wave: shared `pre` @ step 1800 (inherited from baseline fork)\n")
        f.write("- Post wave: `post` @ step 3300 (+1200 steps post-shock)\n")
        f.write(f"- Noise floor: {len(noise_waves)} reps @ frozen step 2100 "
                f"(`{', '.join(noise_waves)}`)\n")
        f.write("- Replicates: 2 independent forks per arm (control / hub / broker)\n")
        f.write(f"- Hub target: {hub_agent}; Broker target: {broker_agent}\n")
        f.write(f"- Primary windowed truth: `{primary_truth}` "
                f"(200 min recent window)\n\n")

        f.write("## Noise band (measurement stochasticity @ step 2100)\n\n")
        f.write("See `ncn_id_noise_band.csv`. "
                "Compare treatment DiD against `2 × noise_std`.\n\n")

        f.write("## Replicated DiD (primary outcomes)\n\n")
        f.write("| Instrument | Truth | Control Δ | Hub DiD | Broker DiD | "
                "2×noise_std | Hub>noise? | Broker>noise? |\n")
        f.write("|---|---|---:|---:|---:|---:|---|---|\n")
        for r in highlights:
            thresh = (2 * r["noise_std"]) if r["noise_std"] is not None else None
            f.write(
                f"| {r['instrument']} | {r['truth_layer']} | "
                f"{_fmt(r['control_delta_mean'])} | "
                f"{_fmt(r['hub_did_mean'])} | "
                f"{_fmt(r['broker_did_mean'])} | "
                f"{_fmt(thresh)} | "
                f"{_flag(r.get('hub_did_exceeds_2x_noise'))} | "
                f"{_flag(r.get('broker_did_exceeds_2x_noise'))} |\n")

        f.write("\n## Files\n\n")
        f.write("- `ncn_id_noise_band.csv` — spread across noise reps\n")
        f.write("- `ncn_id_replicate_deltas.csv` — per-fork pre/post deltas\n")
        f.write("- `ncn_id_did_summary.csv` — full replicated DiD table\n")
        f.write("- `ncn_id_manifest.json` — run manifest\n")

    print(f"Wrote {md_path}")


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return f"{v:.4f}"


def _flag(v):
    if v is None:
        return "?"
    return "yes" if v else "no"


if __name__ == "__main__":
    main()
