"""Pilot diagnostics and explicit missing-data-aware network perception scores."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from pilot_api import estimated_cost
from pilot_state import atomic_json


def score_respondent(metadata, answer):
    import networkx as nx
    from scipy.stats import kendalltau
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    roster = metadata["roster"]
    graph = nx.Graph()
    graph.add_nodes_from(roster)
    graph.add_weighted_edges_from(metadata["truth_edges"])
    row = {"step": metadata["step"], "repeat": metadata["repeat"],
           "respondent": answer["respondent"], "initial": metadata["initial"],
           "brier": None, "degree_tau_b": None, "broker_tau_b": None,
           "community_ari": None, "community_nmi": None,
           "interaction_missing": sum(p is None for _, _, p in answer["interaction"]),
           "invalid_attempts": sum(len(v) for v in answer["invalid_attempts"].values()),
           "truth_edges": graph.number_of_edges()}
    if metadata["initial"]:
        return row
    if row["interaction_missing"] == 0:
        row["brier"] = statistics.mean((p - int(graph.has_edge(a, b)))**2
                                       for a, b, p in answer["interaction"])
    for instrument, truth in (("degree", dict(graph.degree())),
                              ("broker", nx.betweenness_centrality(graph, weight=None))):
        prediction = answer[instrument]
        # tau is undefined for a constant true ranking; do not coerce to zero.
        if prediction is not None and len(set(truth.values())) > 1:
            ranks = {name: -rank for rank, name in enumerate(prediction)}
            statistic = float(kendalltau([truth[n] for n in roster],
                                        [ranks[n] for n in roster], variant="b")[0])
            import math
            row[instrument + "_tau_b"] = statistic if math.isfinite(statistic) else None
    if answer["community"] is not None and graph.number_of_edges():
        communities = nx.community.louvain_communities(graph, weight="weight", seed=42)
        true_labels = {name: i for i, group in enumerate(communities) for name in group}
        labels = {name: i for i, group in enumerate(answer["community"]) for name in group}
        row["community_ari"] = float(adjusted_rand_score(
            [true_labels[n] for n in roster], [labels[n] for n in roster]))
        row["community_nmi"] = float(normalized_mutual_info_score(
            [true_labels[n] for n in roster], [labels[n] for n in roster]))
    return row


def write_report(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows, complete_waves = [], []
    for marker in sorted((run_dir / "surveys").glob("step_*/repeat_*/complete.json")):
        directory = marker.parent
        metadata = json.loads((directory / "metadata.json").read_text())
        complete_waves.append({"step": metadata["step"], "repeat": metadata["repeat"]})
        for file in sorted(directory.glob("respondent_*.json")):
            rows.append(score_respondent(metadata, json.loads(file.read_text())))
    if rows:
        with (run_dir / "perception_scores.csv").open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    metrics = ("brier", "degree_tau_b", "broker_tau_b", "community_ari")
    per_wave = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for metric in metrics:
            if row[metric] is not None:
                per_wave[(row["step"], row["repeat"])][metric].append(row[metric])
    repeated = defaultdict(list)
    for (step, repeat), values in per_wave.items():
        for metric, scores in values.items():
            repeated[(step, metric)].append(statistics.mean(scores))
    noise = [{"step": step, "metric": metric, "repeats": len(values),
              "mean": statistics.mean(values),
              "repeat_sd": statistics.stdev(values) if len(values) > 1 else None}
             for (step, metric), values in sorted(repeated.items())]
    costs = defaultdict(lambda: {"requests": 0, "errors": 0, "missing_usage": 0,
                                 "input_tokens": 0, "output_tokens": 0,
                                 "estimated_usd": 0.0})
    usage_path = run_dir / "usage.jsonl"
    if usage_path.exists():
        for line in usage_path.read_text().splitlines():
            record = json.loads(line)
            stamp = record.get("sim_time")
            hour = int(stamp[11:13]) if stamp else None
            period = "unassigned" if hour is None else "day" if 6 <= hour < 20 else "night"
            bucket = costs[record["phase"] + "/" + period]
            bucket["requests"] += 1
            bucket["errors"] += record["status"] == "error"
            usage = record.get("usage") or {}
            bucket["input_tokens"] += usage.get("prompt_tokens", usage.get("input_tokens", 0))
            bucket["output_tokens"] += usage.get("completion_tokens", usage.get("output_tokens", 0))
            estimate = estimated_cost(record)
            bucket["missing_usage"] += estimate is None
            bucket["estimated_usd"] += estimate or 0
    report = {"status": manifest["status"], "checkpoint_status": manifest["checkpoint_status"],
              "selected_checkpoint": manifest.get("selected_checkpoint"),
              "completed_survey_waves": complete_waves,
              "measurement_noise": noise, "costs": dict(costs),
              "cost_day_definition": "06:00 <= hour < 20:00; accounting only, not a forced sleep schedule",
              "cost_basis": "Public standard token-price estimate; provider invoice, regional premiums and failed-request billing may differ",
              "inference": "One exploratory control trajectory; role-separation eligibility is conditional, not a population-wide effect."}
    atomic_json(run_dir / "report.json", report)
    text = ["# Midnight pilot report", "", f"Status: {manifest['status']}",
            f"Checkpoint: {manifest['checkpoint_status']}", "",
            "Start: Day 1 00:00. End: Day 3 16:00 (64 simulated hours).",
            "", "## Measurement and interpretation", "",
            "Four instruments use the preceding two hours. Initial-state scores are missing by design. "
            "Undefined rank correlations and failed responses remain missing. Community truth is weighted Louvain (seed 42).",
            "Repeated measurements estimate survey noise, not between-trajectory uncertainty.", "",
            f"Completed wave/repeat combinations: {len(complete_waves)}.", "",
            "See perception_scores.csv, report.json, checkpoint_candidates.json and network_history.json for numerical results.",
            "", "## Cost", "", "Day/night bins are 06:00–20:00 / 20:00–06:00 for accounting only.",
            "| Phase / period | Calls | Input tokens | Output tokens | Estimated USD | Missing usage |",
            "|---|---:|---:|---:|---:|---:|"]
    for key, value in sorted(costs.items()):
        text.append(f"| {key} | {value['requests']} | {value['input_tokens']} | {value['output_tokens']} | {value['estimated_usd']:.4f} | {value['missing_usage']} |")
    text.extend(["", "Estimates are not an invoice. No trajectory is discarded when the checkpoint window has no eligible state.", ""])
    (run_dir / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
    return report
