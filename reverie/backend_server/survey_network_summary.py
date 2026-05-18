"""
Cross-checkpoint network diagnostic for a simulation's survey waves.

Reads the existing ground_truth_edges_{step}.csv and ground_truth_chats_{step}.csv
files under <survey_dir>/ground_truth/, and computes per-checkpoint network
metrics. Output: <survey_dir>/network_summary_over_time.csv (one row per
checkpoint, sorted by step).

The script makes ZERO LLM calls and does NOT load ReverieServer; it only
parses the on-disk ground-truth CSVs that ground_truth_log.py already wrote.
It reuses the betweenness/degree helpers from ground_truth_log so the
ranking semantics match exactly what `survey` produces at runtime.

Per-checkpoint columns:
  step, sim_time
  n_personas
  n_chats_cumulative, n_new_chats_since_prev
  n_active_agents_cumulative, n_new_active_agents_since_prev
  n_edges_cumulative, n_edges_recent_8h
  density_cumulative, mean_degree_cumulative
  max_degree, top3_degree
  max_betweenness, top3_betweenness
  hub, broker, hub_eq_broker
  cumulative_* graph-shape diagnostics
  *_recent_8h broker-sensitive graph diagnostics
  transitivity, avg_clustering
  modularity_louvain, n_communities_louvain
  largest_component_diameter, largest_component_avg_path_length

Also writes per-step community assignments:
  ground_truth/ground_truth_communities_{step}.csv
  (columns: step, agent_name, community_id)

Where:
  - "Active agent" = an agent that appears as a participant in at least one
    chat row in the cumulative chat log.
  - "_since_prev" deltas are computed against the immediately previous
    checkpoint in the sorted list (the t=earliest checkpoint reports its
    own cumulative count as the delta).
  - hub_eq_broker is 1 iff the highest-degree agent and highest-betweenness
    agent (by ground_truth_log conventions, alphabetical tiebreak) coincide.
    A value of 1 means a broker probe at this checkpoint cannot be
    distinguished from a hub probe.

Usage (from reverie/backend_server/ or project root):
  python survey_network_summary.py <survey_dir>

Example:
  python survey_network_summary.py \\
    ../../environment/frontend_server/storage/prepost_the_ville_n15_brokerprobe-1/survey
"""
import csv
import json
import os
import re
import sys
from collections import Counter

import networkx as nx

_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from ground_truth_log import (
    agent_betweenness,
    graph_scope_summary,
    highest_betweenness_agent,
    highest_degree_agent,
)


# ── CSV loaders ────────────────────────────────────────────────────────

def _read_edges_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["count_cumulative"] = int(row["count_cumulative"])
            row["count_recent"] = int(row["count_recent"])
            row["tie_cumulative"] = int(row["tie_cumulative"])
            row["tie_recent"] = int(row["tie_recent"])
            rows.append(row)
    return rows


def _read_chats_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── per-checkpoint metric helpers ──────────────────────────────────────

def _degree_dict(edge_rows):
    deg = Counter()
    for row in edge_rows:
        if row["count_cumulative"] > 0:
            deg[row["node_a"]] += 1
            deg[row["node_b"]] += 1
    return deg


def _top3(score_dict, fmt):
    items = sorted(
        ((k, v) for k, v in score_dict.items() if v > 0),
        key=lambda kv: (-kv[1], kv[0]),
    )[:3]
    return "; ".join(fmt.format(k, v) for k, v in items)


def _round_or_none(value, digits=4):
    if value is None:
        return None
    return round(value, digits)


def _build_nx_graph(edge_rows, count_field="count_cumulative"):
    """Build an undirected weighted networkx Graph from edge_rows.

    Edges are weighted by the count_field value (floored to 1 minimum),
    matching the convention in analyze_survey.compute_gt_communities.
    """
    G = nx.Graph()
    for row in edge_rows:
        G.add_node(row["node_a"])
        G.add_node(row["node_b"])
        w = int(row[count_field])
        if w > 0:
            G.add_edge(row["node_a"], row["node_b"], weight=max(1, w))
    return G


def _community_metrics(G):
    """Compute clustering, modularity, and largest-component path metrics.

    Returns a dict of new column values plus the list of Louvain communities
    (for writing the per-step community CSV).

    n_components and largest_component_size are intentionally NOT returned
    here because graph_scope_summary already provides them as
    cumulative_component_count and cumulative_largest_component_size.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if m == 0:
        return {
            "transitivity": 0.0,
            "avg_clustering": 0.0,
            "modularity_louvain": None,
            "n_communities_louvain": 0,
            "largest_component_diameter": None,
            "largest_component_avg_path_length": None,
        }, []

    transitivity = nx.transitivity(G)
    avg_clustering = nx.average_clustering(G)

    communities = nx.community.louvain_communities(G, seed=42, weight="weight")
    modularity = nx.community.modularity(G, communities, weight="weight")

    largest_cc = max(nx.connected_components(G), key=len)
    if len(largest_cc) >= 2:
        lcc_sub = G.subgraph(largest_cc)
        diameter = nx.diameter(lcc_sub)
        avg_path = nx.average_shortest_path_length(lcc_sub)
    else:
        diameter = None
        avg_path = None

    return {
        "transitivity": round(transitivity, 4),
        "avg_clustering": round(avg_clustering, 4),
        "modularity_louvain": round(modularity, 4),
        "n_communities_louvain": len(communities),
        "largest_component_diameter": diameter,
        "largest_component_avg_path_length": _round_or_none(avg_path, 4),
    }, list(communities)


def _write_communities_csv(gt_dir, step, communities, modularity):
    """Write ground_truth_communities_{step}.csv and sidecar meta JSON.

    Format matches analyze_survey.write_gt_communities_csv: columns are
    (agent_name, community_id) with 1-based IDs sorted by min member name.
    """
    if not communities:
        return
    sorted_comms = sorted(communities, key=lambda s: min(s))
    community_map = {}
    for cid, members in enumerate(sorted_comms, start=1):
        for name in members:
            community_map[name] = cid

    path = os.path.join(gt_dir, f"ground_truth_communities_{step}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agent_name", "community_id"])
        for name in sorted(community_map):
            w.writerow([name, community_map[name]])

    meta_path = os.path.join(
        gt_dir, f"ground_truth_communities_{step}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "step":             step,
            "n_communities":    len(sorted_comms),
            "modularity":       modularity,
            "truth_definition": "observed_interaction_louvain_communities",
        }, f, indent=2)


def _checkpoint_metrics(step, edge_rows, chat_rows, prev_chat_keys):
    persona_set = set()
    sim_time = ""
    for r in edge_rows:
        persona_set.add(r["node_a"])
        persona_set.add(r["node_b"])
        if not sim_time:
            sim_time = r.get("sim_time", "")
    n = len(persona_set)
    n_pairs = n * (n - 1) // 2

    n_edges_cum = sum(1 for r in edge_rows if r["tie_cumulative"])
    n_edges_recent = sum(1 for r in edge_rows if r["tie_recent"])
    density_cum = (n_edges_cum / n_pairs) if n_pairs else 0.0
    mean_deg = (2 * n_edges_cum / n) if n else 0.0

    degree = _degree_dict(edge_rows)
    max_deg = max(degree.values()) if degree else 0

    bc = agent_betweenness(edge_rows)
    nonzero_bc = {k: v for k, v in bc.items() if v > 0}
    max_bc = max(nonzero_bc.values()) if nonzero_bc else 0.0

    hub_name, _ = highest_degree_agent(edge_rows)
    broker_name, _ = highest_betweenness_agent(edge_rows)
    hub_eq_broker = (
        hub_name is not None
        and broker_name is not None
        and hub_name == broker_name
    )
    cumulative_shape = graph_scope_summary(edge_rows, "count_cumulative")
    recent_shape = graph_scope_summary(edge_rows, "count_recent")

    G_cum = _build_nx_graph(edge_rows, "count_cumulative")
    comm_metrics, communities = _community_metrics(G_cum)

    active_cum = set()
    for c in chat_rows:
        active_cum.add(c["node_a"])
        active_cum.add(c["node_b"])

    chat_keys = {(c["created"], c["node_a"], c["node_b"]) for c in chat_rows}
    new_keys = chat_keys - prev_chat_keys
    new_chats = [
        c for c in chat_rows
        if (c["created"], c["node_a"], c["node_b"]) in new_keys
    ]
    new_active = set()
    for c in new_chats:
        new_active.add(c["node_a"])
        new_active.add(c["node_b"])

    return {
        "step": step,
        "sim_time": sim_time,
        "n_personas": n,
        "n_chats_cumulative": len(chat_rows),
        "n_new_chats_since_prev": len(new_chats),
        "n_active_agents_cumulative": len(active_cum),
        "n_new_active_agents_since_prev": len(new_active),
        "n_edges_cumulative": n_edges_cum,
        "n_edges_recent_8h": n_edges_recent,
        "density_cumulative": round(density_cum, 4),
        "mean_degree_cumulative": round(mean_deg, 3),
        "max_degree": max_deg,
        "top3_degree": _top3(degree, "{}={}"),
        "max_betweenness": round(max_bc, 4),
        "top3_betweenness": _top3(bc, "{}={:.4f}"),
        "hub": hub_name or "",
        "broker": broker_name or "",
        "hub_eq_broker": int(hub_eq_broker),
        "cumulative_betweenness_top_share": _round_or_none(
            cumulative_shape["betweenness_top_share"], 4),
        "cumulative_betweenness_hhi": _round_or_none(
            cumulative_shape["betweenness_hhi"], 4),
        "cumulative_component_count": cumulative_shape["component_count"],
        "cumulative_largest_component_size": (
            cumulative_shape["largest_component_size"]),
        "cumulative_reachable_pair_fraction": _round_or_none(
            cumulative_shape["reachable_pair_fraction"], 4),
        "cumulative_mean_reachable_shortest_path_length": _round_or_none(
            cumulative_shape["mean_reachable_shortest_path_length"], 4),
        "cumulative_diameter_reachable": cumulative_shape["diameter_reachable"],
        "cumulative_bridge_edge_count": cumulative_shape["bridge_edge_count"],
        "density_recent_8h": _round_or_none(recent_shape["density"], 4),
        "mean_degree_recent_8h": _round_or_none(
            recent_shape["mean_degree"], 3),
        "max_degree_recent_8h": recent_shape["max_degree"],
        "top3_degree_recent_8h": recent_shape["top3_degree"],
        "max_betweenness_recent_8h": _round_or_none(
            recent_shape["max_betweenness"], 4),
        "top3_betweenness_recent_8h": recent_shape["top3_betweenness"],
        "hub_recent_8h": recent_shape["hub"],
        "broker_recent_8h": recent_shape["broker"],
        "hub_eq_broker_recent_8h": recent_shape["hub_eq_broker"],
        "betweenness_top_share_recent_8h": _round_or_none(
            recent_shape["betweenness_top_share"], 4),
        "betweenness_hhi_recent_8h": _round_or_none(
            recent_shape["betweenness_hhi"], 4),
        "component_count_recent_8h": recent_shape["component_count"],
        "largest_component_size_recent_8h": (
            recent_shape["largest_component_size"]),
        "reachable_pair_fraction_recent_8h": _round_or_none(
            recent_shape["reachable_pair_fraction"], 4),
        "mean_reachable_shortest_path_length_recent_8h": _round_or_none(
            recent_shape["mean_reachable_shortest_path_length"], 4),
        "diameter_reachable_recent_8h": recent_shape["diameter_reachable"],
        "bridge_edge_count_recent_8h": recent_shape["bridge_edge_count"],
        "transitivity": comm_metrics["transitivity"],
        "avg_clustering": comm_metrics["avg_clustering"],
        "modularity_louvain": comm_metrics["modularity_louvain"],
        "n_communities_louvain": comm_metrics["n_communities_louvain"],
        "largest_component_diameter": comm_metrics["largest_component_diameter"],
        "largest_component_avg_path_length": (
            comm_metrics["largest_component_avg_path_length"]),
    }, chat_keys, communities


def _discover_steps(survey_dir):
    gt_dir = os.path.join(survey_dir, "ground_truth")
    if not os.path.isdir(gt_dir):
        return []
    edges_re = re.compile(r"^ground_truth_edges_(\d+)\.csv$")
    chats_re = re.compile(r"^ground_truth_chats_(\d+)\.csv$")
    edges_steps = set()
    chats_steps = set()
    for fn in os.listdir(gt_dir):
        m = edges_re.match(fn)
        if m:
            edges_steps.add(int(m.group(1)))
            continue
        m = chats_re.match(fn)
        if m:
            chats_steps.add(int(m.group(1)))
    return sorted(edges_steps & chats_steps)


# ── main ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 2:
        sys.exit(
            "Usage: python survey_network_summary.py <survey_dir>\n"
            "Example: python survey_network_summary.py "
            "../../environment/frontend_server/storage/"
            "prepost_the_ville_n15_brokerprobe-1/survey"
        )

    survey_dir = os.path.abspath(sys.argv[1])
    if not os.path.isdir(survey_dir):
        sys.exit(f"survey dir not found: {survey_dir}")

    steps = _discover_steps(survey_dir)
    if not steps:
        sys.exit(
            f"no ground-truth checkpoints found under "
            f"{survey_dir}/ground_truth"
        )

    print(f"Found {len(steps)} checkpoints: {steps}")

    rows_out = []
    prev_chat_keys = set()
    gt_dir = os.path.join(survey_dir, "ground_truth")
    for step in steps:
        edges = _read_edges_csv(
            os.path.join(gt_dir, f"ground_truth_edges_{step}.csv")
        )
        chats = _read_chats_csv(
            os.path.join(gt_dir, f"ground_truth_chats_{step}.csv")
        )
        metrics, chat_keys, communities = _checkpoint_metrics(
            step, edges, chats, prev_chat_keys
        )
        prev_chat_keys = chat_keys
        rows_out.append(metrics)

        _write_communities_csv(gt_dir, step, communities,
                               metrics.get("modularity_louvain"))

        mod_str = (f"{metrics['modularity_louvain']:.4f}"
                   if metrics['modularity_louvain'] is not None else "n/a")
        diam_str = (str(metrics['largest_component_diameter'])
                    if metrics['largest_component_diameter'] is not None
                    else "n/a")
        print(
            f"  step={step:>5}  sim_time={metrics['sim_time']:<19}  "
            f"edges_cum={metrics['n_edges_cumulative']:>3}  "
            f"max_deg={metrics['max_degree']:>2}  "
            f"max_bc={metrics['max_betweenness']:.4f}  "
            f"hub={metrics['hub']:<20}  "
            f"broker={metrics['broker']:<20}  "
            f"hub==broker={metrics['hub_eq_broker']}  "
            f"new_chats={metrics['n_new_chats_since_prev']:>3}  "
            f"new_active={metrics['n_new_active_agents_since_prev']:>2}  "
            f"trans={metrics['transitivity']:.4f}  "
            f"mod={mod_str}  "
            f"comms={metrics['n_communities_louvain']}  "
            f"comp={metrics['cumulative_component_count']}  "
            f"lcc={metrics['cumulative_largest_component_size']}  "
            f"diam={diam_str}"
        )

    out_path = os.path.join(survey_dir, "network_summary_over_time.csv")
    fieldnames = list(rows_out[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print(f"\nWrote {out_path}")
    print(
        "Open it directly in Excel / VS Code / Cursor to review the trajectory.\n"
        "Look in particular at:\n"
        "  - hub_eq_broker per row (any 0 means the broker probe is feasible "
        "at that checkpoint),\n"
        "  - n_new_chats_since_prev (sleep-period diagnostic; near-zero means "
        "agents stopped interacting),\n"
        "  - max_degree and density_cumulative (intervention strength)."
    )


if __name__ == "__main__":
    main()
