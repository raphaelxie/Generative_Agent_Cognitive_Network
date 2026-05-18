"""
Ground-truth interaction logging for generative agents simulations.

Scans chat ConceptNodes across all personas, deduplicates, and writes:
  (a) a raw chat-event log  (one row per unique conversation)
  (b) an undirected tie graph (one row per unordered persona pair)

Usage (from project root):
  python reverie/backend_server/ground_truth_log.py <sim_code> [wave_id] [output_dir]

Or from reverie/backend_server:
  python ground_truth_log.py <sim_code> [wave_id] [output_dir]
"""
import csv
import itertools
import os
import sys
from collections import defaultdict, deque
from datetime import datetime, timedelta

_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
os.chdir(_backend_dir)

from utils import fs_storage

DT_FMT = "%Y-%m-%d %H:%M:%S"

CHAT_COLUMNS = [
    "sim_code", "step", "sim_time", "created",
    "node_a", "node_b", "n_turns", "description",
]

EDGE_COLUMNS = [
    "wave_id", "step", "sim_time",
    "node_a", "node_b",
    "tie_cumulative", "tie_recent",
    "count_cumulative", "count_recent",
]


def build_ground_truth(personas, sim_code, step, curr_time,
                       recent_window_minutes=480, wave_id=""):
    """
    Return (chat_rows, edge_rows) from in-memory chat ConceptNodes.

    chat_rows : list[dict] -- one per deduplicated conversation.
    edge_rows : list[dict] -- one per unordered persona pair (full roster),
                including pairs that never interacted.
    """
    sim_time_str = curr_time.strftime(DT_FMT)
    recent_cutoff = curr_time - timedelta(minutes=recent_window_minutes)

    # -- collect & dedup chat nodes across all personas --
    seen = set()
    chat_rows = []
    # (node_a, node_b) -> list of created datetimes, for aggregation
    pair_times = defaultdict(list)

    for persona in personas.values():
        for node in persona.a_mem.seq_chat:
            node_a, node_b = sorted([node.subject, node.object])
            dedup_key = (node_a, node_b, node.created)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            filling = node.filling if node.filling else []
            chat_rows.append({
                "sim_code":    sim_code,
                "step":        step,
                "sim_time":    sim_time_str,
                "created":     node.created.strftime(DT_FMT),
                "node_a":      node_a,
                "node_b":      node_b,
                "n_turns":     len(filling),
                "description": node.description,
            })
            pair_times[(node_a, node_b)].append(node.created)

    chat_rows.sort(key=lambda r: r["created"])

    # -- build undirected edge list for the full roster --
    persona_names = sorted(personas.keys())
    edge_rows = []
    for a, b in itertools.combinations(persona_names, 2):
        times = pair_times.get((a, b), [])
        count_cum = len(times)
        count_rec = sum(1 for t in times if t >= recent_cutoff)
        edge_rows.append({
            "wave_id":          wave_id,
            "step":             step,
            "sim_time":         sim_time_str,
            "node_a":           a,
            "node_b":           b,
            "tie_cumulative":   1 if count_cum > 0 else 0,
            "tie_recent":       1 if count_rec > 0 else 0,
            "count_cumulative": count_cum,
            "count_recent":     count_rec,
        })

    return chat_rows, edge_rows


def _safe_count(row, count_field):
    try:
        return int(row.get(count_field, 0))
    except Exception:
        return 0


def graph_nodes(edge_rows, excluded_nodes=None):
    """Return sorted roster nodes, optionally excluding analysis-time nodes."""
    excluded = set(excluded_nodes or [])
    nodes = set()
    for row in edge_rows:
        if row["node_a"] not in excluded:
            nodes.add(row["node_a"])
        if row["node_b"] not in excluded:
            nodes.add(row["node_b"])
    return sorted(nodes)


def graph_adjacency(edge_rows, count_field="count_cumulative",
                    excluded_nodes=None):
    """Build an undirected adjacency map for the requested edge-count field."""
    excluded = set(excluded_nodes or [])
    nodes = graph_nodes(edge_rows, excluded)
    adj = {node: set() for node in nodes}
    for row in edge_rows:
        a, b = row["node_a"], row["node_b"]
        if a in excluded or b in excluded:
            continue
        if _safe_count(row, count_field) > 0:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    return adj


def degree_by_agent(edge_rows, count_field="count_cumulative",
                    excluded_nodes=None):
    """Return degree counts for all roster nodes in the requested graph view."""
    adj = graph_adjacency(edge_rows, count_field, excluded_nodes)
    return {node: len(neighbors) for node, neighbors in adj.items()}


def highest_degree_agent_for_field(edge_rows, count_field="count_cumulative",
                                   excluded_nodes=None):
    """Return the top-degree agent for the requested graph view."""
    degree = {
        node: deg
        for node, deg in degree_by_agent(
            edge_rows, count_field, excluded_nodes).items()
        if deg > 0
    }
    if not degree:
        return None, 0
    best = sorted(degree.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return best[0], best[1]


def highest_degree_agent(edge_rows):
    """Return (name, degree) for the agent with the highest cumulative degree.

    Degree = number of unique partners with count_cumulative > 0.
    Returns (None, 0) when no interactions exist.
    """
    degree = defaultdict(int)
    for row in edge_rows:
        if row["count_cumulative"] > 0:
            degree[row["node_a"]] += 1
            degree[row["node_b"]] += 1
    if not degree:
        return None, 0
    best = max(degree, key=degree.get)
    return best, degree[best]


def agent_betweenness_for_field(edge_rows, count_field="count_cumulative",
                                excluded_nodes=None):
    """Return betweenness scores for the requested graph view.

    Implements Brandes' algorithm on the undirected unweighted graph
    induced by edges with the selected count field > 0. All non-excluded
    roster nodes from edge_rows are included (isolates score 0.0). Scores
    are normalized by 2 / ((n-1)(n-2)) for n >= 3 so they lie in [0, 1];
    for n < 3 the raw score is returned (which will be 0.0 everywhere).

    Reference: Brandes, U. (2001). "A faster algorithm for betweenness
    centrality." Journal of Mathematical Sociology, 25(2), 163-177.
    """
    adj = graph_adjacency(edge_rows, count_field, excluded_nodes)
    nodes = sorted(adj.keys())
    bc = {v: 0.0 for v in nodes}

    for s in nodes:
        stack = []
        pred = {v: [] for v in nodes}
        sigma = {v: 0 for v in nodes}
        sigma[s] = 1
        dist = {v: -1 for v in nodes}
        dist[s] = 0
        queue = deque([s])
        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        delta = {v: 0.0 for v in nodes}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                bc[w] += delta[w]

    for v in bc:
        bc[v] /= 2.0

    n = len(nodes)
    if n >= 3:
        norm = 2.0 / ((n - 1) * (n - 2))
        for v in bc:
            bc[v] *= norm

    return dict(bc)


def agent_betweenness(edge_rows):
    """Return cumulative betweenness scores for all agents in the roster."""
    return agent_betweenness_for_field(edge_rows, "count_cumulative")


def highest_betweenness_agent_for_field(edge_rows,
                                        count_field="count_cumulative",
                                        excluded_nodes=None):
    """Return the top-betweenness agent for the requested graph view."""
    bc = agent_betweenness_for_field(edge_rows, count_field, excluded_nodes)
    if not bc or max(bc.values()) == 0.0:
        return None, 0.0
    best = sorted(bc.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return best[0], best[1]


def highest_betweenness_agent(edge_rows):
    """Return (name, betweenness_score) for the top-betweenness agent.

    Returns (None, 0.0) when the graph has no edges (all scores 0).
    Ties broken alphabetically for determinism.
    """
    return highest_betweenness_agent_for_field(edge_rows, "count_cumulative")


def graph_components(edge_rows, count_field="count_cumulative",
                     excluded_nodes=None):
    """Return connected components for the requested graph view."""
    adj = graph_adjacency(edge_rows, count_field, excluded_nodes)
    seen = set()
    components = []
    for start in sorted(adj):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nbr in sorted(adj[node]):
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        components.append(sorted(comp))
    components.sort(key=lambda c: (-len(c), c[0] if c else ""))
    return components


def shortest_path_summary(edge_rows, count_field="count_cumulative",
                          excluded_nodes=None):
    """Return reachable-pair and shortest-path summaries for a graph view."""
    adj = graph_adjacency(edge_rows, count_field, excluded_nodes)
    nodes = sorted(adj)
    n = len(nodes)
    total_pairs = n * (n - 1) // 2
    distances = []
    for i, src in enumerate(nodes):
        dist = {src: 0}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            for nbr in adj[node]:
                if nbr not in dist:
                    dist[nbr] = dist[node] + 1
                    queue.append(nbr)
        for dst in nodes[i + 1:]:
            if dst in dist:
                distances.append(dist[dst])
    reachable_pairs = len(distances)
    return {
        "n_pairs": total_pairs,
        "reachable_pairs": reachable_pairs,
        "reachable_pair_fraction": (
            reachable_pairs / total_pairs if total_pairs else None),
        "mean_reachable_shortest_path_length": (
            sum(distances) / len(distances) if distances else None),
        "diameter_reachable": max(distances) if distances else None,
    }


def bridge_edges(edge_rows, count_field="count_cumulative",
                 excluded_nodes=None):
    """Return undirected bridge edges for the requested graph view."""
    adj = graph_adjacency(edge_rows, count_field, excluded_nodes)
    timer = [0]
    visited = set()
    tin = {}
    low = {}
    bridges = []

    def dfs(node, parent=None):
        visited.add(node)
        timer[0] += 1
        tin[node] = low[node] = timer[0]
        for nbr in sorted(adj[node]):
            if nbr == parent:
                continue
            if nbr in visited:
                low[node] = min(low[node], tin[nbr])
            else:
                dfs(nbr, node)
                low[node] = min(low[node], low[nbr])
                if low[nbr] > tin[node]:
                    bridges.append(tuple(sorted((node, nbr))))

    for node in sorted(adj):
        if node not in visited:
            dfs(node)
    return sorted(set(bridges))


def _top3(score_dict, fmt):
    items = sorted(
        ((k, v) for k, v in score_dict.items() if v > 0),
        key=lambda kv: (-kv[1], kv[0]),
    )[:3]
    return "; ".join(fmt.format(k, v) for k, v in items)


def graph_scope_summary(edge_rows, count_field="count_cumulative",
                        excluded_nodes=None):
    """Return graph-level diagnostics for one cumulative or recent scope."""
    nodes = graph_nodes(edge_rows, excluded_nodes)
    n = len(nodes)
    n_pairs = n * (n - 1) // 2
    degree = degree_by_agent(edge_rows, count_field, excluded_nodes)
    n_edges = sum(degree.values()) // 2
    bc = agent_betweenness_for_field(edge_rows, count_field, excluded_nodes)
    nonzero_bc = {k: v for k, v in bc.items() if v > 0}
    bc_sum = sum(nonzero_bc.values())
    components = graph_components(edge_rows, count_field, excluded_nodes)
    paths = shortest_path_summary(edge_rows, count_field, excluded_nodes)
    bridges = bridge_edges(edge_rows, count_field, excluded_nodes)
    hub_name, _ = highest_degree_agent_for_field(
        edge_rows, count_field, excluded_nodes)
    broker_name, _ = highest_betweenness_agent_for_field(
        edge_rows, count_field, excluded_nodes)
    max_degree = max(degree.values()) if degree else 0
    max_bc = max(nonzero_bc.values()) if nonzero_bc else 0.0
    hub_eq_broker = (
        hub_name is not None
        and broker_name is not None
        and hub_name == broker_name
    )
    return {
        "n_nodes": n,
        "n_edges": n_edges,
        "density": (n_edges / n_pairs) if n_pairs else 0.0,
        "mean_degree": (2 * n_edges / n) if n else 0.0,
        "max_degree": max_degree,
        "top3_degree": _top3(degree, "{}={}"),
        "max_betweenness": max_bc,
        "top3_betweenness": _top3(bc, "{}={:.4f}"),
        "hub": hub_name or "",
        "broker": broker_name or "",
        "hub_eq_broker": int(hub_eq_broker),
        "betweenness_top_share": (max_bc / bc_sum) if bc_sum else None,
        "betweenness_hhi": (
            sum((v / bc_sum) ** 2 for v in nonzero_bc.values())
            if bc_sum else None),
        "component_count": len(components),
        "largest_component_size": len(components[0]) if components else 0,
        "reachable_pair_fraction": paths["reachable_pair_fraction"],
        "mean_reachable_shortest_path_length": (
            paths["mean_reachable_shortest_path_length"]),
        "diameter_reachable": paths["diameter_reachable"],
        "bridge_edge_count": len(bridges),
    }


def target_graph_summary(edge_rows, target, count_field="count_cumulative"):
    """Return target-specific diagnostics in the observed target-included view."""
    if not target:
        return {}
    degree = degree_by_agent(edge_rows, count_field)
    bc = agent_betweenness_for_field(edge_rows, count_field)
    bc_rank = {
        name: i + 1
        for i, (name, _) in enumerate(
            sorted(bc.items(), key=lambda kv: (-kv[1], kv[0])))
    }
    bridges = bridge_edges(edge_rows, count_field)
    incident_edges = []
    for row in edge_rows:
        if target not in (row["node_a"], row["node_b"]):
            continue
        if _safe_count(row, count_field) > 0:
            incident_edges.append(tuple(sorted((row["node_a"], row["node_b"]))))
    incident_bridge_edges = [edge for edge in incident_edges if edge in bridges]
    return {
        "target_degree": degree.get(target, 0),
        "target_betweenness": bc.get(target, 0.0),
        "target_betweenness_rank": bc_rank.get(target),
        "target_incident_edges": len(incident_edges),
        "target_incident_bridge_edges": len(incident_bridge_edges),
    }


def write_ground_truth_csv(personas, sim_code, step, curr_time,
                           output_dir, recent_window_minutes=480,
                           wave_id=""):
    """Write ground_truth_chats_{step}.csv and ground_truth_edges_{step}.csv."""
    chat_rows, edge_rows = build_ground_truth(
        personas, sim_code, step, curr_time,
        recent_window_minutes=recent_window_minutes,
        wave_id=wave_id,
    )

    os.makedirs(output_dir, exist_ok=True)

    chats_path = os.path.join(output_dir, f"ground_truth_chats_{step}.csv")
    with open(chats_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CHAT_COLUMNS)
        w.writeheader()
        w.writerows(chat_rows)

    edges_path = os.path.join(output_dir, f"ground_truth_edges_{step}.csv")
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EDGE_COLUMNS)
        w.writeheader()
        w.writerows(edge_rows)

    return chats_path, edges_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python ground_truth_log.py <sim_code> [wave_id] [output_dir]")
        print("Example: python ground_truth_log.py base_the_ville_n15")
        sys.exit(1)

    sim_code = sys.argv[1].strip()
    wave_id = sys.argv[2].strip() if len(sys.argv) > 2 else ""
    output_dir = (sys.argv[3].strip() if len(sys.argv) > 3
                  else os.path.join(fs_storage, sim_code, "ground_truth"))

    from reverie import ReverieServer
    print(f"Loading simulation: {sim_code}")
    rs = ReverieServer(sim_code, sim_code)

    chat_rows, edge_rows = build_ground_truth(
        rs.personas, sim_code, rs.step, rs.curr_time, wave_id=wave_id,
    )

    n_chats = len(chat_rows)
    n_ties = sum(1 for r in edge_rows if r["tie_cumulative"])
    n_pairs = len(edge_rows)
    print(f"Unique chats: {n_chats}")
    print(f"Non-zero ties: {n_ties} / {n_pairs} pairs")

    chats_path, edges_path = write_ground_truth_csv(
        rs.personas, sim_code, rs.step, rs.curr_time,
        output_dir, wave_id=wave_id,
    )
    print(f"Wrote {chats_path}")
    print(f"Wrote {edges_path}")


if __name__ == "__main__":
    main()
