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


def agent_betweenness(edge_rows):
    """Return {name: betweenness_score} for all agents in the roster.

    Implements Brandes' algorithm on the undirected unweighted graph
    induced by edges with count_cumulative > 0. All roster nodes from
    edge_rows are included (isolates score 0.0). Scores are normalized
    by 2 / ((n-1)(n-2)) for n >= 3 so they lie in [0, 1]; for n < 3 the
    raw (unnormalized) score is returned (which will be 0.0 everywhere).

    Reference: Brandes, U. (2001). "A faster algorithm for betweenness
    centrality." Journal of Mathematical Sociology, 25(2), 163-177.
    """
    nodes_set = set()
    adj = defaultdict(set)
    for row in edge_rows:
        nodes_set.add(row["node_a"])
        nodes_set.add(row["node_b"])
        if int(row["count_cumulative"]) > 0:
            adj[row["node_a"]].add(row["node_b"])
            adj[row["node_b"]].add(row["node_a"])

    nodes = sorted(nodes_set)
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


def highest_betweenness_agent(edge_rows):
    """Return (name, betweenness_score) for the top-betweenness agent.

    Returns (None, 0.0) when the graph has no edges (all scores 0).
    Ties broken alphabetically for determinism.
    """
    bc = agent_betweenness(edge_rows)
    if not bc or max(bc.values()) == 0.0:
        return None, 0.0
    best = sorted(bc.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return best[0], best[1]


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
