"""Predeclared, API-free rules for the 64-hour network cognition pilot."""
import hashlib
import math
from dataclasses import asdict, dataclass
from datetime import timedelta

PROTOCOL_VERSION = "midnight-pilot-v1"
MODEL = "gpt-6-luna"
SECONDS_PER_STEP = 10
END_STEP = 64 * 360
CHECK_START = 20 * 360
CHECK_END = 32 * 360
CHECK_INTERVAL = 30 * 6
SURVEY_STEPS = (0, 10*360, 14*360, 18*360, 34*360, 38*360,
                42*360, 58*360, 62*360, 64*360)
CHECK_STEPS = tuple(range(CHECK_START, CHECK_END + 1, CHECK_INTERVAL))
FOLLOWUP_OFFSETS = (2*360, 4*360, 24*360)
RECENT_MINUTES = 120


def priority_order(roster, seed):
    # Separate from the simulation RNG: inspecting the graph cannot change it.
    return sorted(roster, key=lambda name: (
        hashlib.sha256(f"{seed}:{name}".encode()).hexdigest(), name))


def graph_from_chats(personas, start, now, recent_minutes=None):
    """Count distinct conversation starts in [start, now); never future events."""
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(sorted(personas))
    cutoff = start if recent_minutes is None else max(
        start, now - timedelta(minutes=recent_minutes))
    seen = set()
    for persona in personas.values():
        for node in persona.a_mem.seq_chat:
            a, b = sorted((node.subject, node.object))
            if a == b or a not in graph or b not in graph:
                continue
            key = (a, b, node.created)
            if key in seen or not cutoff <= node.created < now:
                continue
            seen.add(key)
            count = graph.get_edge_data(a, b, {}).get("weight", 0)
            graph.add_edge(a, b, weight=count + 1)
    return graph


def graph_summary(graph):
    import networkx as nx
    components = sorted((len(c) for c in nx.connected_components(graph)), reverse=True)
    return {"nodes": len(graph), "edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "participating_nodes": sum(d > 0 for _, d in graph.degree()),
            "components": components, "conversations": sum(
                d.get("weight", 1) for _, _, d in graph.edges(data=True))}


@dataclass
class CheckpointSelector:
    priority: list
    previous: dict = None
    selected: dict = None
    records: list = None

    def __post_init__(self):
        if self.records is None:
            self.records = []

    def observe(self, step, graph):
        import networkx as nx
        if step not in CHECK_STEPS:
            raise ValueError("Checkpoint checks must be on the predeclared grid")
        if self.records and step <= self.records[-1]["step"]:
            raise ValueError("Checkpoint checks must be strictly increasing")
        if set(graph) != set(self.priority):
            raise ValueError("Checkpoint roster changed")
        degrees = dict(graph.degree())
        between = nx.betweenness_centrality(graph, normalized=True, weight=None)
        highest_degree = max(degrees.values(), default=0)
        highest_between = max(between.values(), default=0)
        hubs = [n for n in self.priority if degrees[n] == highest_degree]
        brokers = [n for n in self.priority if math.isclose(
            between[n], highest_between, rel_tol=0, abs_tol=1e-12)]
        overlap = sorted(set(hubs) & set(brokers))
        eligible = highest_degree > 0 and highest_between > 0 and not overlap
        record = {"step": step, "degrees": degrees, "betweenness": between,
                  "hub_ties": hubs, "broker_ties": brokers, "overlap": overlap,
                  "hub": hubs[0] if eligible else None,
                  "broker": brokers[0] if eligible else None,
                  "eligible": eligible, "graph": graph_summary(graph)}
        previous = self.previous
        stable = bool(eligible and previous and previous["eligible"]
                      and step - previous["step"] == CHECK_INTERVAL
                      and (record["hub"], record["broker"]) ==
                          (previous["hub"], previous["broker"]))
        record["stable"] = stable
        record["locked_now"] = stable and self.selected is None
        if record["locked_now"]:
            self.selected = dict(record)
        self.records.append(record)
        self.previous = record
        return record

    def status(self, step):
        if self.selected:
            return "selected"
        return "no_eligible_checkpoint" if step >= CHECK_END else "searching"

    def to_dict(self):
        return asdict(self)


def labels_at(step, selected_step=None):
    labels = []
    if step in SURVEY_STEPS:
        labels.append("initial" if step == 0 else "final" if step == END_STEP else "scheduled")
    if selected_step is not None:
        if step == selected_step:
            labels.append("checkpoint")
        for hours, offset in zip((2, 4, 24), FOLLOWUP_OFFSETS):
            if step == selected_step + offset:
                labels.append(f"checkpoint_plus_{hours}h")
    return labels


def next_boundary(step, selected_step=None):
    events = set(SURVEY_STEPS) | set(CHECK_STEPS) | {END_STEP}
    if selected_step is not None:
        events.update(selected_step + offset for offset in FOLLOWUP_OFFSETS)
    # Recovery every 30 minutes, including nights; this never skips engine ticks.
    events.add(min(END_STEP, (step // CHECK_INTERVAL + 1) * CHECK_INTERVAL))
    return min(s for s in events if step < s <= END_STEP)
