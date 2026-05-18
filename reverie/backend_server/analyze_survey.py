"""
Analyze perception survey results against ground truth.

Reads perception_survey_{wave}.csv and ground_truth_edges_{step}.csv,
prints a structured comparison showing what each respondent perceived,
what actually happened, and where perception matched or diverged.

Outputs:
  - Console report (per-respondent + overall per wave)
  - analysis_micro_tie_metrics.csv            (structured metrics)
  - analysis_micro_tie_metrics_README.md      (self-describing sidecar)
  - analysis_micro_tie_metrics_by_truth.csv
  - analysis_micro_tie_metrics_by_truth_README.md
  - analysis_bridge_rank_metrics.csv
  - analysis_bridge_rank_metrics_README.md
  - analysis_community_group_metrics.csv      (NMI/ARI vs Louvain truth)
  - analysis_community_group_metrics_README.md
  - ground_truth/ground_truth_communities_{step}.csv
  - ground_truth/ground_truth_communities_{step}_meta.json

Micro-tie aggregation rule (IMPORTANT):
  Each respondent answers two directed micro-tie questions per unordered
  pair {j, k}: one with target_j=j, target=k and one with target_j=k,
  target=j. For confusion-matrix computation we aggregate these to a
  single unordered "said-tied" judgment using the OR rule: the pair is
  perceived as tied iff AT LEAST ONE directed observation is "1".
  Fail-safe rows and rows with non-binary values are dropped; a pair is
  excluded from n_pairs_evaluated only if ALL of its directed
  observations are dropped. Pre-aggregation directed values remain in
  the raw perception_survey_*.csv for later re-analysis.

Usage (from project root or from reverie/backend_server):
  python analyze_survey.py <survey_dir>

Example:
  python analyze_survey.py ../../environment/frontend_server/storage/July1_the_ville_isabella_maria_klaus-pilot-ncn-1/survey
"""
import csv
import json
import math
import os
import sys
from collections import defaultdict

_ORIG_CWD = os.getcwd()
from ground_truth_log import agent_betweenness
os.chdir(_ORIG_CWD)

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False

# ── CSV loading ────────────────────────────────────────────────────────

def load_survey_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_edges_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── ground-truth helpers ──────────────────────────────────────────────

def build_gt_tie_map(edge_rows):
    """Return {frozenset({a, b}): row_dict} for all pairs."""
    return {frozenset({r["node_a"], r["node_b"]}): r for r in edge_rows}


def gt_has_tie(tie_map, a, b):
    pair = frozenset({a, b})
    row = tie_map.get(pair)
    if row is None:
        return False
    return int(row["tie_cumulative"]) == 1


def gt_degree(tie_map, roster):
    deg = {}
    for name in roster:
        deg[name] = sum(
            1 for other in roster if other != name
            and gt_has_tie(tie_map, name, other)
        )
    return deg


def gt_degree_ranking(tie_map, roster):
    """Return names sorted from highest to lowest degree (stable alpha tiebreak)."""
    deg = gt_degree(tie_map, roster)
    return sorted(roster, key=lambda n: (-deg[n], n))


# ── micro-tie confusion matrix (undirected pairs, OR rule) ───────────
#
# Each respondent answers two directed micro-tie questions per unordered
# pair {j, k}: one with target_j=j, target=k and one with target_j=k,
# target=j. We aggregate to an unordered "said-tied" judgment using the
# OR rule: the pair is perceived as tied iff AT LEAST ONE directed
# observation is "1".
#
# Fail-safe rows (is_fail_safe == "1") and rows with non-binary values
# are dropped from the confusion counts. A pair is excluded (not counted
# in n_pairs_evaluated) only if ALL of its directed observations are
# dropped. The count of dropped directed rows per respondent is reported
# separately as n_fail_safe so the reader can see how much was lost.

METRICS_COLUMNS = [
    "wave_id", "step", "respondent",
    "tp", "fp", "fn", "tn",
    "precision", "recall", "fpr", "fnr", "pair_accuracy",
    "n_fail_safe", "n_pairs_evaluated",
]

METRICS_BY_TRUTH_COLUMNS = [
    "wave_id", "step", "truth_layer", "truth_file", "respondent",
    "tp", "fp", "fn", "tn",
    "precision", "recall", "fpr", "fnr", "pair_accuracy",
    "n_fail_safe", "n_pairs_evaluated",
]

BRIDGE_RANK_METRICS_COLUMNS = [
    "wave_id", "step", "truth_definition", "respondent",
    "actual_top_broker", "actual_top_betweenness",
    "perceived_rank_of_actual_top_broker",
    "bridge_top1_hit", "bridge_top3_hit",
    "bridge_mean_abs_rank_error",
    "n_ranked", "n_expected", "n_fail_safe",
]

# Maps each separated micro-tie construct to its natural ground-truth layer.
# Used by compute_micro_tie_by_construct for both natural-match and cross-scoring.
CONSTRUCT_TRUTH_MAP = {
    "micro_tie_interaction": "observed_interaction",
    "micro_tie_social":      "background_social_tie",
    "micro_tie_group":       "background_or_interaction",
}

BY_CONSTRUCT_COLUMNS = [
    "wave_id", "step", "construct", "truth_layer", "respondent",
    "precision", "recall", "fpr", "fnr", "pair_accuracy",
    "n_pairs", "n_fail_safe",
]


def _parse_perceived(val, is_fs):
    """Return 0, 1, or None (drop). Fail-safe rows are dropped."""
    if is_fs == "1":
        return None
    if val == "0":
        return 0
    if val == "1":
        return 1
    return None


def _safe_div(num, den):
    return (num / den) if den > 0 else None


def _confusion_record(wave_id, step, respondent, tp, fp, fn, tn,
                      n_fail_safe, n_pairs_evaluated):
    return {
        "wave_id":            wave_id,
        "step":               step,
        "respondent":         respondent,
        "tp":                 tp,
        "fp":                 fp,
        "fn":                 fn,
        "tn":                 tn,
        "precision":          _safe_div(tp, tp + fp),
        "recall":             _safe_div(tp, tp + fn),
        "fpr":                _safe_div(fp, fp + tn),
        "fnr":                _safe_div(fn, fn + tp),
        "pair_accuracy":      _safe_div(tp + tn, tp + fp + fn + tn),
        "n_fail_safe":        n_fail_safe,
        "n_pairs_evaluated":  n_pairs_evaluated,
    }


def build_interaction_truth_labels(edge_rows):
    """Return {frozenset({a, b}): 0/1} using observed interaction labels."""
    return {
        frozenset({r["node_a"], r["node_b"]}): int(r["tie_cumulative"])
        for r in edge_rows
    }


def build_background_truth_labels(background_rows):
    """Return {frozenset({a, b}): 0/1} using background social-tie labels."""
    return {
        frozenset({r["node_a"], r["node_b"]}): int(r["tie_background"])
        for r in background_rows
    }


def union_truth_labels(*truth_maps):
    """Return pair labels where a dyad is positive in any supplied truth map."""
    pairs = set()
    for truth_map in truth_maps:
        pairs.update(truth_map.keys())
    return {
        pair: 1 if any(truth_map.get(pair, 0) for truth_map in truth_maps) else 0
        for pair in pairs
    }


def compute_micro_tie_confusion_for_truth(survey_rows, truth_labels,
                                          wave_id, step,
                                          question_type="micro_tie"):
    """Return list of per-respondent confusion dicts plus one __overall__ row.

    Requires truth_labels to be present; returns [] if ground truth is missing.
    Requires survey_rows to have 'target_j' populated; callers should run
    reconstruct_target_j first for old-format CSVs.

    question_type selects which instrument rows to evaluate (default: "micro_tie"
    for backward compatibility). Pass one of "micro_tie_interaction",
    "micro_tie_social", or "micro_tie_group" to score a separated construct.
    """
    if not truth_labels:
        return []

    by_resp = defaultdict(list)
    for r in survey_rows:
        if r.get("question_type") != question_type:
            continue
        by_resp[r["respondent"]].append(r)

    results = []
    total_tp = total_fp = total_fn = total_tn = 0
    total_fs = 0
    total_pairs = 0

    for respondent in sorted(by_resp.keys()):
        rows = by_resp[respondent]

        by_pair = defaultdict(list)
        n_fs_rows = 0
        for r in rows:
            j = r.get("target_j", "")
            k = r.get("target", "")
            if not j or not k or j == k:
                continue
            pair = frozenset({j, k})
            if pair not in truth_labels:
                continue
            is_fs = r.get("is_fail_safe", "0")
            if is_fs == "1":
                n_fs_rows += 1
            perceived = _parse_perceived(r.get("value", ""), is_fs)
            by_pair[pair].append(perceived)

        tp = fp = fn = tn = 0
        for pair, observations in by_pair.items():
            valid = [o for o in observations if o is not None]
            if not valid:
                continue
            perceived_pair = 1 if any(o == 1 for o in valid) else 0
            actual = truth_labels[pair]
            if perceived_pair == 1 and actual == 1:
                tp += 1
            elif perceived_pair == 1 and actual == 0:
                fp += 1
            elif perceived_pair == 0 and actual == 1:
                fn += 1
            else:
                tn += 1

        n_pairs = tp + fp + fn + tn
        results.append(_confusion_record(
            wave_id, step, respondent, tp, fp, fn, tn, n_fs_rows, n_pairs))

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        total_fs += n_fs_rows
        total_pairs += n_pairs

    results.append(_confusion_record(
        wave_id, step, "__overall__",
        total_tp, total_fp, total_fn, total_tn, total_fs, total_pairs))
    return results


def compute_micro_tie_confusion(survey_rows, edge_rows, wave_id, step):
    """Legacy interaction-only confusion metrics."""
    if not edge_rows:
        return []
    return compute_micro_tie_confusion_for_truth(
        survey_rows,
        build_interaction_truth_labels(edge_rows),
        wave_id,
        step,
    )


def add_truth_metadata(records, truth_layer, truth_file):
    out = []
    for rec in records:
        row = dict(rec)
        row["truth_layer"] = truth_layer
        row["truth_file"] = truth_file
        out.append(row)
    return out


def compute_micro_tie_confusion_by_truth(survey_rows, edge_rows,
                                         background_rows, wave_id, step,
                                         interaction_truth_file,
                                         background_truth_file):
    records = []
    if edge_rows:
        interaction_labels = build_interaction_truth_labels(edge_rows)
        interaction_records = compute_micro_tie_confusion_for_truth(
            survey_rows, interaction_labels, wave_id, step)
        records.extend(add_truth_metadata(
            interaction_records,
            "observed_interaction",
            interaction_truth_file,
        ))
    else:
        interaction_labels = {}

    if background_rows:
        background_labels = build_background_truth_labels(background_rows)
        background_records = compute_micro_tie_confusion_for_truth(
            survey_rows, background_labels, wave_id, step)
        records.extend(add_truth_metadata(
            background_records,
            "background_social_tie",
            background_truth_file,
        ))

        if interaction_labels:
            union_labels = union_truth_labels(interaction_labels, background_labels)
            union_records = compute_micro_tie_confusion_for_truth(
                survey_rows, union_labels, wave_id, step)
            records.extend(add_truth_metadata(
                union_records,
                "background_or_interaction",
                f"{background_truth_file}+{interaction_truth_file}",
            ))

    return records


def compute_micro_tie_by_construct(survey_rows, edge_rows, background_rows,
                                   wave_id, step):
    """Score each separated micro-tie construct against all available truth layers.

    Produces a 3x3 cross-scoring matrix (3 constructs x up to 3 truth layers)
    plus per-respondent rows and one __overall__ row per (construct, truth_layer)
    combination.  Only constructs present in survey_rows and truth layers with
    available ground-truth data generate rows; missing combinations are skipped
    silently.

    Returns a list of dicts with columns matching BY_CONSTRUCT_COLUMNS.
    """
    truth_layers = {}
    if edge_rows:
        truth_layers["observed_interaction"] = build_interaction_truth_labels(edge_rows)
    if background_rows:
        truth_layers["background_social_tie"] = build_background_truth_labels(background_rows)
    if "observed_interaction" in truth_layers and "background_social_tie" in truth_layers:
        truth_layers["background_or_interaction"] = union_truth_labels(
            truth_layers["observed_interaction"],
            truth_layers["background_social_tie"],
        )

    if not truth_layers:
        return []

    records = []
    for construct in CONSTRUCT_TRUTH_MAP:
        for layer_name, labels in truth_layers.items():
            conf = compute_micro_tie_confusion_for_truth(
                survey_rows, labels, wave_id, step, question_type=construct)
            for rec in conf:
                records.append({
                    "wave_id":       wave_id,
                    "step":          step,
                    "construct":     construct,
                    "truth_layer":   layer_name,
                    "respondent":    rec["respondent"],
                    "precision":     rec["precision"],
                    "recall":        rec["recall"],
                    "fpr":           rec["fpr"],
                    "fnr":           rec["fnr"],
                    "pair_accuracy": rec["pair_accuracy"],
                    "n_pairs":       rec["n_pairs_evaluated"],
                    "n_fail_safe":   rec["n_fail_safe"],
                })
    return records


def observed_betweenness_ranking(edge_rows):
    """Return observed-interaction cumulative betweenness ranking."""
    if not edge_rows:
        return [], {}
    bc = agent_betweenness(edge_rows)
    ordered = sorted(bc.keys(), key=lambda n: (-bc[n], n))
    return ordered, bc


def compute_bridge_rank_metrics(survey_rows, edge_rows, wave_id, step):
    """Score bridge_rank rows against cumulative observed betweenness rank."""
    actual_order, bc = observed_betweenness_ranking(edge_rows)
    if not actual_order:
        return []
    actual_rank = {name: i + 1 for i, name in enumerate(actual_order)}
    actual_top = actual_order[0]
    actual_top_bc = bc.get(actual_top, 0.0)
    n_expected = len(actual_order)

    by_resp = defaultdict(list)
    for row in survey_rows:
        if row.get("question_type") == "bridge_rank":
            by_resp[row["respondent"]].append(row)

    records = []
    summary_vals = defaultdict(list)
    total_fs = 0
    for respondent in sorted(by_resp.keys()):
        rows = by_resp[respondent]
        n_fs = sum(1 for r in rows if r.get("is_fail_safe") == "1")
        total_fs += n_fs
        valid_rows = [
            r for r in rows
            if r.get("is_fail_safe") != "1"
            and r.get("target") in actual_rank
            and str(r.get("value", "")).strip().isdigit()
        ]
        perceived_rank = {
            r["target"]: int(r["value"])
            for r in valid_rows
        }
        errors = [
            abs(perceived_rank[name] - actual_rank[name])
            for name in actual_rank
            if name in perceived_rank
        ]
        top_rank = perceived_rank.get(actual_top)
        rec = {
            "wave_id": wave_id,
            "step": step,
            "truth_definition": "observed_interaction_cumulative_betweenness_rank",
            "respondent": respondent,
            "actual_top_broker": actual_top,
            "actual_top_betweenness": actual_top_bc,
            "perceived_rank_of_actual_top_broker": top_rank,
            "bridge_top1_hit": 1 if top_rank == 1 else 0 if top_rank else None,
            "bridge_top3_hit": 1 if top_rank and top_rank <= 3 else 0 if top_rank else None,
            "bridge_mean_abs_rank_error": (
                sum(errors) / len(errors) if errors else None),
            "n_ranked": len(perceived_rank),
            "n_expected": n_expected,
            "n_fail_safe": n_fs,
        }
        records.append(rec)
        for key in (
            "perceived_rank_of_actual_top_broker",
            "bridge_top1_hit",
            "bridge_top3_hit",
            "bridge_mean_abs_rank_error",
            "n_ranked",
        ):
            if rec[key] is not None:
                summary_vals[key].append(rec[key])

    if records:
        overall = {
            "wave_id": wave_id,
            "step": step,
            "truth_definition": "observed_interaction_cumulative_betweenness_rank",
            "respondent": "__overall__",
            "actual_top_broker": actual_top,
            "actual_top_betweenness": actual_top_bc,
            "perceived_rank_of_actual_top_broker": _mean_or_none(
                summary_vals["perceived_rank_of_actual_top_broker"]),
            "bridge_top1_hit": _mean_or_none(summary_vals["bridge_top1_hit"]),
            "bridge_top3_hit": _mean_or_none(summary_vals["bridge_top3_hit"]),
            "bridge_mean_abs_rank_error": _mean_or_none(
                summary_vals["bridge_mean_abs_rank_error"]),
            "n_ranked": _mean_or_none(summary_vals["n_ranked"]),
            "n_expected": n_expected,
            "n_fail_safe": total_fs,
        }
        records.append(overall)
    return records


def _mean_or_none(values):
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _fmt_ratio(v):
    return "   NA" if v is None else f"{v:.3f}"


# ── ground-truth community detection (Louvain via networkx) ───────────

def compute_gt_communities(edge_rows):
    """Louvain community detection on the cumulative observed-interaction graph.

    Uses `tie_cumulative == 1` edges weighted by `count_cumulative`.
    Returns (community_map, modularity, n_communities) where
    community_map = {agent_name: community_id (1-indexed)}.
    Returns ({}, None, 0) when networkx is unavailable or graph is empty.
    """
    if not _HAS_NX:
        return {}, None, 0
    G = nx.Graph()
    all_nodes = set()
    for r in edge_rows:
        all_nodes.add(r["node_a"])
        all_nodes.add(r["node_b"])
        if int(r.get("tie_cumulative", 0)) == 1:
            G.add_edge(r["node_a"], r["node_b"],
                       weight=max(1, int(r.get("count_cumulative", 1))))
    G.add_nodes_from(all_nodes)
    if G.number_of_nodes() == 0:
        return {}, None, 0
    communities = nx.community.louvain_communities(G, seed=42, weight="weight")
    community_map = {}
    for cid, members in enumerate(
            sorted(communities, key=lambda s: min(s)), start=1):
        for name in members:
            community_map[name] = cid
    try:
        modularity = nx.community.modularity(G, communities, weight="weight")
    except Exception:
        modularity = None
    return community_map, modularity, len(communities)


def write_gt_communities_csv(community_map, modularity, n_communities,
                              step, gt_dir):
    """Write ground_truth_communities_{step}.csv and a sidecar meta JSON."""
    if not community_map:
        return None
    os.makedirs(gt_dir, exist_ok=True)
    csv_path = os.path.join(gt_dir, f"ground_truth_communities_{step}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agent_name", "community_id"])
        for name in sorted(community_map):
            w.writerow([name, community_map[name]])
    meta_path = os.path.join(
        gt_dir, f"ground_truth_communities_{step}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "step":           step,
            "n_communities":  n_communities,
            "modularity":     modularity,
            "truth_definition": "observed_interaction_louvain_communities",
        }, f, indent=2)
    return csv_path


# ── pure-Python NMI and ARI helpers ───────────────────────────────────

def _nmi(labels_true, labels_pred):
    """Normalized Mutual Information (arithmetic mean normalisation)."""
    n = len(labels_true)
    if n == 0:
        return None
    classes_t = set(labels_true)
    classes_p = set(labels_pred)
    # contingency table
    contingency = defaultdict(int)
    for t, p in zip(labels_true, labels_pred):
        contingency[(t, p)] += 1
    # marginals
    count_t = defaultdict(int)
    count_p = defaultdict(int)
    for t, p in zip(labels_true, labels_pred):
        count_t[t] += 1
        count_p[p] += 1

    def _entropy(counts):
        total = sum(counts.values())
        h = 0.0
        for c in counts.values():
            if c > 0:
                p = c / total
                h -= p * math.log(p)
        return h

    h_t = _entropy(count_t)
    h_p = _entropy(count_p)
    mi = 0.0
    for (t, p), n_tp in contingency.items():
        if n_tp > 0:
            mi += (n_tp / n) * math.log(
                (n_tp * n) / (count_t[t] * count_p[p]))
    denom = (h_t + h_p) / 2.0
    if denom == 0:
        return 1.0 if mi == 0 else None
    return mi / denom


def _ari(labels_true, labels_pred):
    """Adjusted Rand Index."""
    n = len(labels_true)
    if n < 2:
        return None
    contingency = defaultdict(int)
    for t, p in zip(labels_true, labels_pred):
        contingency[(t, p)] += 1
    # sum of C(n_ij, 2)
    sum_comb = sum(v * (v - 1) // 2 for v in contingency.values())
    count_t = defaultdict(int)
    count_p = defaultdict(int)
    for t, p in zip(labels_true, labels_pred):
        count_t[t] += 1
        count_p[p] += 1
    sum_a = sum(v * (v - 1) // 2 for v in count_t.values())
    sum_b = sum(v * (v - 1) // 2 for v in count_p.values())
    n_choose_2 = n * (n - 1) // 2
    expected = (sum_a * sum_b) / n_choose_2 if n_choose_2 > 0 else 0
    max_val = (sum_a + sum_b) / 2.0
    denom = max_val - expected
    if denom == 0:
        return 1.0 if sum_comb == expected else 0.0
    return (sum_comb - expected) / denom


# ── community group perception metrics ────────────────────────────────

COMMUNITY_GROUP_METRICS_COLUMNS = [
    "wave_id", "step", "truth_definition", "respondent",
    "n_perceived_groups", "n_actual_groups",
    "nmi", "ari",
    "n_agents_placed", "n_expected", "n_fail_safe",
]


def compute_community_group_metrics(survey_rows, edge_rows, wave_id, step):
    """Score community_group rows against Louvain ground-truth communities.

    Returns a list of per-respondent dicts plus one __overall__ row.
    Requires networkx; returns [] if unavailable or no edge truth.
    """
    if not _HAS_NX or not edge_rows:
        return []
    community_map, modularity, n_actual = compute_gt_communities(edge_rows)
    if not community_map:
        return []

    by_resp = defaultdict(list)
    for r in survey_rows:
        if r.get("question_type") == "community_group":
            by_resp[r["respondent"]].append(r)

    if not by_resp:
        return []

    records = []
    summary_nmi = []
    summary_ari = []
    summary_n_perceived = []
    summary_n_placed = []
    total_fs = 0

    for respondent in sorted(by_resp.keys()):
        rows = by_resp[respondent]
        n_fs = sum(1 for r in rows if r.get("is_fail_safe") == "1")
        total_fs += n_fs

        perceived_map = {}
        for r in rows:
            name = r.get("target", "")
            val = r.get("value", "")
            if name and val.strip().isdigit():
                perceived_map[name] = int(val)

        # Align to agents present in both partitions
        common = [a for a in community_map if a in perceived_map]
        n_placed = len(common)
        n_expected = len(community_map)

        if n_placed >= 2:
            gt_labels   = [community_map[a]   for a in common]
            perc_labels = [perceived_map[a] for a in common]
            nmi_val = _nmi(gt_labels, perc_labels)
            ari_val = _ari(gt_labels, perc_labels)
        else:
            nmi_val = None
            ari_val = None

        n_perceived = len(set(perceived_map.values())) if perceived_map else 0

        rec = {
            "wave_id":          wave_id,
            "step":             step,
            "truth_definition": "observed_interaction_louvain_communities",
            "respondent":       respondent,
            "n_perceived_groups": n_perceived,
            "n_actual_groups":  n_actual,
            "nmi":              nmi_val,
            "ari":              ari_val,
            "n_agents_placed":  n_placed,
            "n_expected":       n_expected,
            "n_fail_safe":      n_fs,
        }
        records.append(rec)

        if nmi_val is not None:
            summary_nmi.append(nmi_val)
        if ari_val is not None:
            summary_ari.append(ari_val)
        summary_n_perceived.append(n_perceived)
        summary_n_placed.append(n_placed)

    if records:
        records.append({
            "wave_id":          wave_id,
            "step":             step,
            "truth_definition": "observed_interaction_louvain_communities",
            "respondent":       "__overall__",
            "n_perceived_groups": _mean_or_none(summary_n_perceived),
            "n_actual_groups":  n_actual,
            "nmi":              _mean_or_none(summary_nmi),
            "ari":              _mean_or_none(summary_ari),
            "n_agents_placed":  _mean_or_none(summary_n_placed),
            "n_expected":       len(community_map),
            "n_fail_safe":      total_fs,
        })
    return records


COMMUNITY_GROUP_METRICS_README = """\
# analysis_community_group_metrics.csv

Structured community/group perception metrics produced by `analyze_survey.py`.

Rows are one per `(wave_id, respondent)` plus one `__overall__` row per wave.
Truth definition: `observed_interaction_louvain_communities`.

Ground truth is computed via Louvain modularity maximisation
(`networkx.community.louvain_communities`, seed=42, weight=count_cumulative)
on the cumulative observed-interaction graph (`tie_cumulative == 1` edges).
Community IDs are 1-indexed, sorted lexicographically by the minimum member
name within each community.

## Columns

- `wave_id`            — survey wave identifier
- `step`               — simulation step at which the survey was taken
- `truth_definition`   — always `observed_interaction_louvain_communities`
- `respondent`         — agent name (or `__overall__` for the wave mean)
- `n_perceived_groups` — number of distinct groups the respondent named
- `n_actual_groups`    — number of Louvain communities in ground truth
- `nmi`                — Normalized Mutual Information between perceived and
                         ground-truth partition (arithmetic-mean normalisation);
                         1.0 = perfect agreement, 0.0 = no mutual information
- `ari`                — Adjusted Rand Index between perceived and ground-truth
                         partition; 1.0 = perfect, 0.0 = random, negative =
                         worse than random
- `n_agents_placed`    — number of agents present in both the perceived map
                         and the ground-truth map (used for NMI/ARI)
- `n_expected`         — total agents in the ground-truth community map
- `n_fail_safe`        — number of community_group rows flagged is_fail_safe

The `__overall__` row reports respondent means for `n_perceived_groups`,
`nmi`, `ari`, and `n_agents_placed`; `n_fail_safe` is the wave total.
"""


def write_community_group_metrics_csv(all_records, survey_dir):
    """Write one row per (wave, respondent) plus one __overall__ row per wave.

    Also writes a self-describing sidecar README.
    """
    if not all_records:
        return None
    path = os.path.join(survey_dir, "analysis_community_group_metrics.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COMMUNITY_GROUP_METRICS_COLUMNS)
        for rec in all_records:
            w.writerow([
                fmt_metric_cell(rec[c]) for c in COMMUNITY_GROUP_METRICS_COLUMNS
            ])
    readme_path = os.path.join(
        survey_dir, "analysis_community_group_metrics_README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(COMMUNITY_GROUP_METRICS_README)
    return path


MICRO_TIE_METRICS_README = """\
# analysis_micro_tie_metrics.csv

Structured micro-tie perception metrics produced by `analyze_survey.py`.

## Aggregation rule

Each respondent answers two directed micro-tie questions per unordered
pair {j, k}: one with `target_j = j, target = k` and one with
`target_j = k, target = j`. For confusion-matrix computation, these are
aggregated to a single unordered "said-tied" judgment using the **OR
rule**:

    perceived_pair = 1  iff  at least one directed observation is "1"
    perceived_pair = 0  iff  all valid directed observations are "0"

## Fail-safe policy

Rows with `is_fail_safe == "1"` or non-binary `value` are dropped from
the confusion counts. A pair is excluded from `n_pairs_evaluated` only
if ALL of its directed observations are dropped. The count of dropped
directed rows per respondent is reported in the `n_fail_safe` column.

## Ground truth

`tie_cumulative` from the matching `ground_truth/ground_truth_edges_{step}.csv`
is the reference label for each unordered pair. A pair with
`tie_cumulative == 1` is the actual-positive class.

## Columns

| Column               | Meaning                                                              |
|----------------------|----------------------------------------------------------------------|
| wave_id              | Survey wave identifier (e.g. "pre", "post")                          |
| step                 | Simulation step at which the wave was captured                        |
| respondent           | Agent name, or "__overall__" for the per-wave micro-average          |
| tp                   | True positives: said-tied AND tie_cumulative==1                       |
| fp                   | False positives: said-tied AND tie_cumulative==0                      |
| fn                   | False negatives: said-not-tied AND tie_cumulative==1                  |
| tn                   | True negatives: said-not-tied AND tie_cumulative==0                   |
| precision            | tp / (tp + fp); empty cell when denominator is 0                      |
| recall               | tp / (tp + fn); empty cell when denominator is 0                      |
| fpr                  | fp / (fp + tn); empty cell when denominator is 0                      |
| fnr                  | fn / (fn + tp); empty cell when denominator is 0                      |
| pair_accuracy        | (tp + tn) / (tp + fp + fn + tn)                                       |
| n_fail_safe          | Number of directed micro-tie rows with is_fail_safe==1 (dropped)      |
| n_pairs_evaluated    | Number of unordered pairs with at least one valid observation         |

The `__overall__` row per wave is a **micro-average**: TP/FP/FN/TN are
summed across respondents and the ratios are recomputed from the sums.

## Invariants

    tp + fp + fn + tn == n_pairs_evaluated    # for every row
"""


MICRO_TIE_METRICS_BY_TRUTH_README = """\
# analysis_micro_tie_metrics_by_truth.csv

Structured micro-tie perception metrics produced by `analyze_survey.py`,
computed against multiple explicitly named truth layers.

## Truth layers

- `observed_interaction`: dynamic interaction truth from
  `ground_truth/ground_truth_edges_{step}.csv`; a positive dyad means the
  pair has at least one observed chat by that wave's simulation step.
- `background_social_tie`: baseline social-tie truth from
  `ground_truth/background_social_edges.csv`; a positive dyad means the
  pair has conservative evidence of a pre-existing social tie.
- `background_or_interaction`: derived comparison view; a positive dyad
  means the pair is positive in either `observed_interaction` or
  `background_social_tie`.

## Interpretation

The same perceived micro-tie can be a false positive under
`observed_interaction` but a true positive under `background_social_tie`.
For the broad survey wording ("connection or relationship"),
`background_or_interaction` is the least construct-mismatched comparison:
false positives there are perceived ties supported by neither prior social
history nor observed interaction.

All other aggregation, fail-safe, and metric definitions match
`analysis_micro_tie_metrics.csv`.
"""


BY_CONSTRUCT_README = """\
# analysis_micro_tie_by_construct.csv

Structured micro-tie perception metrics produced by `analyze_survey.py`,
computed separately for each of the four micro-tie constructs and scored
against every available ground-truth layer (3x3 cross-scoring matrix).

## The Four Constructs

| question_type          | Instrument wording (paraphrased)                          |
|------------------------|-----------------------------------------------------------|
| micro_tie              | "connection or relationship" (broad; original instrument) |
| micro_tie_interaction  | "directly interacted" (conversation, joint activity)      |
| micro_tie_social       | "social relationship or personal connection" (friends,    |
|                        |  acquaintances, meaningful social bond)                   |
| micro_tie_group        | "same social group or circle" (friend group, social       |
|                        |  circle, regularly spend time with same people)           |

This file covers only the three separated constructs (`micro_tie_interaction`,
`micro_tie_social`, `micro_tie_group`). Broad `micro_tie` metrics are in
`analysis_micro_tie_metrics_by_truth.csv`.

## Natural Truth-Layer Mapping

Each separated construct has a natural ground-truth layer that best matches
its theoretical definition:

| construct             | natural truth_layer          | rationale                              |
|-----------------------|------------------------------|----------------------------------------|
| micro_tie_interaction | observed_interaction         | Truth = observed chats/joint activity  |
| micro_tie_social      | background_social_tie        | Truth = pre-existing social tie record |
| micro_tie_group       | background_or_interaction    | Truth = either tie type (union)        |

On-diagonal cells in the cross-scoring matrix (construct x truth_layer) are
natural matches; they represent the theoretically grounded accuracy question.

## Cross-Scoring Matrix

This file scores every construct against every available truth layer:

    rows:    construct in {micro_tie_interaction, micro_tie_social, micro_tie_group}
    columns: truth_layer in {observed_interaction, background_social_tie,
                             background_or_interaction}

Off-diagonal cells answer "what if we measured construct X against truth Y?"
For example:
- micro_tie_social x observed_interaction: do agents who perceive a social
  relationship match the interaction record? High precision suggests agents
  only infer social ties from actually observed interactions.
- micro_tie_interaction x background_social_tie: do agents who perceive
  direct interaction map onto background tie pairs? High recall would imply
  background-tied pairs are perceived as having interacted regardless of
  actual conversation history.

## Aggregation and Fail-Safe Policy

Same as `analysis_micro_tie_metrics_by_truth.csv`:
- Directed rows are OR-aggregated per unordered pair per respondent.
- Fail-safe rows (is_fail_safe == "1") are dropped before aggregation.
- Pairs with all observations dropped are excluded from n_pairs_evaluated.
- The `__overall__` row per (construct, truth_layer, wave) is a micro-average
  (TP/FP/FN/TN summed across respondents, ratios recomputed from sums).

## Cost Note

Adding the three separated instruments roughly 4x the number of micro-tie API
calls per survey wave compared to the original single-instrument design:
- 1 broad connection call per (respondent, j) pair  [original]
- 1 interaction call per (respondent, j) pair        [new]
- 1 social-tie call per (respondent, j) pair         [new]
- 1 group co-membership call per (respondent, j) pair [new]
Each call also triggers an independent retrieval pass with a construct-specific
focal query, so memory retrieval volume also quadruples.

## Columns

| Column        | Meaning                                                                   |
|---------------|---------------------------------------------------------------------------|
| wave_id       | Survey wave identifier (e.g. "pre", "post")                               |
| step          | Simulation step at which the wave was captured                            |
| construct     | Instrument variant (micro_tie_interaction / social / group)               |
| truth_layer   | Ground-truth layer scored against                                         |
| respondent    | Agent name, or "__overall__" for the per-(construct, layer) micro-average |
| precision     | tp / (tp + fp); empty when denominator is 0                               |
| recall        | tp / (tp + fn); empty when denominator is 0                               |
| fpr           | fp / (fp + tn); empty when denominator is 0                               |
| fnr           | fn / (fn + tp); empty when denominator is 0                               |
| pair_accuracy | (tp + tn) / n_pairs                                                       |
| n_pairs       | Unordered pairs with at least one valid observation                       |
| n_fail_safe   | Directed rows dropped due to is_fail_safe==1                              |
"""


BRIDGE_RANK_METRICS_README = """\
# analysis_bridge_rank_metrics.csv

Structured bridge-role perception metrics produced by `analyze_survey.py`.

Rows are one per `(wave_id, respondent)` plus one `__overall__` row per wave.
The first-pass truth definition is intentionally singular:
`observed_interaction_cumulative_betweenness_rank`.

This means the actual bridge/broker ordering is computed from the observed
interaction graph's cumulative ties (`count_cumulative > 0`) using normalized
betweenness centrality. It does not use background social-tie truth, recent
window truth, target-excluded views, or post-shock counterfactuals.

Core columns:
- `actual_top_broker`: highest-betweenness agent in observed cumulative truth.
- `perceived_rank_of_actual_top_broker`: where a respondent ranked that agent
  in the full-roster `bridge_rank` question.
- `bridge_top1_hit`: 1 iff the respondent ranked the actual top broker first.
- `bridge_top3_hit`: 1 iff the respondent ranked the actual top broker in the
  top three.
- `bridge_mean_abs_rank_error`: mean absolute difference between perceived
  bridge rank and actual cumulative-betweenness rank across ranked agents.

The `__overall__` row reports respondent means for rank, hit-rate, rank-error,
and ranked-count fields.
"""


def fmt_metric_cell(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return v


def write_micro_tie_metrics_csv(all_records, survey_dir):
    """Write one row per (wave, respondent) plus one __overall__ row per wave.

    Also writes a self-describing sidecar README next to the CSV so the
    aggregation rule and column definitions travel with the data.
    """
    if not all_records:
        return None
    path = os.path.join(survey_dir, "analysis_micro_tie_metrics.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(METRICS_COLUMNS)
        for rec in all_records:
            w.writerow([fmt_metric_cell(rec[c]) for c in METRICS_COLUMNS])

    readme_path = os.path.join(
        survey_dir, "analysis_micro_tie_metrics_README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(MICRO_TIE_METRICS_README)

    return path


def write_micro_tie_metrics_by_truth_csv(all_records, survey_dir):
    """Write one row per (wave, truth layer, respondent)."""
    if not all_records:
        return None
    path = os.path.join(survey_dir, "analysis_micro_tie_metrics_by_truth.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(METRICS_BY_TRUTH_COLUMNS)
        for rec in all_records:
            w.writerow([fmt_metric_cell(rec[c]) for c in METRICS_BY_TRUTH_COLUMNS])

    readme_path = os.path.join(
        survey_dir, "analysis_micro_tie_metrics_by_truth_README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(MICRO_TIE_METRICS_BY_TRUTH_README)

    return path


def write_micro_tie_by_construct_csv(all_records, survey_dir):
    """Write one row per (wave, construct, truth_layer, respondent) plus __overall__.

    Also writes a sidecar README explaining the constructs, truth-layer mapping,
    and how to read the 3x3 cross-scoring matrix.
    """
    if not all_records:
        return None
    path = os.path.join(survey_dir, "analysis_micro_tie_by_construct.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(BY_CONSTRUCT_COLUMNS)
        for rec in all_records:
            w.writerow([fmt_metric_cell(rec[c]) for c in BY_CONSTRUCT_COLUMNS])

    readme_path = os.path.join(
        survey_dir, "analysis_micro_tie_by_construct_README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(BY_CONSTRUCT_README)

    return path


def write_bridge_rank_metrics_csv(all_records, survey_dir):
    """Write one row per (wave, respondent) plus one __overall__ row per wave."""
    if not all_records:
        return None
    path = os.path.join(survey_dir, "analysis_bridge_rank_metrics.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(BRIDGE_RANK_METRICS_COLUMNS)
        for rec in all_records:
            w.writerow([
                fmt_metric_cell(rec[c]) for c in BRIDGE_RANK_METRICS_COLUMNS
            ])

    readme_path = os.path.join(
        survey_dir, "analysis_bridge_rank_metrics_README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(BRIDGE_RANK_METRICS_README)

    return path


# ── reconstruct target_j for old-format CSVs ─────────────────────────

def reconstruct_target_j(survey_rows, roster):
    """Add target_j to micro_tie rows that lack it, using row-order convention.

    Convention: for each respondent, micro_tie rows cycle through
    j in sorted(roster), with len(roster)-1 rows per j (one per k != j).
    """
    n = len(roster)
    k_per_j = n - 1

    respondent_blocks = defaultdict(list)
    for row in survey_rows:
        if row["question_type"] == "micro_tie":
            respondent_blocks[row["respondent"]].append(row)

    for respondent, rows in respondent_blocks.items():
        expected = n * k_per_j
        if len(rows) != expected:
            for row in rows:
                row["target_j"] = "?"
            continue
        for i, row in enumerate(rows):
            j_index = i // k_per_j
            row["target_j"] = roster[j_index]


# ── discovery ─────────────────────────────────────────────────────────

def discover_waves(survey_dir):
    """Return list of (wave_id, survey_path, edges_path, step) sorted by step."""
    waves = []
    gt_dir = os.path.join(survey_dir, "ground_truth")

    for fname in sorted(os.listdir(survey_dir)):
        if not fname.startswith("perception_survey_") or not fname.endswith(".csv"):
            continue
        wave_id = fname[len("perception_survey_"):-len(".csv")]
        survey_path = os.path.join(survey_dir, fname)

        rows = load_survey_csv(survey_path)
        if not rows:
            continue
        step = rows[0]["step"]

        edges_path = os.path.join(gt_dir, f"ground_truth_edges_{step}.csv")
        if not os.path.exists(edges_path):
            edges_path = None

        waves.append((wave_id, survey_path, edges_path, step))

    waves.sort(key=lambda w: int(w[3]))
    return waves


def discover_background_edges(survey_dir):
    """Return background_social_edges.csv path if present."""
    path = os.path.join(
        survey_dir, "ground_truth", "background_social_edges.csv")
    return path if os.path.exists(path) else None


def truth_file_label(survey_dir, path):
    if not path:
        return ""
    return os.path.relpath(path, survey_dir)


# ── printing helpers ──────────────────────────────────────────────────

def print_header(text):
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def print_subheader(text):
    print(f"\n  --- {text} ---")


# ── single-wave analysis ─────────────────────────────────────────────

def analyze_wave(wave_id, survey_rows, edge_rows):
    if not survey_rows:
        print("  (no survey rows)")
        return []

    step = survey_rows[0]["step"]
    sim_time = survey_rows[0]["sim_time"]
    print_header(f"Wave: {wave_id}  |  step={step}  |  sim_time={sim_time}")

    # Build ground truth
    tie_map = build_gt_tie_map(edge_rows) if edge_rows else {}
    roster = sorted({r["node_a"] for r in edge_rows} | {r["node_b"] for r in edge_rows}) if edge_rows else []

    if not roster:
        roster = sorted({r["respondent"] for r in survey_rows})

    # Ensure target_j is present (handles old-format CSVs)
    has_target_j = "target_j" in survey_rows[0]
    if not has_target_j:
        reconstruct_target_j(survey_rows, roster)

    # Confusion matrix records (per respondent + __overall__)
    conf_records = compute_micro_tie_confusion(
        survey_rows, edge_rows, wave_id, step)
    conf_by_resp = {r["respondent"]: r for r in conf_records}

    # ── Ground Truth Summary ──
    print_subheader("Ground Truth (observed interactions)")
    if edge_rows:
        deg = gt_degree(tie_map, roster)
        for r in edge_rows:
            tie = "TIE" if int(r["tie_cumulative"]) else "   "
            print(f"    {r['node_a']:25s} <-> {r['node_b']:25s}  "
                  f"{tie}  (chats={r['count_cumulative']})")
        print(f"    Degree: ", end="")
        print(", ".join(f"{n}={deg[n]}" for n in roster))
    else:
        print("    (no ground-truth edge file found)")

    # ── Per-respondent detail ──
    respondents = sorted({r["respondent"] for r in survey_rows})
    micro_rows = [r for r in survey_rows if r["question_type"] == "micro_tie"]
    rank_rows = [r for r in survey_rows if r["question_type"] == "centrality_rank"]
    bridge_rows = [r for r in survey_rows if r["question_type"] == "bridge_rank"]
    self_rows = [r for r in survey_rows if r["question_type"] == "self_position"]

    for respondent in respondents:
        print_subheader(f"Respondent: {respondent}")

        # Micro-tie
        my_micro = [r for r in micro_rows if r["respondent"] == respondent]
        if my_micro:
            print(f"    Micro-tie reports (\"does j know k?\"):")
            correct = 0
            total = 0
            for r in my_micro:
                j = r.get("target_j", "?")
                k = r["target"]
                val = r["value"]
                perceived = int(val) if val in ("0", "1") else "?"

                if edge_rows:
                    actual = 1 if gt_has_tie(tie_map, j, k) else 0
                    match = "OK" if perceived == actual else "MISS"
                    if perceived != "?":
                        total += 1
                        if perceived == actual:
                            correct += 1
                else:
                    actual = "?"
                    match = ""

                print(f"      {j:25s} -> {k:25s}  "
                      f"said={perceived}  actual={actual}  {match}")

            if total > 0:
                print(f"    Accuracy: {correct}/{total} ({100*correct/total:.0f}%)")

            rec = conf_by_resp.get(respondent)
            if rec and rec["n_pairs_evaluated"] > 0:
                print(f"    Confusion (unordered pairs, OR rule): "
                      f"TP={rec['tp']} FP={rec['fp']} "
                      f"FN={rec['fn']} TN={rec['tn']}  "
                      f"prec={_fmt_ratio(rec['precision'])} "
                      f"rec={_fmt_ratio(rec['recall'])} "
                      f"FPR={_fmt_ratio(rec['fpr'])} "
                      f"FNR={_fmt_ratio(rec['fnr'])} "
                      f"pair_acc={_fmt_ratio(rec['pair_accuracy'])}  "
                      f"(n_pairs={rec['n_pairs_evaluated']}, "
                      f"n_fs_rows={rec['n_fail_safe']})")

        # Centrality rank
        my_rank = sorted(
            [r for r in rank_rows if r["respondent"] == respondent],
            key=lambda r: int(r["value"])
        )
        if my_rank:
            perceived_order = [r["target"] for r in my_rank]
            print(f"    Centrality ranking (perceived): "
                  + " > ".join(perceived_order))
            if edge_rows:
                actual_order = gt_degree_ranking(tie_map, roster)
                print(f"    Centrality ranking (actual degree): "
                      + " > ".join(actual_order))
                match = "MATCH" if perceived_order == actual_order else "DIFFERS"
                print(f"    Ranking comparison: {match}")

        # Bridge rank
        my_bridge = sorted(
            [r for r in bridge_rows if r["respondent"] == respondent],
            key=lambda r: int(r["value"])
        )
        if my_bridge:
            perceived_order = [r["target"] for r in my_bridge]
            print(f"    Bridge ranking (perceived): "
                  + " > ".join(perceived_order))
            if edge_rows:
                actual_order, _ = observed_betweenness_ranking(edge_rows)
                print(f"    Bridge ranking (actual betweenness): "
                      + " > ".join(actual_order))
                match = "MATCH" if perceived_order == actual_order else "DIFFERS"
                print(f"    Bridge comparison: {match}")

        # Self-position
        my_self = [r for r in self_rows if r["respondent"] == respondent]
        if my_self:
            self_val = my_self[0]["value"]
            print(f"    Self-position rating: {self_val}/5", end="")
            if edge_rows:
                deg = gt_degree(tie_map, roster)
                own_deg = deg.get(respondent, 0)
                max_deg = max(deg.values()) if deg else 0
                print(f"  (actual degree={own_deg}, max_degree={max_deg})")
            else:
                print()

    # Overall wave summary (micro-average across all respondents)
    overall = conf_by_resp.get("__overall__")
    if overall and overall["n_pairs_evaluated"] > 0:
        print_subheader("Overall (all respondents, unordered pairs)")
        print(f"    TP={overall['tp']} FP={overall['fp']} "
              f"FN={overall['fn']} TN={overall['tn']}  "
              f"prec={_fmt_ratio(overall['precision'])} "
              f"rec={_fmt_ratio(overall['recall'])} "
              f"FPR={_fmt_ratio(overall['fpr'])} "
              f"FNR={_fmt_ratio(overall['fnr'])} "
              f"pair_acc={_fmt_ratio(overall['pair_accuracy'])}  "
              f"(n_pairs={overall['n_pairs_evaluated']}, "
              f"n_fs_rows={overall['n_fail_safe']})")

    return conf_records


# ── pre/post comparison ──────────────────────────────────────────────

def compare_waves(wave_a, rows_a, edges_a, wave_b, rows_b, edges_b):
    print_header(f"Pre/Post Comparison: {wave_a} -> {wave_b}")

    respondents = sorted(
        {r["respondent"] for r in rows_a} | {r["respondent"] for r in rows_b}
    )

    tie_map_a = build_gt_tie_map(edges_a) if edges_a else {}
    tie_map_b = build_gt_tie_map(edges_b) if edges_b else {}
    roster = sorted(
        ({r["node_a"] for r in edges_a} | {r["node_b"] for r in edges_a})
        if edges_a else {r["respondent"] for r in rows_a}
    )

    # Ground truth change
    print_subheader("Ground Truth Changes")
    if edges_a and edges_b:
        map_a = {(r["node_a"], r["node_b"]): r for r in edges_a}
        map_b = {(r["node_a"], r["node_b"]): r for r in edges_b}
        for pair in sorted(map_a.keys()):
            ca = int(map_a[pair]["count_cumulative"])
            cb = int(map_b.get(pair, map_a[pair])["count_cumulative"])
            ta = int(map_a[pair]["tie_cumulative"])
            tb = int(map_b.get(pair, map_a[pair])["tie_cumulative"])
            if ca != cb or ta != tb:
                delta = cb - ca
                print(f"    {pair[0]:25s} <-> {pair[1]:25s}  "
                      f"tie: {ta}->{tb}  chats: {ca}->{cb} (delta={delta:+d})")
        deg_a = gt_degree(tie_map_a, roster)
        deg_b = gt_degree(tie_map_b, roster)
        for name in roster:
            if deg_a.get(name, 0) != deg_b.get(name, 0):
                print(f"    Degree change: {name}  {deg_a[name]} -> {deg_b[name]}")

    # Per-respondent perception changes
    def index_by_respondent(rows, qtype):
        out = defaultdict(list)
        for r in rows:
            if r["question_type"] == qtype:
                out[r["respondent"]].append(r)
        return out

    micro_a = index_by_respondent(rows_a, "micro_tie")
    micro_b = index_by_respondent(rows_b, "micro_tie")
    rank_a = index_by_respondent(rows_a, "centrality_rank")
    rank_b = index_by_respondent(rows_b, "centrality_rank")
    bridge_a = index_by_respondent(rows_a, "bridge_rank")
    bridge_b = index_by_respondent(rows_b, "bridge_rank")
    self_a = index_by_respondent(rows_a, "self_position")
    self_b = index_by_respondent(rows_b, "self_position")

    for resp in respondents:
        print_subheader(f"Perception Changes: {resp}")

        # Micro-tie changes
        def micro_key(r):
            return (r.get("target_j", "?"), r["target"])

        vals_a = {micro_key(r): r["value"] for r in micro_a.get(resp, [])}
        vals_b = {micro_key(r): r["value"] for r in micro_b.get(resp, [])}
        all_keys = sorted(set(vals_a) | set(vals_b))
        changes = []
        for key in all_keys:
            va = vals_a.get(key, "")
            vb = vals_b.get(key, "")
            if va != vb:
                changes.append((key[0], key[1], va, vb))
        if changes:
            print(f"    Micro-tie changes:")
            for j, k, va, vb in changes:
                print(f"      \"{j} -> {k}\":  {va} -> {vb}")
        else:
            print(f"    Micro-tie: no changes")

        # Centrality rank changes
        order_a = [r["target"] for r in sorted(
            rank_a.get(resp, []), key=lambda r: int(r["value"]))]
        order_b = [r["target"] for r in sorted(
            rank_b.get(resp, []), key=lambda r: int(r["value"]))]
        if order_a != order_b:
            print(f"    Rank changed: {' > '.join(order_a)}  ->  {' > '.join(order_b)}")
        else:
            print(f"    Rank: unchanged ({' > '.join(order_a)})")

        # Bridge rank changes
        bridge_order_a = [r["target"] for r in sorted(
            bridge_a.get(resp, []), key=lambda r: int(r["value"]))]
        bridge_order_b = [r["target"] for r in sorted(
            bridge_b.get(resp, []), key=lambda r: int(r["value"]))]
        if bridge_order_a or bridge_order_b:
            if bridge_order_a != bridge_order_b:
                print(f"    Bridge rank changed: "
                      f"{' > '.join(bridge_order_a)}  ->  "
                      f"{' > '.join(bridge_order_b)}")
            else:
                print(f"    Bridge rank: unchanged "
                      f"({' > '.join(bridge_order_a)})")

        # Self-position changes
        sp_a = self_a.get(resp, [{}])[0].get("value", "?") if self_a.get(resp) else "?"
        sp_b = self_b.get(resp, [{}])[0].get("value", "?") if self_b.get(resp) else "?"
        if sp_a != sp_b:
            print(f"    Self-position changed: {sp_a} -> {sp_b}")
        else:
            print(f"    Self-position: unchanged ({sp_a})")

    return


# ── main ─────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_survey.py <survey_dir>")
        print("  survey_dir should contain perception_survey_*.csv files")
        print("  and a ground_truth/ subdirectory with edge CSVs.")
        sys.exit(1)

    survey_dir = sys.argv[1].strip()
    if not os.path.isdir(survey_dir):
        print(f"Error: '{survey_dir}' is not a directory.")
        sys.exit(1)

    waves = discover_waves(survey_dir)
    if not waves:
        print(f"No perception_survey_*.csv files found in {survey_dir}")
        sys.exit(1)

    print(f"Found {len(waves)} wave(s) in {survey_dir}")

    # Load shock log if present
    shock_path = os.path.join(survey_dir, "shock_log.jsonl")
    if os.path.exists(shock_path):
        print(f"\nShock log ({shock_path}):")
        with open(shock_path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                print(f"  step={entry.get('step')}  {entry.get('command')}  "
                      f"agent={entry.get('target_agent', entry.get('agents_unshocked', ''))}")
    else:
        print("  (no shock_log.jsonl found)")

    background_edges_path = discover_background_edges(survey_dir)
    if background_edges_path:
        background_rows = load_edges_csv(background_edges_path)
        print(f"  Background social truth: {background_edges_path}")
    else:
        background_rows = []
        print("  (no background_social_edges.csv found)")

    # Analyze each wave independently
    loaded = []
    all_metric_records = []
    all_metric_records_by_truth = []
    all_construct_records = []
    all_bridge_rank_records = []
    all_community_group_records = []
    for wave_id, survey_path, edges_path, step in waves:
        survey_rows = load_survey_csv(survey_path)
        edge_rows = load_edges_csv(edges_path) if edges_path else []

        roster = sorted(
            {r["node_a"] for r in edge_rows} | {r["node_b"] for r in edge_rows}
        ) if edge_rows else sorted({r["respondent"] for r in survey_rows})

        has_target_j = survey_rows and "target_j" in survey_rows[0]
        if not has_target_j:
            reconstruct_target_j(survey_rows, roster)

        records = analyze_wave(wave_id, survey_rows, edge_rows)
        if records:
            all_metric_records.extend(records)
        by_truth_records = compute_micro_tie_confusion_by_truth(
            survey_rows,
            edge_rows,
            background_rows,
            wave_id,
            step,
            truth_file_label(survey_dir, edges_path),
            truth_file_label(survey_dir, background_edges_path),
        )
        if by_truth_records:
            all_metric_records_by_truth.extend(by_truth_records)

        construct_records = compute_micro_tie_by_construct(
            survey_rows, edge_rows, background_rows, wave_id, step)
        if construct_records:
            all_construct_records.extend(construct_records)

        bridge_records = compute_bridge_rank_metrics(
            survey_rows, edge_rows, wave_id, step)
        if bridge_records:
            all_bridge_rank_records.extend(bridge_records)

        # Community group perception metrics + ground-truth CSV
        if edge_rows and _HAS_NX:
            gt_dir = os.path.join(survey_dir, "ground_truth")
            community_map, modularity, n_communities = compute_gt_communities(
                edge_rows)
            gt_csv = write_gt_communities_csv(
                community_map, modularity, n_communities, step, gt_dir)
            if gt_csv:
                print(f"  Wrote {gt_csv}")
        community_records = compute_community_group_metrics(
            survey_rows, edge_rows, wave_id, step)
        if community_records:
            all_community_group_records.extend(community_records)

        loaded.append((wave_id, survey_rows, edge_rows))

    # Write structured micro-tie metrics CSV (one row per wave x respondent,
    # plus one __overall__ row per wave).
    if all_metric_records:
        metrics_path = write_micro_tie_metrics_csv(
            all_metric_records, survey_dir)
        if metrics_path:
            print(f"\n  Wrote {metrics_path}")

    if all_metric_records_by_truth:
        metrics_by_truth_path = write_micro_tie_metrics_by_truth_csv(
            all_metric_records_by_truth, survey_dir)
        if metrics_by_truth_path:
            print(f"  Wrote {metrics_by_truth_path}")

    if all_construct_records:
        construct_path = write_micro_tie_by_construct_csv(
            all_construct_records, survey_dir)
        if construct_path:
            print(f"  Wrote {construct_path}")

    if all_bridge_rank_records:
        bridge_metrics_path = write_bridge_rank_metrics_csv(
            all_bridge_rank_records, survey_dir)
        if bridge_metrics_path:
            print(f"  Wrote {bridge_metrics_path}")

    if all_community_group_records:
        cg_metrics_path = write_community_group_metrics_csv(
            all_community_group_records, survey_dir)
        if cg_metrics_path:
            print(f"  Wrote {cg_metrics_path}")

    # Compare consecutive waves
    if len(loaded) >= 2:
        w_a, rows_a, edges_a = loaded[0]
        w_b, rows_b, edges_b = loaded[-1]
        compare_waves(w_a, rows_a, edges_a, w_b, rows_b, edges_b)

    print(f"\n{'=' * 70}")
    print("  Analysis complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
