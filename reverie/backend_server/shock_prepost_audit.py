"""
Shock-aligned pre/post audit for one existing survey directory.

This script is analysis-only: it reads already-generated survey outputs,
aligns the nearest pre/post survey waves around shock_log.jsonl, and writes
compact evidence artifacts. It does not load ReverieServer, call an LLM, add
prompts, or change simulation state.

Usage:
  python reverie/backend_server/shock_prepost_audit.py <survey_dir>
"""
import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict

_ORIG_CWD = os.getcwd()
from ground_truth_log import graph_scope_summary, target_graph_summary
os.chdir(_ORIG_CWD)


TRUTH_LAYERS = [
    "observed_interaction",
    "background_social_tie",
    "background_or_interaction",
]

CONSTRUCTS = [
    "micro_tie_interaction",
    "micro_tie_social",
    "micro_tie_group",
]

CONSTRUCT_NATURAL_TRUTH = {
    "micro_tie_interaction": "observed_interaction",
    "micro_tie_social":      "background_social_tie",
    "micro_tie_group":       "background_or_interaction",
}

MICRO_METRICS = [
    "tp", "fp", "fn", "tn",
    "precision", "recall", "fpr", "fnr", "pair_accuracy",
    "n_fail_safe", "n_pairs_evaluated",
]

CONSTRUCT_METRICS = [
    "precision", "recall", "fpr", "fnr", "pair_accuracy",
    "n_pairs", "n_fail_safe",
]

GRAPH_SCOPES = [
    ("cumulative", "count_cumulative"),
    ("recent_8h", "count_recent"),
]

BROKER_GRAPH_TEXT_FIELDS = [
    "top3_degree", "top3_betweenness", "hub", "broker",
]

BROKER_GRAPH_NUMERIC_FIELDS = [
    "n_nodes",
    "n_edges",
    "density",
    "mean_degree",
    "max_degree",
    "max_betweenness",
    "hub_eq_broker",
    "betweenness_top_share",
    "betweenness_hhi",
    "component_count",
    "largest_component_size",
    "reachable_pair_fraction",
    "mean_reachable_shortest_path_length",
    "diameter_reachable",
    "bridge_edge_count",
    "target_degree",
    "target_betweenness",
    "target_betweenness_rank",
    "target_incident_edges",
    "target_incident_bridge_edges",
]


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _safe_int(value, default=0):
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def _safe_float(value, default=None):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _mean(values):
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _median(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def _delta(post, pre):
    if post is None or pre is None:
        return None
    if not isinstance(post, (int, float)) or not isinstance(pre, (int, float)):
        return None
    return post - pre


def _wave_sort_key(wave):
    return _safe_int(wave["step"])


def discover_waves(survey_dir):
    waves = []
    for fname in sorted(os.listdir(survey_dir)):
        if not fname.startswith("perception_survey_") or not fname.endswith(".csv"):
            continue
        wave_id = fname[len("perception_survey_"):-len(".csv")]
        path = os.path.join(survey_dir, fname)
        rows = _read_csv(path)
        if not rows:
            continue
        waves.append({
            "wave_id": wave_id,
            "step": _safe_int(rows[0].get("step")),
            "sim_time": rows[0].get("sim_time", ""),
            "path": path,
            "rows": rows,
        })
    waves.sort(key=_wave_sort_key)
    return waves


def load_shock(survey_dir):
    path = os.path.join(survey_dir, "shock_log.jsonl")
    if not os.path.exists(path):
        return None, path
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("treatment_type") in ("hub_removal", "broker_removal"):
                return entry, path
    return None, path


def align_pre_post(waves, shock):
    if not shock:
        return None, None, "missing_shock"
    shock_step = _safe_int(shock.get("step"))
    pre = [w for w in waves if w["step"] <= shock_step]
    post = [w for w in waves if w["step"] > shock_step]
    if not pre and not post:
        return None, None, "no_survey_waves"
    if not pre:
        return None, post[0] if post else None, "missing_pre_wave"
    if not post:
        return pre[-1], None, "missing_post_wave"
    return pre[-1], post[0], "ok"


def load_network_summary(survey_dir):
    path = os.path.join(survey_dir, "network_summary_over_time.csv")
    if not os.path.exists(path):
        return {}, path
    rows = _read_csv(path)
    return {_safe_int(r.get("step")): r for r in rows}, path


def load_edges_for_step(survey_dir, step):
    path = os.path.join(survey_dir, "ground_truth", f"ground_truth_edges_{step}.csv")
    if not os.path.exists(path):
        return [], path
    return _read_csv(path), path


def degree_ranks(edge_rows):
    degree = defaultdict(int)
    names = set()
    for row in edge_rows:
        a, b = row["node_a"], row["node_b"]
        names.add(a)
        names.add(b)
        if _safe_int(row.get("tie_cumulative")) == 1:
            degree[a] += 1
            degree[b] += 1
    for name in names:
        degree[name] += 0
    ordered = sorted(names, key=lambda n: (-degree[n], n))
    ranks = {name: i + 1 for i, name in enumerate(ordered)}
    return dict(degree), ranks, ordered


def load_micro_by_truth(survey_dir, pre_wave, post_wave, shock, alignment_status):
    path = os.path.join(survey_dir, "analysis_micro_tie_metrics_by_truth.csv")
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {path}; run background_social_truth.py and analyze_survey.py first"
        )
    rows = _read_csv(path)
    out = []
    for layer in TRUTH_LAYERS:
        pre = _overall_truth_row(rows, pre_wave, layer) if pre_wave else None
        post = _overall_truth_row(rows, post_wave, layer) if post_wave else None
        row = _base_row(survey_dir, shock, pre_wave, post_wave, alignment_status)
        row["truth_layer"] = layer
        for metric in MICRO_METRICS:
            pre_val = _metric_value(pre, metric)
            post_val = _metric_value(post, metric)
            row[f"pre_{metric}"] = _fmt(pre_val)
            row[f"post_{metric}"] = _fmt(post_val)
            row[f"delta_{metric}"] = _fmt(_delta(post_val, pre_val))
        out.append(row)
    return out


def load_micro_by_construct(survey_dir, pre_wave, post_wave,
                            shock, alignment_status):
    """Load per-construct x per-truth-layer pre/post deltas.

    Reads analysis_micro_tie_by_construct.csv produced by analyze_survey.py.
    Returns [] gracefully if the file does not exist (no separated constructs
    were collected for this experiment).
    """
    path = os.path.join(survey_dir, "analysis_micro_tie_by_construct.csv")
    if not os.path.exists(path):
        return []
    rows = _read_csv(path)
    out = []
    for construct in CONSTRUCTS:
        for layer in TRUTH_LAYERS:
            pre = (_overall_construct_row(rows, pre_wave, construct, layer)
                   if pre_wave else None)
            post = (_overall_construct_row(rows, post_wave, construct, layer)
                    if post_wave else None)
            if pre is None and post is None:
                continue
            row = _base_row(survey_dir, shock, pre_wave, post_wave,
                            alignment_status)
            row["construct"] = construct
            row["truth_layer"] = layer
            row["is_natural_match"] = int(
                CONSTRUCT_NATURAL_TRUTH.get(construct) == layer)
            for metric in CONSTRUCT_METRICS:
                pre_val = _metric_value(pre, metric)
                post_val = _metric_value(post, metric)
                row[f"pre_{metric}"] = _fmt(pre_val)
                row[f"post_{metric}"] = _fmt(post_val)
                row[f"delta_{metric}"] = _fmt(_delta(post_val, pre_val))
            out.append(row)
    return out


def _overall_construct_row(rows, wave, construct, layer):
    for row in rows:
        if (row.get("wave_id") == wave["wave_id"]
                and row.get("construct") == construct
                and row.get("truth_layer") == layer
                and row.get("respondent") == "__overall__"):
            return row
    return None


def _overall_truth_row(rows, wave, layer):
    for row in rows:
        if (row.get("wave_id") == wave["wave_id"]
                and row.get("truth_layer") == layer
                and row.get("respondent") == "__overall__"):
            return row
    return None


def _metric_value(row, metric):
    if row is None:
        return None
    if metric in ("tp", "fp", "fn", "tn", "n_fail_safe", "n_pairs_evaluated", "n_pairs"):
        return _safe_int(row.get(metric), default=None)
    return _safe_float(row.get(metric))


def _base_row(survey_dir, shock, pre_wave, post_wave, alignment_status):
    return {
        "survey_dir": os.path.basename(os.path.dirname(survey_dir)),
        "treatment_type": shock.get("treatment_type", "") if shock else "",
        "target_agent": shock.get("target_agent", "") if shock else "",
        "shock_step": shock.get("step", "") if shock else "",
        "alignment_status": alignment_status,
        "pre_wave": pre_wave["wave_id"] if pre_wave else "",
        "pre_step": pre_wave["step"] if pre_wave else "",
        "post_wave": post_wave["wave_id"] if post_wave else "",
        "post_step": post_wave["step"] if post_wave else "",
    }


def centrality_self_position_rows(survey_dir, pre_wave, post_wave,
                                  shock, alignment_status):
    rows = []
    wave_summaries = {}
    for phase, wave in (("pre", pre_wave), ("post", post_wave)):
        if not wave:
            continue
        summary = summarize_centrality_self_position(survey_dir, wave)
        wave_summaries[phase] = summary
        row = _base_row(survey_dir, shock, pre_wave, post_wave, alignment_status)
        row.update({"phase": phase, "wave_id": wave["wave_id"], "step": wave["step"]})
        row.update({k: _fmt(v) for k, v in summary.items()})
        rows.append(row)

    if "pre" in wave_summaries and "post" in wave_summaries:
        row = _base_row(survey_dir, shock, pre_wave, post_wave, alignment_status)
        row.update({"phase": "delta", "wave_id": "", "step": ""})
        for key in wave_summaries["pre"]:
            row[key] = _fmt(_delta(wave_summaries["post"].get(key),
                                   wave_summaries["pre"].get(key)))
        rows.append(row)
    return rows


def summarize_centrality_self_position(survey_dir, wave):
    edge_rows, _ = load_edges_for_step(survey_dir, wave["step"])
    degree, actual_rank, actual_order = degree_ranks(edge_rows)
    actual_top = actual_order[0] if actual_order else ""

    centrality_by_resp = defaultdict(list)
    self_values = {}
    n_centrality_fs = 0
    n_self_fs = 0
    for row in wave["rows"]:
        qtype = row.get("question_type")
        if qtype == "centrality_rank":
            centrality_by_resp[row["respondent"]].append(row)
            if row.get("is_fail_safe") == "1":
                n_centrality_fs += 1
        elif qtype == "self_position":
            if row.get("is_fail_safe") == "1":
                n_self_fs += 1
            self_values[row["respondent"]] = _safe_int(row.get("value"), default=None)

    respondent_rank_errors = []
    actual_top_ranks = []
    top1_hits = []
    top3_hits = []
    for resp, rank_rows in centrality_by_resp.items():
        order = [
            r["target"]
            for r in sorted(rank_rows, key=lambda x: _safe_int(x.get("value"), 999))
        ]
        if not order:
            continue
        perceived_rank = {name: i + 1 for i, name in enumerate(order)}
        errors = [
            abs(perceived_rank[name] - actual_rank[name])
            for name in actual_rank
            if name in perceived_rank
        ]
        respondent_rank_errors.append(_mean(errors))
        if actual_top in perceived_rank:
            actual_top_ranks.append(perceived_rank[actual_top])
            top1_hits.append(1 if perceived_rank[actual_top] == 1 else 0)
            top3_hits.append(1 if perceived_rank[actual_top] <= 3 else 0)

    self_rank = {
        name: i + 1
        for i, (name, _) in enumerate(sorted(
            self_values.items(), key=lambda kv: (-(kv[1] or 0), kv[0])))
    }
    self_rank_errors = [
        abs(self_rank[name] - actual_rank[name])
        for name in self_rank
        if name in actual_rank
    ]

    return {
        "actual_top_degree_agent": actual_top,
        "actual_top_degree": degree.get(actual_top, 0) if actual_top else None,
        "centrality_mean_abs_rank_error": _mean(respondent_rank_errors),
        "centrality_actual_top_mean_perceived_rank": _mean(actual_top_ranks),
        "centrality_top1_hit_rate": _mean(top1_hits),
        "centrality_top3_hit_rate": _mean(top3_hits),
        "centrality_fail_safe_rows": n_centrality_fs,
        "self_position_mean_value": _mean(list(self_values.values())),
        "self_position_mean_abs_rank_error": _mean(self_rank_errors),
        "self_position_fail_safe_rows": n_self_fs,
    }


def retrieval_quality_rows(survey_dir, pre_wave, post_wave, shock, alignment_status):
    rows = []
    summaries = {}
    for phase, wave in (("pre", pre_wave), ("post", post_wave)):
        if not wave:
            continue
        summary = summarize_retrieval_quality(survey_dir, wave)
        summaries[phase] = summary
        row = _base_row(survey_dir, shock, pre_wave, post_wave, alignment_status)
        row.update({"phase": phase, "wave_id": wave["wave_id"], "step": wave["step"]})
        row.update({k: _fmt(v) for k, v in summary.items()})
        rows.append(row)

    if "pre" in summaries and "post" in summaries:
        row = _base_row(survey_dir, shock, pre_wave, post_wave, alignment_status)
        row.update({"phase": "delta", "wave_id": "", "step": ""})
        for key in summaries["pre"]:
            row[key] = _fmt(_delta(summaries["post"].get(key),
                                   summaries["pre"].get(key)))
        rows.append(row)
    return rows


def summarize_retrieval_quality(survey_dir, wave):
    by_type = defaultdict(lambda: {"n": 0, "fs": 0})
    for row in wave["rows"]:
        qtype = row.get("question_type", "")
        by_type[qtype]["n"] += 1
        if row.get("is_fail_safe") == "1":
            by_type[qtype]["fs"] += 1

    diag_path = os.path.join(
        survey_dir, f"retrieval_diagnostics_{wave['wave_id']}.jsonl")
    diags = []
    if os.path.exists(diag_path):
        with open(diag_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    diags.append(json.loads(line))

    n_candidates = [_safe_int(d.get("n_candidates"), None) for d in diags]
    n_returned = [_safe_int(d.get("n_returned"), None) for d in diags]
    micro_diags = [d for d in diags if d.get("query_target_person")]
    mention_hits = []
    for diag in micro_diags:
        nodes = diag.get("nodes") or []
        mention_hits.append(
            1 if any(n.get("mentions_query_target") is True for n in nodes) else 0
        )

    def rate(qtype):
        n = by_type[qtype]["n"]
        return (by_type[qtype]["fs"] / n) if n else None

    zero_return = [
        1 if d.get("n_returned") == 0 else 0
        for d in diags
    ]

    result = {
        "n_rows": len(wave["rows"]),
        "n_micro_tie_rows": by_type["micro_tie"]["n"],
        "micro_tie_fail_safe_rate": rate("micro_tie"),
        "n_micro_tie_interaction_rows": by_type["micro_tie_interaction"]["n"],
        "micro_tie_interaction_fail_safe_rate": rate("micro_tie_interaction"),
        "n_micro_tie_social_rows": by_type["micro_tie_social"]["n"],
        "micro_tie_social_fail_safe_rate": rate("micro_tie_social"),
        "n_micro_tie_group_rows": by_type["micro_tie_group"]["n"],
        "micro_tie_group_fail_safe_rate": rate("micro_tie_group"),
        "n_centrality_rank_rows": by_type["centrality_rank"]["n"],
        "centrality_rank_fail_safe_rate": rate("centrality_rank"),
        "n_self_position_rows": by_type["self_position"]["n"],
        "self_position_fail_safe_rate": rate("self_position"),
        "retrieval_calls": len(diags),
        "mean_n_candidates": _mean(n_candidates),
        "median_n_candidates": _median(n_candidates),
        "mean_n_returned": _mean(n_returned),
        "median_n_returned": _median(n_returned),
        "zero_return_rate": _mean(zero_return),
        "micro_query_target_mention_rate": _mean(mention_hits),
    }
    return result


def broker_graph_diagnostic_rows(survey_dir, pre_wave, post_wave,
                                 shock, alignment_status):
    """Build graph-only broker diagnostics without changing observed graphs.

    Rows with analysis_view="analysis_time_target_excluded" are counterfactual
    analysis-time views that remove the shock target from the graph calculation;
    they are not observed post-shock graph states.
    """
    target = shock.get("target_agent", "") if shock else ""
    rows = []
    summaries = {}
    for phase, wave in (("pre", pre_wave), ("post", post_wave)):
        if not wave:
            continue
        edge_rows, _ = load_edges_for_step(survey_dir, wave["step"])
        for scope_label, count_field in GRAPH_SCOPES:
            for analysis_view, excluded_nodes in (
                ("observed_target_included", None),
                ("analysis_time_target_excluded", [target] if target else None),
            ):
                is_excluded = analysis_view == "analysis_time_target_excluded"
                summary = graph_scope_summary(
                    edge_rows, count_field, excluded_nodes=excluded_nodes)
                target_summary = (
                    {} if is_excluded
                    else target_graph_summary(edge_rows, target, count_field)
                )
                raw = {}
                raw.update(summary)
                raw.update(target_summary)
                summaries[(phase, scope_label, analysis_view)] = raw

                row = _base_row(
                    survey_dir, shock, pre_wave, post_wave, alignment_status)
                row.update({
                    "phase": phase,
                    "wave_id": wave["wave_id"],
                    "step": wave["step"],
                    "graph_scope": scope_label,
                    "edge_count_field": count_field,
                    "analysis_view": analysis_view,
                    "analysis_time_target_excluded": int(is_excluded),
                    "target_excluded_agent": target if is_excluded else "",
                })
                _copy_broker_graph_fields(row, raw)
                rows.append(row)

    if pre_wave and post_wave:
        for scope_label, _ in GRAPH_SCOPES:
            for analysis_view in (
                "observed_target_included",
                "analysis_time_target_excluded",
            ):
                pre = summaries.get(("pre", scope_label, analysis_view), {})
                post = summaries.get(("post", scope_label, analysis_view), {})
                if not pre or not post:
                    continue
                is_excluded = analysis_view == "analysis_time_target_excluded"
                row = _base_row(
                    survey_dir, shock, pre_wave, post_wave, alignment_status)
                row.update({
                    "phase": "delta",
                    "wave_id": "",
                    "step": "",
                    "graph_scope": scope_label,
                    "edge_count_field": dict(GRAPH_SCOPES).get(scope_label, ""),
                    "analysis_view": analysis_view,
                    "analysis_time_target_excluded": int(is_excluded),
                    "target_excluded_agent": target if is_excluded else "",
                })
                for field in BROKER_GRAPH_TEXT_FIELDS:
                    row[field] = ""
                for field in BROKER_GRAPH_NUMERIC_FIELDS:
                    row[field] = _fmt(_delta(
                        _numeric_broker_graph_value(post.get(field)),
                        _numeric_broker_graph_value(pre.get(field))))
                rows.append(row)
    return rows


def _copy_broker_graph_fields(row, raw):
    for field in BROKER_GRAPH_TEXT_FIELDS:
        row[field] = raw.get(field, "")
    for field in BROKER_GRAPH_NUMERIC_FIELDS:
        row[field] = _fmt(_numeric_broker_graph_value(raw.get(field)))


def _numeric_broker_graph_value(value):
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    return _safe_float(value)


def prepost_summary_row(survey_dir, shock, pre_wave, post_wave,
                        alignment_status, network_rows, micro_rows,
                        centrality_rows, retrieval_rows,
                        construct_rows=None):
    row = _base_row(survey_dir, shock, pre_wave, post_wave, alignment_status)

    pre_net = network_rows.get(pre_wave["step"]) if pre_wave else None
    post_net = network_rows.get(post_wave["step"]) if post_wave else None
    for key in [
        "n_edges_cumulative",
        "density_cumulative",
        "mean_degree_cumulative",
        "max_degree",
        "max_betweenness",
        "hub_eq_broker",
    ]:
        pre_val = _summary_value(pre_net, key)
        post_val = _summary_value(post_net, key)
        row[f"pre_{key}"] = _fmt(pre_val)
        row[f"post_{key}"] = _fmt(post_val)
        row[f"delta_{key}"] = _fmt(_delta(post_val, pre_val))
    row["pre_hub"] = pre_net.get("hub", "") if pre_net else ""
    row["post_hub"] = post_net.get("hub", "") if post_net else ""
    row["pre_broker"] = pre_net.get("broker", "") if pre_net else ""
    row["post_broker"] = post_net.get("broker", "") if post_net else ""

    union_micro = next(
        (r for r in micro_rows if r.get("truth_layer") == "background_or_interaction"),
        None,
    )
    if union_micro:
        for key in ("precision", "recall", "fpr", "pair_accuracy"):
            row[f"union_pre_{key}"] = union_micro.get(f"pre_{key}", "")
            row[f"union_post_{key}"] = union_micro.get(f"post_{key}", "")
            row[f"union_delta_{key}"] = union_micro.get(f"delta_{key}", "")

    for construct, natural_layer in CONSTRUCT_NATURAL_TRUTH.items():
        match = next(
            (r for r in (construct_rows or [])
             if r.get("construct") == construct
             and r.get("truth_layer") == natural_layer),
            None,
        )
        short = construct.replace("micro_tie_", "")
        for key in ("precision", "recall", "fpr", "pair_accuracy"):
            row[f"{short}_pre_{key}"] = match.get(f"pre_{key}", "") if match else ""
            row[f"{short}_post_{key}"] = match.get(f"post_{key}", "") if match else ""
            row[f"{short}_delta_{key}"] = match.get(f"delta_{key}", "") if match else ""

    centrality_pre = next((r for r in centrality_rows if r.get("phase") == "pre"), {})
    centrality_post = next((r for r in centrality_rows if r.get("phase") == "post"), {})
    for key in ("centrality_top1_hit_rate", "centrality_top3_hit_rate",
                "self_position_mean_abs_rank_error"):
        pre_val = _safe_float(centrality_pre.get(key))
        post_val = _safe_float(centrality_post.get(key))
        row[f"pre_{key}"] = _fmt(pre_val)
        row[f"post_{key}"] = _fmt(post_val)
        row[f"delta_{key}"] = _fmt(_delta(post_val, pre_val))

    retrieval_pre = next((r for r in retrieval_rows if r.get("phase") == "pre"), {})
    retrieval_post = next((r for r in retrieval_rows if r.get("phase") == "post"), {})
    fs_rate_keys = ["micro_tie_fail_safe_rate", "zero_return_rate",
                    "micro_query_target_mention_rate"]
    for c in CONSTRUCTS:
        fs_rate_keys.append(f"{c}_fail_safe_rate")
    for key in fs_rate_keys:
        pre_val = _safe_float(retrieval_pre.get(key))
        post_val = _safe_float(retrieval_post.get(key))
        row[f"pre_{key}"] = _fmt(pre_val)
        row[f"post_{key}"] = _fmt(post_val)
        row[f"delta_{key}"] = _fmt(_delta(post_val, pre_val))

    return row


def _summary_value(row, key):
    if not row:
        return None
    if key in ("n_edges_cumulative", "max_degree", "hub_eq_broker"):
        return _safe_int(row.get(key), None)
    return _safe_float(row.get(key))


def write_evidence_note(path, summary_row, micro_rows,
                        centrality_rows, retrieval_rows,
                        construct_rows=None):
    lines = [
        "# Shock-Aligned Pre/Post Evidence Note",
        "",
        "## Alignment",
        "",
        f"- Treatment: `{summary_row.get('treatment_type', '')}`",
        f"- Target agent: `{summary_row.get('target_agent', '')}`",
        f"- Shock step: `{summary_row.get('shock_step', '')}`",
        f"- Pre wave: `{summary_row.get('pre_wave', '')}` at step `{summary_row.get('pre_step', '')}`",
        f"- Post wave: `{summary_row.get('post_wave', '')}` at step `{summary_row.get('post_step', '')}`",
        f"- Alignment status: `{summary_row.get('alignment_status', '')}`",
        "",
        "## Structural Check",
        "",
        f"- Edges: {summary_row.get('pre_n_edges_cumulative', '')} -> "
        f"{summary_row.get('post_n_edges_cumulative', '')} "
        f"(delta {summary_row.get('delta_n_edges_cumulative', '')})",
        f"- Density: {summary_row.get('pre_density_cumulative', '')} -> "
        f"{summary_row.get('post_density_cumulative', '')} "
        f"(delta {summary_row.get('delta_density_cumulative', '')})",
        f"- Hub: `{summary_row.get('pre_hub', '')}` -> `{summary_row.get('post_hub', '')}`",
        f"- Broker: `{summary_row.get('pre_broker', '')}` -> `{summary_row.get('post_broker', '')}`",
        "",
        "## Micro-Tie Summary",
        "",
    ]
    for row in micro_rows:
        lines.append(
            f"- `{row['truth_layer']}`: precision {row.get('pre_precision', '')} -> "
            f"{row.get('post_precision', '')}; recall {row.get('pre_recall', '')} -> "
            f"{row.get('post_recall', '')}; FPR {row.get('pre_fpr', '')} -> "
            f"{row.get('post_fpr', '')}."
        )

    if construct_rows:
        lines.extend(["", "## Micro-Tie by Construct", ""])
        for construct in CONSTRUCTS:
            natural = CONSTRUCT_NATURAL_TRUTH[construct]
            match = next(
                (r for r in construct_rows
                 if r.get("construct") == construct
                 and r.get("truth_layer") == natural),
                None,
            )
            short = construct.replace("micro_tie_", "")
            if match:
                lines.append(
                    f"- `{short}` (vs `{natural}`): "
                    f"precision {match.get('pre_precision', '')} -> "
                    f"{match.get('post_precision', '')}; "
                    f"recall {match.get('pre_recall', '')} -> "
                    f"{match.get('post_recall', '')}; "
                    f"FPR {match.get('pre_fpr', '')} -> "
                    f"{match.get('post_fpr', '')}."
                )
            else:
                lines.append(f"- `{short}`: no data available.")

    c_pre = next((r for r in centrality_rows if r.get("phase") == "pre"), {})
    c_post = next((r for r in centrality_rows if r.get("phase") == "post"), {})
    r_pre = next((r for r in retrieval_rows if r.get("phase") == "pre"), {})
    r_post = next((r for r in retrieval_rows if r.get("phase") == "post"), {})

    lines.extend([
        "",
        "## Centrality And Self-Position",
        "",
        f"- Centrality top-1 hit rate: {c_pre.get('centrality_top1_hit_rate', '')} -> "
        f"{c_post.get('centrality_top1_hit_rate', '')}",
        f"- Centrality top-3 hit rate: {c_pre.get('centrality_top3_hit_rate', '')} -> "
        f"{c_post.get('centrality_top3_hit_rate', '')}",
        f"- Self-position mean absolute rank error: "
        f"{c_pre.get('self_position_mean_abs_rank_error', '')} -> "
        f"{c_post.get('self_position_mean_abs_rank_error', '')}",
        "",
        "## Retrieval And Fail-Safe Context",
        "",
        f"- Micro-tie fail-safe rate (broad): "
        f"{r_pre.get('micro_tie_fail_safe_rate', '')} -> "
        f"{r_post.get('micro_tie_fail_safe_rate', '')}",
        f"- Micro-tie fail-safe rate (interaction): "
        f"{r_pre.get('micro_tie_interaction_fail_safe_rate', '')} -> "
        f"{r_post.get('micro_tie_interaction_fail_safe_rate', '')}",
        f"- Micro-tie fail-safe rate (social): "
        f"{r_pre.get('micro_tie_social_fail_safe_rate', '')} -> "
        f"{r_post.get('micro_tie_social_fail_safe_rate', '')}",
        f"- Micro-tie fail-safe rate (group): "
        f"{r_pre.get('micro_tie_group_fail_safe_rate', '')} -> "
        f"{r_post.get('micro_tie_group_fail_safe_rate', '')}",
        f"- Zero-return retrieval rate: {r_pre.get('zero_return_rate', '')} -> "
        f"{r_post.get('zero_return_rate', '')}",
        f"- Micro query-target mention rate: "
        f"{r_pre.get('micro_query_target_mention_rate', '')} -> "
        f"{r_post.get('micro_query_target_mention_rate', '')}",
        "",
        "## Interpretation Template",
        "",
        "This note is a standardized summary, not an automatic conclusion. "
        "Interpret the cognition deltas alongside the structural manipulation "
        "check and retrieval/fail-safe context. The `background_or_interaction` "
        "micro-tie layer is the least construct-mismatched comparison for the "
        "broad social-connection prompt. The separated constructs (interaction, "
        "social, group) each have a natural truth layer shown in the "
        "'Micro-Tie by Construct' section; off-diagonal cross-scoring is in "
        "shock_aligned_micro_tie_by_construct.csv. Centrality and self-position "
        "remain observed-interaction degree comparisons.",
        "",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_audit(survey_dir):
    survey_dir = os.path.abspath(survey_dir)
    if not os.path.isdir(survey_dir):
        raise SystemExit(f"survey dir not found: {survey_dir}")

    shock, _ = load_shock(survey_dir)
    waves = discover_waves(survey_dir)
    pre_wave, post_wave, alignment_status = align_pre_post(waves, shock)
    network_rows, _ = load_network_summary(survey_dir)

    micro_rows = load_micro_by_truth(
        survey_dir, pre_wave, post_wave, shock, alignment_status)
    construct_rows = load_micro_by_construct(
        survey_dir, pre_wave, post_wave, shock, alignment_status)
    centrality_rows = centrality_self_position_rows(
        survey_dir, pre_wave, post_wave, shock, alignment_status)
    retrieval_rows = retrieval_quality_rows(
        survey_dir, pre_wave, post_wave, shock, alignment_status)
    broker_graph_rows = broker_graph_diagnostic_rows(
        survey_dir, pre_wave, post_wave, shock or {}, alignment_status)
    summary = prepost_summary_row(
        survey_dir, shock or {}, pre_wave, post_wave, alignment_status,
        network_rows, micro_rows, centrality_rows, retrieval_rows,
        construct_rows=construct_rows)

    summary_path = os.path.join(survey_dir, "shock_aligned_prepost_summary.csv")
    micro_path = os.path.join(survey_dir, "shock_aligned_micro_tie_by_truth.csv")
    construct_path = os.path.join(
        survey_dir, "shock_aligned_micro_tie_by_construct.csv")
    centrality_path = os.path.join(
        survey_dir, "shock_aligned_centrality_self_position.csv")
    retrieval_path = os.path.join(survey_dir, "shock_aligned_retrieval_quality.csv")
    broker_graph_path = os.path.join(
        survey_dir, "shock_aligned_broker_graph_diagnostics.csv")
    note_path = os.path.join(survey_dir, "shock_aligned_evidence_note.md")

    _write_csv(summary_path, [summary], list(summary.keys()))
    _write_csv(micro_path, micro_rows, list(micro_rows[0].keys()))
    if construct_rows:
        _write_csv(construct_path, construct_rows,
                   list(construct_rows[0].keys()))
    _write_csv(centrality_path, centrality_rows, list(centrality_rows[0].keys()))
    _write_csv(retrieval_path, retrieval_rows, list(retrieval_rows[0].keys()))
    _write_csv(
        broker_graph_path, broker_graph_rows, list(broker_graph_rows[0].keys()))
    write_evidence_note(note_path, summary, micro_rows, centrality_rows,
                        retrieval_rows, construct_rows=construct_rows)

    print(f"Wrote {summary_path}")
    print(f"Wrote {micro_path}")
    if construct_rows:
        print(f"Wrote {construct_path}")
    print(f"Wrote {centrality_path}")
    print(f"Wrote {retrieval_path}")
    print(f"Wrote {broker_graph_path}")
    print(f"Wrote {note_path}")
    print(
        f"Aligned pre={pre_wave['wave_id'] if pre_wave else ''} "
        f"post={post_wave['wave_id'] if post_wave else ''} "
        f"status={alignment_status}"
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python shock_prepost_audit.py <survey_dir>"
        )
    run_audit(sys.argv[1])


if __name__ == "__main__":
    main()
