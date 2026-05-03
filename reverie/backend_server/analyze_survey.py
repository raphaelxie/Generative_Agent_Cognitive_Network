"""
Analyze perception survey results against ground truth.

Reads perception_survey_{wave}.csv and ground_truth_edges_{step}.csv,
prints a structured comparison showing what each respondent perceived,
what actually happened, and where perception matched or diverged.

Outputs:
  - Console report (per-respondent + overall per wave)
  - analysis_micro_tie_metrics.csv       (structured metrics)
  - analysis_micro_tie_metrics_README.md (self-describing sidecar)
  - analysis_micro_tie_metrics_by_truth.csv
  - analysis_micro_tie_metrics_by_truth_README.md

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
import os
import sys
from collections import defaultdict

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
                                          wave_id, step):
    """Return list of per-respondent confusion dicts plus one __overall__ row.

    Requires truth_labels to be present; returns [] if ground truth is missing.
    Requires survey_rows to have 'target_j' populated; callers should run
    reconstruct_target_j first for old-format CSVs.
    """
    if not truth_labels:
        return []

    by_resp = defaultdict(list)
    for r in survey_rows:
        if r.get("question_type") != "micro_tie":
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


def _fmt_ratio(v):
    return "   NA" if v is None else f"{v:.3f}"


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
