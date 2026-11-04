"""Versioned CSS scores; leave the frozen pilot report and old probabilities intact."""
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import css_cumulative_survey as css
import pilot_report as legacy
from pilot_state import atomic_json

SCORING_VERSION = "css-binary-uncertainty-v1"
LEGACY_VERSIONS = {"recent-network-v1", "cumulative-connection-v1"}
BINARY_METRICS = ("tp", "fp", "fn", "tn", "accuracy", "precision", "recall", "fpr", "fnr",
                  "balanced_accuracy", "f1", "jaccard", "confident_error_rate", "uncertain_error_rate",
                  "perceived_edges", "density_error", "network_degree_tau_b", "network_broker_tau_b",
                  "network_community_ari", "network_community_nmi")


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def score_respondent(metadata, answer):
    version = metadata["survey_version"]
    if version in LEGACY_VERSIONS:
        return legacy.score_respondent(metadata, answer)
    if version != css.SURVEY_VERSION or answer.get("survey_version") != version:
        raise ValueError("Unsupported or mixed CSS survey versions")
    css.validate_pair_rows(answer, metadata["roster"])
    row = legacy.score_respondent(metadata, answer)
    # For binary judgments Brier is the error fraction, not a probability-calibration score.
    row["classification_error"] = row.pop("brier")
    row.update(survey_version=version, scoring_version=SCORING_VERSION)
    row.update({key: None for key in BINARY_METRICS})
    valid = [(a, b, tie, flag[2]) for (a, b, tie), flag in zip(
        answer["interaction"], answer["interaction_uncertainty"]) if tie is not None]
    row.update(pairs_evaluated=len(valid), pair_coverage=ratio(len(valid), len(answer["interaction"])),
               uncertain_pairs=sum(flag for _, _, _, flag in valid),
               confident_pairs=sum(not flag for _, _, _, flag in valid))
    row["uncertain_fraction"] = ratio(row["uncertain_pairs"], len(valid))
    # Preserve the pilot rule: collect baseline judgments but do not score initial accuracy.
    if metadata["initial"]:
        return row
    truth_edges = {frozenset((a, b)) for a, b, _ in metadata["truth_edges"]}
    counts = {key: 0 for key in ("tp", "fp", "fn", "tn")}
    errors = {False: [], True: []}
    for a, b, tie, flag in valid:
        actual = int(frozenset((a, b)) in truth_edges)
        counts[{(1, 1): "tp", (1, 0): "fp", (0, 1): "fn", (0, 0): "tn"}[tie, actual]] += 1
        errors[flag].append(int(tie != actual))
    row.update(counts)
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    row.update(accuracy=ratio(tp + tn, len(valid)), classification_error=ratio(fp + fn, len(valid)),
               precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn), fpr=ratio(fp, fp + tn),
               fnr=ratio(fn, fn + tp), f1=ratio(2 * tp, 2 * tp + fp + fn),
               jaccard=ratio(tp, tp + fp + fn),
               confident_error_rate=ratio(sum(errors[False]), len(errors[False])),
               uncertain_error_rate=ratio(sum(errors[True]), len(errors[True])))
    specificity = ratio(tn, tn + fp)
    if row["recall"] is not None and specificity is not None:
        row["balanced_accuracy"] = (row["recall"] + specificity) / 2
    # A partial response is not a complete network. Never turn failed items into absent edges.
    if row["interaction_missing"]:
        return row
    import networkx as nx
    from scipy.stats import kendalltau
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    roster = metadata["roster"]
    truth, perceived = nx.Graph(), nx.Graph()
    for graph in (truth, perceived):
        graph.add_nodes_from(roster)
    truth.add_edges_from((a, b) for a, b, _ in metadata["truth_edges"])
    perceived.add_edges_from((a, b) for a, b, tie, _ in valid if tie == 1)
    row["perceived_edges"] = perceived.number_of_edges()
    row["density_error"] = nx.density(perceived) - nx.density(truth)
    for key, measure in (("network_degree_tau_b", lambda g: dict(g.degree())),
                         ("network_broker_tau_b", lambda g: nx.betweenness_centrality(g, weight=None))):
        actual, predicted = measure(truth), measure(perceived)
        if len(set(actual.values())) > 1 and len(set(predicted.values())) > 1:
            tau = float(kendalltau([actual[n] for n in roster], [predicted[n] for n in roster]).statistic)
            row[key] = tau if math.isfinite(tau) else None

    def partition(graph):
        groups = (nx.community.louvain_communities(graph, weight=None, seed=42)
                  if graph.number_of_edges() else [{n} for n in roster])
        labels = {name: i for i, group in enumerate(groups) for name in group}
        return [labels[n] for n in roster]

    actual, predicted = partition(truth), partition(perceived)
    row["network_community_ari"] = float(adjusted_rand_score(actual, predicted))
    row["network_community_nmi"] = float(normalized_mutual_info_score(actual, predicted))
    return row


def write_report(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    version = manifest.get("survey_version", manifest.get("config", {}).get("survey_version"))
    if version in LEGACY_VERSIONS:
        return legacy.write_report(run_dir)
    if version != css.SURVEY_VERSION:
        raise ValueError("Unsupported CSS survey version")
    rows = []
    for marker in sorted((run_dir / "surveys").glob("step_*/repeat_*/complete.json")):
        answers = css.read_survey_results(marker.parent, require_complete=True)
        metadata = json.loads((marker.parent / "metadata.json").read_text())
        rows.extend(score_respondent(metadata, answer) for answer in answers.values())
    # Reuse frozen cost accounting; replace its probability scores with the versioned results.
    report = legacy.write_report(run_dir)
    if rows:
        with (run_dir / "perception_scores.csv").open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    metrics = ("classification_error", "accuracy", "precision", "recall", "fpr", "fnr", "f1",
               "balanced_accuracy", "jaccard", "pair_coverage", "uncertain_fraction",
               "confident_error_rate", "uncertain_error_rate", "density_error",
               "network_degree_tau_b", "network_broker_tau_b", "network_community_ari",
               "network_community_nmi", "degree_tau_b", "broker_tau_b", "community_ari", "community_nmi")
    per_wave = defaultdict(list)
    for row in rows:
        for metric in metrics:
            if row[metric] is not None:
                per_wave[row["step"], row["repeat"], metric].append(row[metric])
    repeated = defaultdict(list)
    for (step, repeat, metric), scores in per_wave.items():
        repeated[step, metric].append(statistics.mean(scores))
    report.update(survey_version=version, scoring_version=SCORING_VERSION,
                  truth_definition=css.TRUTH_DEFINITION,
                  measurement_noise=[{"step": step, "metric": metric, "repeats": len(values),
                                      "mean": statistics.mean(values),
                                      "repeat_sd": statistics.stdev(values) if len(values) > 1 else None}
                                     for (step, metric), values in sorted(repeated.items())])
    atomic_json(run_dir / "report.json", report)
    report_path = run_dir / "REPORT.md"
    text = report_path.read_text(encoding="utf-8").replace(
        "Four instruments use the preceding two hours.",
        f"Four instruments ask about connections without a time window ({version}). "
        "Pair answers are binary judgments plus separate self-reported uncertainty. "
        "Truth is the cumulative observed conversation network; general social ties can differ from it.")
    text += ("\n## Binary judgments and uncertainty\n\n"
             "interaction stores [person_a, person_b, tie]; interaction_uncertainty stores "
             "[person_a, person_b, uncertain]. A failed batch has null in both fields. "
             "Uncertainty is not a probability, an edge weight, or missing data.\n\n"
             "Pair metrics include all valid judgments, including uncertain ones; always read them "
             "with pair_coverage. Zero denominators produce missing scores. classification_error "
             "replaces brier for this binary instrument. uncertain_fraction uses valid pairs as its "
             "denominator; confident_error_rate and uncertain_error_rate use their respective subsets.\n\n"
             "network_* scores are derived from the binary adjacency matrix, only when all pairs "
             "are valid. Both networks are undirected and unweighted, with all roster members; "
             "community comparisons use Louvain (seed 42), treating an empty graph as singleton groups. "
             "density_error is perceived minus true density. degree_tau_b, broker_tau_b, community_ari "
             "and community_nmi retain the separately elicited rankings/partition; their community "
             "reference remains the legacy conversation-count-weighted Louvain partition.\n\n"
             "Wave summaries average available respondent scores, then summarize across repeats. "
             "Uncertainty is self-reported, not calibrated confidence. Initial-state accuracy and "
             "structure scores remain missing by design; coverage and uncertainty are still recorded.\n")
    report_path.write_text(text, encoding="utf-8")
    return report
