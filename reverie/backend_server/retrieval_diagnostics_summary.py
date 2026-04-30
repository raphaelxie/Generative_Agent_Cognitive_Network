"""
Summarize retrieval diagnostics for wave t600.

Usage:
  python retrieval_diagnostics_summary.py <survey_dir>
"""
import json
import os
import sys
from collections import defaultdict


def _safe_pct(num, den):
    return (float(num) / float(den) * 100.0) if den else 0.0


def _safe_mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _entry_metrics(entry):
    respondent = entry.get("respondent", "")
    target = entry.get("query_target_person", "")
    nodes = entry.get("nodes", []) or []
    n_returned = int(entry.get("n_returned", 0) or 0)

    target_mentions = sum(1 for n in nodes if bool(n.get("mentions_query_target")))
    all_self = bool(nodes) and all(bool(n.get("is_self_mention")) for n in nodes)
    any_target = target_mentions > 0
    frac_target = (target_mentions / n_returned) if n_returned else 0.0
    non_self_query = (respondent != target and target != "")
    return {
        "respondent": respondent,
        "non_self_query": non_self_query,
        "any_target": any_target,
        "all_self": all_self,
        "frac_target": frac_target,
    }


def _aggregate(metrics):
    n = len(metrics)
    any_target_n = sum(1 for m in metrics if m["any_target"])
    all_self_n = sum(1 for m in metrics if m["all_self"])
    frac_mean = _safe_mean([m["frac_target"] for m in metrics])
    return {
        "n_calls": n,
        "pct_calls_with_any_target_mention": round(_safe_pct(any_target_n, n), 2),
        "pct_calls_all_self_mention": round(_safe_pct(all_self_n, n), 2),
        "mean_target_mention_fraction": round(frac_mean, 4),
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python retrieval_diagnostics_summary.py <survey_dir>")

    survey_dir = os.path.abspath(sys.argv[1])
    in_path = os.path.join(survey_dir, "retrieval_diagnostics_t600.jsonl")
    out_path = os.path.join(survey_dir, "retrieval_diagnostics_t600_summary.json")

    entries = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    metrics = [_entry_metrics(e) for e in entries]
    non_self = [m for m in metrics if m["non_self_query"]]

    per_resp_bucket = defaultdict(list)
    for m in non_self:
        per_resp_bucket[m["respondent"]].append(m)

    per_respondent = {}
    for resp in sorted(per_resp_bucket.keys()):
        per_respondent[resp] = _aggregate(per_resp_bucket[resp])

    result = {
        "wave_id": "t600",
        "global_excluding_self_queries": _aggregate(non_self),
        "per_respondent_excluding_self_queries": per_respondent,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
