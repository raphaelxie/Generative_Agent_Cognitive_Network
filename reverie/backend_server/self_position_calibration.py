"""
Compute self-position calibration for wave t600.

Usage:
  python self_position_calibration.py <survey_dir>
"""
import csv
import os
import sys
from collections import defaultdict


def _load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rank_desc(values_by_name):
    ordered = sorted(values_by_name.items(), key=lambda kv: (-kv[1], kv[0]))
    out = {}
    rank = 1
    for name, val in ordered:
        out[name] = rank
        rank += 1
    return out


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python self_position_calibration.py <survey_dir>")

    survey_dir = os.path.abspath(sys.argv[1])
    survey_csv = os.path.join(survey_dir, "perception_survey_t600.csv")
    edges_csv = os.path.join(survey_dir, "ground_truth", "ground_truth_edges_600.csv")
    out_csv = os.path.join(survey_dir, "self_position_calibration_t600.csv")

    survey_rows = _load_csv(survey_csv)
    edge_rows = _load_csv(edges_csv)

    degree = defaultdict(int)
    personas = set()
    for r in edge_rows:
        a, b = r["node_a"], r["node_b"]
        personas.add(a)
        personas.add(b)
        if int(r["tie_cumulative"]) == 1:
            degree[a] += 1
            degree[b] += 1
    for p in personas:
        degree[p] += 0

    gt_rank = _rank_desc(dict(degree))

    self_rows = [r for r in survey_rows if r.get("question_type") == "self_position"]
    if not self_rows:
        raise SystemExit("No self_position rows found in perception_survey_t600.csv")

    self_values = {}
    for r in self_rows:
        name = r["respondent"]
        val = r.get("value", "").strip()
        try:
            self_values[name] = int(val)
        except Exception:
            self_values[name] = 0
    self_rank = _rank_desc(self_values)

    out_rows = []
    for r in sorted(self_rows, key=lambda x: x["respondent"]):
        name = r["respondent"]
        self_val = self_values.get(name, 0)
        row = {
            "persona": name,
            "self_perceived_value": self_val,
            "self_perceived_rank": self_rank.get(name, ""),
            "gt_degree": degree.get(name, 0),
            "gt_degree_rank": gt_rank.get(name, ""),
            "signed_error": (self_rank.get(name, 0) - gt_rank.get(name, 0)),
            "absolute_error": abs(self_rank.get(name, 0) - gt_rank.get(name, 0)),
            "is_fail_safe": int(r.get("is_fail_safe", "0") or 0),
        }
        out_rows.append(row)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
