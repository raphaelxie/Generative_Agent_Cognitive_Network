"""
Build a t600-only micro-tie summary table with an OVERALL row.

Usage:
  python micro_tie_t600_summary.py <survey_dir>
"""
import csv
import os
import sys


def _fdiv(num, den):
    return float(num) / float(den) if den else 0.0


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python micro_tie_t600_summary.py <survey_dir>")

    survey_dir = os.path.abspath(sys.argv[1])
    in_path = os.path.join(survey_dir, "analysis_micro_tie_metrics.csv")
    out_path = os.path.join(survey_dir, "analysis_micro_tie_t600_summary.csv")

    with open(in_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    t600 = [
        r for r in rows
        if r.get("wave_id") == "t600"
        and r.get("respondent") not in {"__overall__", "OVERALL"}
    ]
    if not t600:
        raise SystemExit("No rows with wave_id=t600 found in analysis_micro_tie_metrics.csv")

    total_tp = sum(int(r["tp"]) for r in t600)
    total_fp = sum(int(r["fp"]) for r in t600)
    total_fn = sum(int(r["fn"]) for r in t600)
    total_tn = sum(int(r["tn"]) for r in t600)
    total_fail_safe = sum(int(r.get("n_fail_safe", 0)) for r in t600)
    total_pairs = sum(int(r.get("n_pairs_evaluated", 0)) for r in t600)

    overall = {
        "wave_id": "t600",
        "step": "600",
        "respondent": "OVERALL",
        "tp": str(total_tp),
        "fp": str(total_fp),
        "fn": str(total_fn),
        "tn": str(total_tn),
        "precision": f"{_fdiv(total_tp, (total_tp + total_fp)):.6f}",
        "recall": f"{_fdiv(total_tp, (total_tp + total_fn)):.6f}",
        "fpr": f"{_fdiv(total_fp, (total_fp + total_tn)):.6f}",
        "fnr": f"{_fdiv(total_fn, (total_tp + total_fn)):.6f}",
        "pair_accuracy": f"{_fdiv((total_tp + total_tn), (total_tp + total_fp + total_fn + total_tn)):.6f}",
        "n_fail_safe": str(total_fail_safe),
        "n_pairs_evaluated": str(total_pairs),
    }

    fieldnames = list(t600[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(t600)
        w.writerow(overall)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
