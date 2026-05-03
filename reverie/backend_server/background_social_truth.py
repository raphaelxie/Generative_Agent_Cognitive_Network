"""
Build a conservative background social-tie layer from agent history CSVs.

This is intentionally separate from observed interaction ground truth. It can
be run after a simulation on an existing survey directory; no ReverieServer
load or LLM calls are required.

Usage:
  python reverie/backend_server/background_social_truth.py <survey_dir>
  python reverie/backend_server/background_social_truth.py <survey_dir> --history-csv <path>
"""
import argparse
import csv
import itertools
import os
import re
import sys


_backend_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_backend_dir, "..", ".."))


BACKGROUND_COLUMNS = [
    "node_a",
    "node_b",
    "tie_background",
    "needs_review",
    "relation_types",
    "evidence_a",
    "evidence_b",
    "review_notes",
    "source_file",
    "extraction_method",
]

EXTRACTION_METHOD = "conservative_relation_cue_rules_v1"


STRONG_PATTERNS = [
    ("housemate", re.compile(r"\bhousemates?\b", re.I)),
    ("dormmate", re.compile(r"\bdormmates?\b", re.I)),
    ("classmate", re.compile(r"\bclassmates?\b", re.I)),
    ("neighbor", re.compile(r"\b(?:next door )?neighbors?\b", re.I)),
    ("friend", re.compile(r"\b(?:close |good |loyal )?friends?\b", re.I)),
    ("close", re.compile(r"\b(?:somewhat )?close\b|\bclose with\b", re.I)),
    ("known", re.compile(r"\b(?:have|has|had|you've|you have|you two have)\s+known\b|\bknown each other\b", re.I)),
    ("knows_as_role", re.compile(r"\bknow(?:n|s)?\s+.+\s+as\s+(?:a|an|your)\b", re.I)),
    ("customer", re.compile(r"\bknown? .+ as a customer\b|\bcustomer at your\b", re.I)),
    ("crush", re.compile(r"\bcrush on\b|\bsecret crush\b", re.I)),
    ("student_professor", re.compile(r"\bprofessor\b|\bstudents?\b|\bcollege\b", re.I)),
    ("antagonism", re.compile(r"\bdo not like\b|\bdon't like\b|\brather not see\b|\bfight\b", re.I)),
    ("talks_often", re.compile(r"\btalk(?:s|ed)? (?:to|with) .*\boften\b|\bsometimes talk\b", re.I)),
]

AMBIGUOUS_PATTERNS = [
    ("not_talked", re.compile(r"\b(?:haven't|have not|hasn't|has not)\s+(?:really\s+)?talked\b|\bnot really talked\b", re.I)),
    ("met_not_talked", re.compile(r"\bmet\b.*\bnot really talked\b|\bnot really talked\b.*\bmet\b", re.I)),
    ("met", re.compile(r"\b(?:met|you've met|you have met)\b", re.I)),
    ("frequent_customer", re.compile(r"\bfrequent customer\b", re.I)),
    ("seen", re.compile(r"\b(?:see|seen|saw)\b", re.I)),
    ("works_at", re.compile(r"\bworks? at\b|\bruns\b", re.I)),
    ("vague_like", re.compile(r"\bthink .+ (?:cute|cool|kind-hearted|dedicated)\b", re.I)),
    ("location_overlap", re.compile(r"\bfrequent(?:s)?\b|\bspend time at\b|\bgo to\b", re.I)),
]

NEGATING_REVIEW_PATTERNS = [
    re.compile(r"\b(?:haven't|have not|hasn't|has not)\s+(?:really\s+)?talked\b", re.I),
    re.compile(r"\bdon't know much\b|\bdo not know much\b", re.I),
]


README_TEXT = """# background_social_edges.csv

Conservative background social-tie layer derived from an agent history CSV.

This file is not observed interaction ground truth. Observed interactions remain
in `ground_truth_edges_{step}.csv` and are generated from chat ConceptNodes.
This file captures baseline social facts encoded before or outside the current
simulation run, such as housemates, dormmates, classmates, friends, neighbors,
known customer/bartender relationships, professor/student relationships, crushes,
or antagonistic relationships.

## Conservative Coding Rule

A name mention is only evidence; it is not automatically a positive tie.

- `tie_background = 1`, `needs_review = 0`: strong relation cues are present.
  Examples include housemate, dormmate, classmate, friend, close, neighbor,
  have known each other, known as a customer, professor/student, crush, or
  explicit antagonism.
- `tie_background = 1`, `needs_review = 1`: strong relation cues are present,
  but the evidence also contains a qualifying cue such as "have not really
  talked" or "do not know much."
- `tie_background = 0`, `needs_review = 1`: only weak or ambiguous contact cues
  are present. Examples include met, seen, works at, frequent customer, or
  shared-location evidence without a direct social relation.
- `tie_background = 0`, `needs_review = 0`: no evidence was found for the dyad.

The rule intentionally prefers false negatives over false positives. Ambiguous
rows preserve evidence for manual review instead of being promoted to positives.

## Columns

- `node_a`, `node_b`: unordered dyad.
- `tie_background`: conservative baseline social-tie label.
- `needs_review`: 1 when evidence is ambiguous or qualified.
- `relation_types`: semicolon-separated relation cues found in evidence.
- `evidence_a`, `evidence_b`: semicolon-separated source snippets from each
  agent's history, when available.
- `review_notes`: why the row was marked for review.
- `source_file`: history CSV used to produce this file.
- `extraction_method`: rule set used by the writer.
"""


def _read_history(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = {"Name", "Whisper"} - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"history CSV missing columns: {sorted(missing)}")
    return {r["Name"].strip(): r["Whisper"].strip() for r in rows}


def _discover_edges_csv(survey_dir):
    gt_dir = os.path.join(survey_dir, "ground_truth")
    if not os.path.isdir(gt_dir):
        return None
    candidates = [
        fn for fn in os.listdir(gt_dir)
        if re.match(r"^ground_truth_edges_\d+\.csv$", fn)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda fn: int(re.search(r"(\d+)", fn).group(1)))
    return os.path.join(gt_dir, candidates[0])


def _read_roster(edges_csv, history_by_name):
    if not edges_csv:
        return sorted(history_by_name.keys())

    names = set()
    with open(edges_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names.add(row["node_a"])
            names.add(row["node_b"])
    return sorted(names)


def _default_history_path(roster_size):
    return os.path.join(
        _project_root,
        "environment",
        "frontend_server",
        "static_dirs",
        "assets",
        "the_ville",
        f"agent_history_init_n{roster_size}.csv",
    )


def _snippets_for(whisper, other_name):
    snippets = []
    for raw in whisper.split(";"):
        snippet = raw.strip()
        if other_name in snippet:
            snippets.append(snippet)
    return snippets


def _classify_snippets(snippets):
    strong = set()
    ambiguous = set()
    notes = set()

    for snippet in snippets:
        for label, pattern in STRONG_PATTERNS:
            if pattern.search(snippet):
                strong.add(label)
        for label, pattern in AMBIGUOUS_PATTERNS:
            if pattern.search(snippet):
                ambiguous.add(label)
        for pattern in NEGATING_REVIEW_PATTERNS:
            if pattern.search(snippet):
                notes.add("strong evidence is qualified by limited contact")

    if snippets and not strong and not ambiguous:
        ambiguous.add("name_mention_without_relation_cue")
        notes.add("name mention lacks a strong relation cue")

    return strong, ambiguous, notes


def _build_row(a, b, history_by_name, source_file):
    snippets_a = _snippets_for(history_by_name.get(a, ""), b)
    snippets_b = _snippets_for(history_by_name.get(b, ""), a)

    strong_a, ambiguous_a, notes_a = _classify_snippets(snippets_a)
    strong_b, ambiguous_b, notes_b = _classify_snippets(snippets_b)

    strong = strong_a | strong_b
    ambiguous = ambiguous_a | ambiguous_b
    notes = notes_a | notes_b

    tie_background = 1 if strong else 0
    if ambiguous and not strong:
        notes.add("only weak or ambiguous contact cues found")

    needs_review = 1 if notes else 0
    relation_types = sorted(strong | ambiguous)

    return {
        "node_a": a,
        "node_b": b,
        "tie_background": tie_background,
        "needs_review": needs_review,
        "relation_types": ";".join(relation_types),
        "evidence_a": " || ".join(snippets_a),
        "evidence_b": " || ".join(snippets_b),
        "review_notes": "; ".join(sorted(notes)),
        "source_file": source_file,
        "extraction_method": EXTRACTION_METHOD,
    }


def build_background_social_edges(history_csv, roster=None):
    history_by_name = _read_history(history_csv)
    names = sorted(roster if roster is not None else history_by_name.keys())
    missing = [name for name in names if name not in history_by_name]
    if missing:
        raise ValueError(
            "history CSV missing roster names: " + ", ".join(sorted(missing))
        )

    return [
        _build_row(a, b, history_by_name, os.path.basename(history_csv))
        for a, b in itertools.combinations(names, 2)
    ]


def write_background_social_truth(history_csv, output_dir, roster=None):
    rows = build_background_social_edges(history_csv, roster=roster)
    os.makedirs(output_dir, exist_ok=True)

    edges_path = os.path.join(output_dir, "background_social_edges.csv")
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BACKGROUND_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    readme_path = os.path.join(output_dir, "background_social_edges_README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(README_TEXT)

    return edges_path, readme_path, rows


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Build conservative background social-tie truth artifacts."
    )
    parser.add_argument(
        "survey_dir",
        help="Existing survey directory; output defaults to <survey_dir>/ground_truth.",
    )
    parser.add_argument(
        "--history-csv",
        default=None,
        help="Agent history CSV. Defaults to the_ville/agent_history_init_n{roster}.csv.",
    )
    parser.add_argument(
        "--edges-csv",
        default=None,
        help="Existing ground_truth_edges_{step}.csv used only to infer roster.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <survey_dir>/ground_truth.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    survey_dir = os.path.abspath(args.survey_dir)
    if not os.path.isdir(survey_dir):
        raise SystemExit(f"survey dir not found: {survey_dir}")

    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.join(survey_dir, "ground_truth")
    )

    edges_csv = os.path.abspath(args.edges_csv) if args.edges_csv else None
    if not edges_csv:
        edges_csv = _discover_edges_csv(survey_dir)

    if args.history_csv:
        history_csv = os.path.abspath(args.history_csv)
        history_for_roster = _read_history(history_csv)
        roster = _read_roster(edges_csv, history_for_roster)
    else:
        if not edges_csv:
            raise SystemExit(
                "could not infer roster: pass --edges-csv or --history-csv"
            )
        # Read once with a temporary empty history to get the roster size.
        roster = _read_roster(edges_csv, {})
        history_csv = _default_history_path(len(roster))
        history_for_roster = _read_history(history_csv)

    if not os.path.isfile(history_csv):
        raise SystemExit(f"history CSV not found: {history_csv}")

    edges_path, readme_path, rows = write_background_social_truth(
        history_csv, output_dir, roster=roster
    )

    n_positive = sum(int(r["tie_background"]) for r in rows)
    n_review = sum(int(r["needs_review"]) for r in rows)
    print(f"Wrote {edges_path}")
    print(f"Wrote {readme_path}")
    print(f"Input history: {history_csv}")
    print(f"Input roster edges: {edges_csv or '(history roster)'}")
    print(f"Dyads: {len(rows)}")
    print(f"Positive background ties: {n_positive}")
    print(f"Needs review: {n_review}")


if __name__ == "__main__":
    main()
