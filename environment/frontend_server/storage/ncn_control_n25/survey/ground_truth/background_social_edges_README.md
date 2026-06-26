# background_social_edges.csv

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
