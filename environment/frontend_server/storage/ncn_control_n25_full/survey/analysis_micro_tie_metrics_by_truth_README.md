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
