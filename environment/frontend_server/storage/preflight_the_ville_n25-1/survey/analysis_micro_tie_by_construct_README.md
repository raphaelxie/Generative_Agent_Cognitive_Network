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
