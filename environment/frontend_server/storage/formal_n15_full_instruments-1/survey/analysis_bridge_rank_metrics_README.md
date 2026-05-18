# analysis_bridge_rank_metrics.csv

Structured bridge-role perception metrics produced by `analyze_survey.py`.

Rows are one per `(wave_id, respondent)` plus one `__overall__` row per wave.
The first-pass truth definition is intentionally singular:
`observed_interaction_cumulative_betweenness_rank`.

This means the actual bridge/broker ordering is computed from the observed
interaction graph's cumulative ties (`count_cumulative > 0`) using normalized
betweenness centrality. It does not use background social-tie truth, recent
window truth, target-excluded views, or post-shock counterfactuals.

Core columns:
- `actual_top_broker`: highest-betweenness agent in observed cumulative truth.
- `perceived_rank_of_actual_top_broker`: where a respondent ranked that agent
  in the full-roster `bridge_rank` question.
- `bridge_top1_hit`: 1 iff the respondent ranked the actual top broker first.
- `bridge_top3_hit`: 1 iff the respondent ranked the actual top broker in the
  top three.
- `bridge_mean_abs_rank_error`: mean absolute difference between perceived
  bridge rank and actual cumulative-betweenness rank across ranked agents.

The `__overall__` row reports respondent means for rank, hit-rate, rank-error,
and ranked-count fields.
