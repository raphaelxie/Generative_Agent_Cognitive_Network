# analysis_community_group_metrics.csv

Structured community/group perception metrics produced by `analyze_survey.py`.

Rows are one per `(wave_id, respondent)` plus one `__overall__` row per wave.
Truth definition: `observed_interaction_louvain_communities`.

Ground truth is computed via Louvain modularity maximisation
(`networkx.community.louvain_communities`, seed=42, weight=count_cumulative)
on the cumulative observed-interaction graph (`tie_cumulative == 1` edges).
Community IDs are 1-indexed, sorted lexicographically by the minimum member
name within each community.

## Columns

- `wave_id`            — survey wave identifier
- `step`               — simulation step at which the survey was taken
- `truth_definition`   — always `observed_interaction_louvain_communities`
- `respondent`         — agent name (or `__overall__` for the wave mean)
- `n_perceived_groups` — number of distinct groups the respondent named
- `n_actual_groups`    — number of Louvain communities in ground truth
- `nmi`                — Normalized Mutual Information between perceived and
                         ground-truth partition (arithmetic-mean normalisation);
                         1.0 = perfect agreement, 0.0 = no mutual information
- `ari`                — Adjusted Rand Index between perceived and ground-truth
                         partition; 1.0 = perfect, 0.0 = random, negative =
                         worse than random
- `n_agents_placed`    — number of agents present in both the perceived map
                         and the ground-truth map (used for NMI/ARI)
- `n_expected`         — total agents in the ground-truth community map
- `n_fail_safe`        — number of community_group rows flagged is_fail_safe

The `__overall__` row reports respondent means for `n_perceived_groups`,
`nmi`, `ari`, and `n_agents_placed`; `n_fail_safe` is the wave total.
