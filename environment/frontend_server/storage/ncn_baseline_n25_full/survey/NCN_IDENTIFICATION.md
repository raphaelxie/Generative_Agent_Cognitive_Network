# NCN Identification Experiment (n=25, R=2 per arm)

## Design

- Pre wave: shared `pre` @ step 1800 (inherited from baseline fork)
- Post wave: `post` @ step 3300 (+1200 steps post-shock)
- Noise floor: 5 reps @ frozen step 2100 (`noise_r01, noise_r02, noise_r03, noise_r04, noise_r05`)
- Replicates: 2 independent forks per arm (control / hub / broker)
- Hub target: John Lin; Broker target: Eddy Lin
- Primary windowed truth: `observed_interaction_recent` (200 min recent window)

## Noise band (measurement stochasticity @ step 2100)

See `ncn_id_noise_band.csv`. Compare treatment DiD against `2 × noise_std`.

## Replicated DiD (primary outcomes)

| Instrument | Truth | Control Δ | Hub DiD | Broker DiD | 2×noise_std | Hub>noise? | Broker>noise? |
|---|---|---:|---:|---:|---:|---|---|
| micro_tie | observed_interaction_recent | -0.0562 | 0.0039 | 0.0115 | 0.0050 | no | yes |
| micro_tie | observed_interaction_recent_betweenness_rank |  |  |  |  | ? | ? |
| micro_tie | observed_interaction_recent_louvain_communities |  |  |  |  | ? | ? |
| bridge_rank | observed_interaction_recent |  |  |  |  | ? | ? |
| bridge_rank | observed_interaction_recent_betweenness_rank | 0.7120 | 0.3104 | 0.0288 | 0.1766 | yes | no |
| bridge_rank | observed_interaction_recent_louvain_communities |  |  |  |  | ? | ? |
| community_nmi | observed_interaction_recent |  |  |  |  | ? | ? |
| community_nmi | observed_interaction_recent_betweenness_rank |  |  |  |  | ? | ? |
| community_nmi | observed_interaction_recent_louvain_communities | 0.0714 | -0.0661 | -0.0503 | 0.0278 | yes | yes |

## Files

- `ncn_id_noise_band.csv` — spread across noise reps
- `ncn_id_replicate_deltas.csv` — per-fork pre/post deltas
- `ncn_id_did_summary.csv` — full replicated DiD table
- `ncn_id_manifest.json` — run manifest
