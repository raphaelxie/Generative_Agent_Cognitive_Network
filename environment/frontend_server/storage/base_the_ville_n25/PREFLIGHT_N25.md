# n=25 pre-flight (`preflight_the_ville_n25-1`)

Forked from `base_the_ville_n25`. Use this folder for n=25 ground-truth layer setup and plumbing checks.

## Ground truth layers

| Layer | File | When written |
|-------|------|--------------|
| `observed_interaction` | `survey/ground_truth/ground_truth_edges_{step}.csv` | Each `survey` wave |
| `background_social_tie` | `survey/ground_truth/background_social_edges.csv` | `background_social_truth.py` |
| `background_or_interaction` | Derived in `analyze_survey.py` | Union of layers 1 and 2 |

## Automated sequence

From `reverie/backend_server/`:

```bash
python run_preflight_n25.py
```

This headless script: fork → 1800-step burn-in → `survey pre` → `background_social_truth.py` → `analyze_survey.py`.

## Manual alternative

1. Django `runserver` + `python reverie.py`, load `preflight_the_ville_n25-1`.
2. Burn-in: `run 1800` (or use headless script).
3. `save`, then `python preflight_n25_batch.py` (survey + shock probe).
4. `python background_social_truth.py ../../environment/frontend_server/storage/preflight_the_ville_n25-1/survey`
5. `python analyze_survey.py ../../environment/frontend_server/storage/preflight_the_ville_n25-1/survey`
6. `python verify_preflight_n25.py ../../environment/frontend_server/storage/preflight_the_ville_n25-1/survey`

## History CSV

Background truth uses `environment/frontend_server/static_dirs/assets/the_ville/agent_history_init_n25.csv` (25 agents, 300 dyads).
