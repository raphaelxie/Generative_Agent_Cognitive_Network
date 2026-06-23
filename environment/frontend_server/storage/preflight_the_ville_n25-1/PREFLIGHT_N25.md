# n=25 pre-flight (`preflight_the_ville_n25-1`)

Forked from `base_the_ville_n25`. Headless burn-in, perception survey, three ground-truth layers, and scored analysis for the 25-agent roster.

## Results summary

| Item | Value |
|------|-------|
| **Status** | Complete (burn-in + survey + analysis) |
| **Agents** | 25 |
| **Burn-in steps** | 1800 (0 → 1800) |
| **Sim start** | February 13, 2023, **08:00:00** |
| **Sim end (survey wave)** | February 13, 2023, **13:00:00** |
| **`sec_per_step`** | 10 (5 sim hours of activity after 08:00 start) |
| **Observed chats logged** | 127 rows (`ground_truth_chats_1800.csv`) |

### Ground truth layers (300 undirected dyads each)

| Layer | File | Positive ties | Description |
|-------|------|---------------|-------------|
| **Layer 1 — `observed_interaction`** | `survey/ground_truth/ground_truth_edges_1800.csv` | **46 / 300** | Cumulative chat/interaction graph at step 1800 |
| **Layer 2 — `background_social_tie`** | `survey/ground_truth/background_social_edges.csv` | **86 / 300** | Conservative pre-existing ties from `agent_history_init_n25.csv` |
| **Layer 3 — `background_or_interaction`** | Derived in `analyze_survey.py` | **107 / 300** | Union of layers 1 and 2 |

Layer 1 is non-empty, confirming agents were awake and conversing during burn-in (contrast with the initial midnight-start attempt, which produced **0** observed ties after 1800 steps because all agents remained asleep until ~05:00).

### Survey (`pre` wave at step 1800)

| Metric | Value |
|--------|-------|
| Respondents | 25 |
| Total survey rows | 61,925 |
| Micro-tie rows (×4 instruments) | 15,000 each |
| Centrality / bridge / community rows | 625 each |
| Self-position rows | 25 |
| Retrieval calls (design) | 2,525 |
| Fail-safe responses | 266 |

Primary output: `survey/perception_survey_pre.csv`  
Metadata: `survey/perception_survey_pre_meta.json`

### Micro-tie accuracy (`__overall__`, `pre` wave)

From `survey/analysis_micro_tie_metrics_by_truth.csv`:

| Truth layer | Precision | Recall | Pair accuracy |
|-------------|-----------|--------|---------------|
| `observed_interaction` | 0.165 | 0.490 | 0.541 |
| `background_social_tie` | 0.278 | 0.442 | 0.511 |
| `background_or_interaction` | 0.356 | 0.454 | 0.512 |

Per-construct and per-respondent breakdowns: `analysis_micro_tie_by_construct.csv`, `analysis_micro_tie_metrics.csv`.

## Configuration fix applied

`base_the_ville_n25/reverie/meta.json` **`curr_time`** was changed from **`00:00:00`** (midnight) to **`08:00:00`** to match `base_the_ville_n15`. With `sec_per_step=10`, 1800 steps advances the clock from 08:00 to 13:00 — within normal waking hours — instead of 00:00 to 05:00 when every agent was still sleeping.

Agent-history typos fixed in `agent_history_init_n25.csv` (Abigail **Chen**, Hailey **Johnson**) so background-truth name matching works.

## Artifacts

```
preflight_the_ville_n25-1/
├── reverie/meta.json                    # step=1800, curr_time=13:00
├── survey/
│   ├── perception_survey_pre.csv
│   ├── perception_survey_pre_meta.json
│   ├── retrieval_diagnostics_pre.jsonl  # see note below
│   ├── analysis_micro_tie_metrics_by_truth.csv
│   ├── analysis_micro_tie_by_construct.csv
│   ├── analysis_micro_tie_metrics.csv
│   ├── analysis_bridge_rank_metrics.csv
│   ├── analysis_community_group_metrics.csv
│   └── ground_truth/
│       ├── ground_truth_edges_1800.csv
│       ├── ground_truth_chats_1800.csv
│       ├── ground_truth_communities_1800.csv
│       └── background_social_edges.csv
```

## Known issue

`retrieval_diagnostics_pre.jsonl` may be **incomplete** (partial overwrite from a later aborted survey restart). Core deliverables (survey CSV, ground truth, analysis) are complete. Full `verify_preflight_n25.py` checks diagnostics line count (expected 2,525); use `--layers-only` to validate the three truth layers without the diagnostics file:

```bash
cd reverie/backend_server
python verify_preflight_n25.py ../../environment/frontend_server/storage/preflight_the_ville_n25-1/survey --layers-only
```

## Automated pipeline

From `reverie/backend_server/`:

```bash
python run_preflight_n25.py
```

Sequence: fork → 1800-step burn-in → `survey pre` → `background_social_truth.py` → `analyze_survey.py`.

**Resume:** If this folder already exists, `run_preflight_n25.py` loads `preflight_the_ville_n25-1` at the step in `reverie/meta.json` and continues. Survey-only after burn-in:

```bash
python run_survey_analysis_n25.py preflight_the_ville_n25-1
```

## Manual alternative

1. Django `runserver` + `python reverie.py`, load `preflight_the_ville_n25-1`.
2. Burn-in: `run 1800` (or use headless script).
3. `save`, then `python preflight_n25_batch.py` (survey + shock probe).
4. `python background_social_truth.py ../../environment/frontend_server/storage/preflight_the_ville_n25-1/survey`
5. `python analyze_survey.py ../../environment/frontend_server/storage/preflight_the_ville_n25-1/survey`
6. `python verify_preflight_n25.py ../../environment/frontend_server/storage/preflight_the_ville_n25-1/survey`

## History CSV

Background truth uses `environment/frontend_server/static_dirs/assets/the_ville/agent_history_init_n25.csv` (25 agents, 300 dyads).
