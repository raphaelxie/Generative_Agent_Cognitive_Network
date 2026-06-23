# Generative Agents for Network Cognition Experiments

This repository extends the Generative Agents simulation environment to study network cognition: whether agents form coherent perceptions of social structure at multiple levels of resolution, and whether those perceptions update after structural shocks. The project implements a diagnostic instrument that administers structured perception surveys to a live n=15 Smallville simulation, records retrieval diagnostics non-invasively, computes ground truth from the observed interaction graph, and compares pre- and post-shock perception responses to identify whether agents encode changes in network position.


## Perception Instruments

Eight question types are administered per survey wave. The four micro-tie instruments (CSN batch) ask each respondent about every ordered pair in the roster; the four network-level instruments (NCN) ask each respondent once about the full roster.

| Instrument | `question_type` | Prompt file | What it measures |
|---|---|---|---|
| Micro-tie (broad) | `micro_tie` | `csn_connection_batch_v1.txt` | Perceived pairwise connection or relationship |
| Micro-tie (interaction) | `micro_tie_interaction` | `csn_interaction_batch_v1.txt` | Perceived direct interaction or joint activity |
| Micro-tie (social) | `micro_tie_social` | `csn_social_tie_batch_v1.txt` | Perceived social relationship or personal bond |
| Micro-tie (group) | `micro_tie_group` | `csn_group_batch_v1.txt` | Perceived group or social circle co-membership |
| Centrality ranking | `centrality_rank` | `ncn_centrality_rank_v1.txt` | Perceived ordering of roster by social connectedness |
| Bridge ranking | `bridge_rank` | `ncn_bridge_rank_v1.txt` | Perceived ordering of roster by brokerage role |
| Community groups | `community_group` | `ncn_community_group_v1.txt` | Perceived partition of roster into social clusters |
| Self-position | `self_position` | `ncn_self_position_v1.txt` | Perceived own connectedness on a 1-5 scale |

Prompt files are located in `reverie/backend_server/persona/prompt_template/v3_ChatGPT/`.

The survey runner (`perception_survey.py`) snapshots and restores `last_accessed` timestamps on all memory nodes so retrieval side-effects are fully reversed after each wave. A per-wave retrieval diagnostics JSONL records scores and mention flags for every memory node retrieved during the survey.


## Ground Truth Layers

Each survey wave generates matching ground truth files. Analysis is scored against up to three truth layers per wave.

| Layer | Source file | Definition |
|---|---|---|
| `observed_interaction` | `ground_truth/ground_truth_edges_{step}.csv` | Cumulative interaction graph built from observed chats; `tie_cumulative == 1` for any pair that conversed by that step |
| `background_social_tie` | `ground_truth/background_social_edges.csv` | Conservative pre-existing social ties seeded at simulation start; does not change during the run |
| `background_or_interaction` | Derived (union) | Positive if a pair is positive in either of the above layers; the least construct-mismatched reference for broad perceived ties |
| `ground_truth_communities` | `ground_truth/ground_truth_communities_{step}.csv` | Louvain community partition (`networkx`, seed=42, weighted by `count_cumulative`) on the cumulative observed interaction graph |

Betweenness centrality for the bridge rank truth is computed from the same cumulative observed interaction graph using normalized betweenness (via `ground_truth_log.agent_betweenness`).


## Structural Shocks

Two shock treatments are implemented as soft isolation:

- **Hub removal** (`shock isolate-hub`): targets the agent with the highest degree in the cumulative observed interaction graph (Hailey Johnson in calibration runs). The agent remains present in the world but is blocked from initiating new conversations.
- **Broker removal** (`shock isolate-broker`): targets the agent with the highest betweenness centrality in the same graph (Carlos Gomez in calibration runs). Isolation mechanism is identical.

Soft isolation means the shocked agent does not disappear from the roster, so perception survey questions still include them as a target. This preserves the ability to detect whether other agents update their perception of the isolated agent's role. Shock events are logged to `shock_log.jsonl` with the agent name, degree, betweenness, and step.


## Analysis Pipeline

All scripts below are analysis-only (no LLM calls, no state mutation).

**`reverie/backend_server/analyze_survey.py`**  
Primary analysis driver. Reads `perception_survey_*.csv` and ground truth edge files; outputs:
- `analysis_micro_tie_metrics.csv` — per-respondent confusion matrix (TP/FP/FN/TN, precision, recall, FPR, FNR, pair accuracy) against `observed_interaction`
- `analysis_micro_tie_metrics_by_truth.csv` — same metrics scored against all three truth layers
- `analysis_micro_tie_by_construct.csv` — 3×3 cross-scoring matrix (each separated micro-tie construct × each truth layer)
- `analysis_bridge_rank_metrics.csv` — bridge top-1 hit rate, top-3 hit rate, and mean absolute rank error vs. betweenness ground truth
- `analysis_community_group_metrics.csv` — NMI and ARI vs. Louvain ground-truth partition per respondent

Usage: `python analyze_survey.py <survey_dir>`

**`reverie/backend_server/survey_network_summary.py`**  
Cross-checkpoint network diagnostics. Reads ground truth edge/chat CSVs across all waves; outputs `network_summary_over_time.csv` with one row per checkpoint (density, clustering, modularity, diameter, top-degree agent, top-betweenness agent, etc.).

Usage: `python survey_network_summary.py <survey_dir>`

**`reverie/backend_server/shock_prepost_audit.py`**  
Shock-aligned pre/post comparison. Reads existing survey outputs, aligns the nearest pre- and post-shock waves around `shock_log.jsonl`, and writes `shock_aligned_evidence_note.md` with per-instrument perception deltas and ground truth changes for the shocked agent.

Usage: `python shock_prepost_audit.py <survey_dir>`

**`reverie/backend_server/figures/generate_figures.py`**  
Publication-quality figure pipeline. Accepts one or more survey directories; outputs PDF and PNG for five figures:

| Figure | File stem | Content |
|---|---|---|
| 1 | `fig1_network_structure` | Ground truth network structure over time |
| 2 | `fig2_micro_tie_accuracy` | Micro-tie precision/recall by truth layer |
| 3 | `fig3_centrality_bridge` | Centrality and bridge rank accuracy |
| 4 | `fig4_shock_delta_summary` | Pre/post perception deltas aligned to shock |
| 5 | `fig5_retrieval_diagnostics` | Retrieval score distributions |

Usage: `python figures/generate_figures.py <survey_dir> [<survey_dir2> ...] [--output-dir DIR]`


## Key Outputs per Run

Each completed run under `environment/frontend_server/storage/<run>/survey/` contains:

**Survey waves** (one set per `survey <wave_id>` call):
- `perception_survey_{wave_id}.csv` — all instrument responses
- `perception_survey_{wave_id}_meta.json` — row counts, fail-safe rates, retrieval call count
- `retrieval_diagnostics_{wave_id}.jsonl` — per-node retrieval scores and mention flags

**Ground truth** (written by `perception_survey.py` at each wave):
- `ground_truth/ground_truth_chats_{step}.csv`
- `ground_truth/ground_truth_edges_{step}.csv`
- `ground_truth/background_social_edges.csv`
- `ground_truth/ground_truth_communities_{step}.csv` (written by `analyze_survey.py`)

**Analysis** (written by `analyze_survey.py`):
- `analysis_micro_tie_metrics.csv` + `_README.md`
- `analysis_micro_tie_metrics_by_truth.csv` + `_README.md`
- `analysis_micro_tie_by_construct.csv` + `_README.md`
- `analysis_bridge_rank_metrics.csv` + `_README.md`
- `analysis_community_group_metrics.csv` + `_README.md`
- `network_summary_over_time.csv`

**Shock and audit**:
- `shock_log.jsonl`
- `shock_aligned_evidence_note.md`

**Figures** (written by `generate_figures.py`):
- `figures/fig{1..5}_*.{pdf,png}`


## Experimental Runs

Completed runs are stored under `environment/frontend_server/storage/`.

| Run | Treatment | Steps | Notes |
|---|---|---|---|
| `prepost_n15_calibration-1` | Baseline | 0–1800 | Calibrated n=15 state; pre-shock survey reference |
| `prepost_n15_calibration-1_post_hub_t2400` | Hub removal | 1800–2400 | Hailey Johnson isolated (highest degree) |
| `prepost_n15_calibration-1_post_broker_t2400` | Broker removal | 1800–2400 | Carlos Gomez isolated (highest betweenness) |
| `bridge_probe_n15_baseline_t1800-1` | None | 0–1800 | Bridge perception baseline; no shock |
| `bridge_probe_n15_broker_t2400-1` | Broker probe | 0–2400 | Bridge perception following broker removal |
| `formal_n15_full_instruments-1` | Hub removal | 1800–3000 | Full instrument verification run |
| `preflight_the_ville_n25-1` | Baseline (n=25) | 0–1800 | n=25 ground-truth layer setup; see `PREFLIGHT_N25.md` in run folder |


### n=25 pre-flight

From `reverie/backend_server/`:

```bash
python run_preflight_n25.py          # headless: burn-in + survey + all 3 truth layers
python bootstrap_n25_ground_truth_layers.py preflight_the_ville_n25-1  # layers 1–2 only
python verify_preflight_n25.py ../../environment/frontend_server/storage/preflight_the_ville_n25-1/survey
```

Background social truth uses `agent_history_init_n25.csv` (25 agents, 300 dyads).


### Step 1. Generate utils.py

In `reverie/backend_server/`, create `utils.py`:

```python
openai_api_key = "<Your OpenAI API Key>"
key_owner = "<Name>"

maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

fs_storage = "../../environment/frontend_server/storage"
fs_temp_storage = "../../environment/frontend_server/temp_storage"

collision_block_id = "32125"
debug = True
```

### Step 2. Install dependencies

```
pip install -r requirements.txt
```

Python 3.9.12 is recommended. `networkx` is required for Louvain community detection in `analyze_survey.py`.

### Step 3. Start the environment server

```
cd environment/frontend_server
python manage.py runserver
```

Confirm the server is running at [http://localhost:8000/](http://localhost:8000/).

### Step 4. Start the simulation server

```
cd reverie/backend_server
python reverie.py
```

When prompted for the base simulation, use:

```
base_the_ville_n15
```

Enter a name for the new simulation, then run steps with `run <step-count>`.

### Running a Survey

At the "Enter option:" prompt, issue:

```
survey <wave_id>
```

For example, `survey pre` collects a full eight-instrument wave and writes all output files to the run's `survey/` directory. Survey wave identifiers are arbitrary strings; the convention in these experiments is `pre`, `post`, or `t{step}`.

### Running Analysis

From the project root, after one or more survey waves have been collected:

```bash
# Primary analysis (micro-tie, bridge rank, community group metrics)
python reverie/backend_server/analyze_survey.py \
  environment/frontend_server/storage/<run>/survey

# Cross-checkpoint network summary
python reverie/backend_server/survey_network_summary.py \
  environment/frontend_server/storage/<run>/survey

# Shock-aligned pre/post audit
python reverie/backend_server/shock_prepost_audit.py \
  environment/frontend_server/storage/<run>/survey

# Figure generation
python reverie/backend_server/figures/generate_figures.py \
  environment/frontend_server/storage/<run>/survey \
  --output-dir environment/frontend_server/storage/<run>/survey/figures
```


## Project Structure

```
generative_agents-main/
├── analysis/
│   └── network_from_nodes.py        helper for network construction
├── environment/
│   └── frontend_server/
│       ├── manage.py                Django entry point
│       ├── static_dirs/             map assets and agent history files
│       └── storage/                 one folder per saved simulation run
├── reverie/
│   ├── compress_sim_storage.py
│   └── backend_server/
│       ├── reverie.py               simulation loop and survey/shock commands
│       ├── perception_survey.py     eight-instrument survey runner
│       ├── analyze_survey.py        post-hoc metric computation
│       ├── survey_network_summary.py network diagnostics over time
│       ├── shock_prepost_audit.py   shock-aligned pre/post comparison
│       ├── ground_truth_log.py      ground truth CSV writers
│       ├── background_social_truth.py background tie seeding
│       ├── run_formal_protocol.py   formal experiment automation
│       ├── verify_instruments.py    instrument output verification
│       ├── figures/
│       │   └── generate_figures.py  publication figure pipeline
│       └── persona/
│           └── prompt_template/v3_ChatGPT/  instrument prompt files
├── requirements.txt
└── README.md
```


## Upstream Attribution

This project builds on the research paper "[Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)." It contains the core simulation module for generative agents—computational agents that simulate believable human behaviors—and their game environment.

**Authors:** Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein

Please cite the upstream paper if you use the simulation infrastructure:

```
@inproceedings{Park2023GenerativeAgents,
  author    = {Park, Joon Sung and O'Brien, Joseph C. and Cai, Carrie J. and Morris, Meredith Ringel and Liang, Percy and Bernstein, Michael S.},
  title     = {Generative Agents: Interactive Simulacra of Human Behavior},
  year      = {2023},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  booktitle = {In the 36th Annual ACM Symposium on User Interface Software and Technology (UIST '23)},
  keywords  = {Human-AI interaction, agents, generative AI, large language models},
  location  = {San Francisco, CA, USA},
  series    = {UIST '23}
}
```

Game asset credits: background art by [PixyMoon (@_PixyMoon)](https://twitter.com/_PixyMoon_), furniture/interior design by [LimeZu (@lime_px)](https://twitter.com/lime_px), character design by [ぴぽ (@pipohi)](https://twitter.com/pipohi).
