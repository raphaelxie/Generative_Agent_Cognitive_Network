# Generative Agents: Network Cognition Experiment

This repository contains the simulation engine, map assets, initial agent templates, and offline tests for a network-cognition experiment built on generative agents. The main design uses **25 agents**, a multi-day simulated trajectory, and repeated surveys that measure how agents perceive social ties relative to observed interaction.

Local experiment runs, snapshots, measurement outputs, credentials, and operator handbooks stay on the machine and are not published here.

## What is in this repo

| Path | Purpose |
|---|---|
| `reverie/backend_server/` | Simulation runtime, pilot protocol modules, surveys, and offline tests |
| `reverie/backend_server/persona/prompt_template/` | Prompt templates used by the agent stack |
| `environment/frontend_server/static_dirs/` | Map and character initialization assets |
| `environment/frontend_server/storage/base_the_ville_n25/` | Current 25-agent initial template |
| `environment/frontend_server/storage/base_the_ville_n15/` | Smaller initialization template |
| `environment/frontend_server/storage/base_the_ville_isabella_maria_klaus/` | Original three-agent template |
| `requirements.txt` / `requirements-pilot.txt` | Python dependencies |

Older N15 / N25 intervention and analysis scripts remain as code history. They are not the current entry point. The `base_*` directories are initialization templates, not experiment results.

## Local-only paths (ignored)

These are kept locally and excluded from GitHub:

- `.env` — API credentials
- `pilot_runs/` — trajectories, snapshots, checks, and measurements
- `_feedback/` / `_report/` — private notes and reports
- `**/retrieval_diagnostics*.jsonl` — large diagnostics
- Operator handoffs / run manuals (for example `MIDNIGHT_PILOT.md`, `HANDOFF_*.md`, `CSS_SURVEY.md`)

## Offline checks

```bash
PYTHONHASHSEED=104729 OPENAI_MODEL=gpt-6-luna python3 \
  -m pytest -q reverie/backend_server/tests
```

Use the Python environment and model pins documented in your local run notes. Do not upload credentials, snapshots, or full model logs to GitHub.

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
