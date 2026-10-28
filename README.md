# Generative Agents: Network Cognition Experiment

The current design uses **25 agents, 64 simulated hours, and 3 independent surveys at each measurement snapshot**. The pipeline is: advance the simulation, save behavior and interaction-network snapshots, measure network cognition in a separate process, then analyze the results.

This repository keeps the simulation engine, map, initial templates, and offline tests. Experiment records from before 1 August 2026 have been removed from the working tree. The scope is in the [cleanup record](REPOSITORY_CLEANUP_20261026.md). The old data can still be recovered from Git history.

## Documents to use

- [Current experiment protocol](MIDNIGHT_PILOT.md): the 64-hour schedule, checkpoint selection, surveys, diagnostics, and analysis.
- [Repair and validation handoff](HANDOFF_SCHEDULE_REVISION.md): the implementation, real-run acceptance results, past stops, and remaining issues.

The current behavior trajectory `smoke_mobility_v5` is saved through Day 1 14:00 (step 5040). It stopped once because a task-decomposition reply was invalid, then resumed with authorization and passed the 14:00 check. The full 64 hours are not finished, so this is not an uninterrupted run. Schedule revision now has the model choose what to keep and the code assign the minutes. Occasional task-decomposition failures are not fixed yet.

## Continue the existing trajectory

Keep using the Python 3.11.5 environment already verified on this machine, the existing run directory, and the local configuration. Do not reinstall or change locked dependencies. Do not upload credentials, snapshots, or full model logs to GitHub.

From the project root, check the saved progress. This does not call a paid API:

```bash
PYTHONHASHSEED=104729 OPENAI_MODEL=gpt-6-luna /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
  reverie/backend_server/run_midnight_pilot.py verify \
  --output pilot_runs/smoke_mobility_v5 --seed 104729 --repeats 2
```

`--repeats 2` is the frozen configuration of this trajectory. A resume must keep that value and pass `--defer-surveys`. The actual surveys run three times through a separate entry point, so changing the measurement count does not require rerunning the behavior trajectory:

```bash
PYTHONHASHSEED=104729 OPENAI_MODEL=gpt-6-luna /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
  reverie/backend_server/measure_saved_pilot.py \
  --source-run pilot_runs/smoke_mobility_v5 \
  --output pilot_runs/smoke_mobility_v5_measurements_r3 --repeats 3
```

The survey command calls a paid API. Run simulation and surveys one after the other, and run only one task at a time. The initial, 10:00, and 14:00 snapshots are saved. There are no survey answers and no completed measurements yet.

Continue the behavior trajectory on this machine to the planned end. This calls a paid API:

```bash
PYTHONHASHSEED=104729 OPENAI_MODEL=gpt-6-luna /usr/bin/caffeinate -i -s \
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -u \
  reverie/backend_server/run_midnight_pilot.py run \
  --output pilot_runs/smoke_mobility_v5 --seed 104729 --repeats 2 --defer-surveys
```

Do not change the source while a run is in progress. If it stops again, keep the logs and diagnose the cause. Do not resume automatically in a loop.

## Layout and outputs

| Path | Purpose |
|---|---|
| `reverie/backend_server/run_midnight_pilot.py` | Simulate, save, and resume |
| `reverie/backend_server/pilot_*.py` | Current protocol, schedule, actions, surveys, state, and analysis |
| `reverie/backend_server/measure_saved_pilot.py` | Measure saved snapshots in a separate process |
| `reverie/backend_server/tests/` | Regression tests that do not call a paid API |
| `environment/frontend_server/static_dirs/` | Map and character initialization assets |
| `environment/frontend_server/storage/base_the_ville_n25/` | Current 25-agent initial template |
| `pilot_runs/` | Local private trajectories, snapshots, checks, and measurements; not in Git |

Measurement output for the current run is `REPORT.md`, `report.json`, and `perception_scores.csv` in a separate directory. Three surveys estimate measurement noise. They are not three independent simulations.

Older N15 / N25 intervention and analysis scripts remain as code history. They are not the current entry point. The other `base_*` directories are initialization templates, not experiment results waiting to be deleted. Resume checks a source digest, so do not casually delete or rewrite the engine or prompt files.

## Offline checks

```bash
PYTHONHASHSEED=104729 OPENAI_MODEL=gpt-6-luna /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
  -m pytest -q reverie/backend_server/tests
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
