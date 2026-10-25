# Handoff: schedule-revision contract (next agent starts here)

Date: 2026-10-24/25. Workspace: `/Users/mac/Documents/PythonNote/SocResearch/generative_agents-main`.
Python: `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3` (the `.env` here sets a non-Luna model; the runner re-pins `gpt-6-luna`, and tests must pin it too).

## 1. What was wrong in the original experiment (solved)

The 25-agent, 3-day run (`main_10212026_v3_1_continuation`, copy under `/Users/mac/Documents/ChatGPT/Social Network Research/`) produced 3 edges (only the couples sharing a bedroom) and betweenness 0 for everyone.

Cause: the action-location prompts (`action_location_sector_v1`, `action_location_object_vMar11`) are completion-style and end with an open `{`. The old validator required `}` in the reply. `gpt-6-luna` replies in chat style, so all retries were rejected and the fail-safes sent every agent to its own home and own room. About 3,800 location decisions all resolved to home. The event-triple and `decide_to_talk` prompts failed the same way (the latter defaulting to "yes").

That old network result reflects this bug. It does not show that the model cannot form networks.

## 2. What was done (all in the workspace, not committed)

- Ported the v3.1 runtime from `pilot_runtime_v3_1` into the workspace: `pilot_tasks.py`, `pilot_schedule.py`, `pilot_api.py`, `run_midnight_pilot.py`, `plan.py`, `perceive.py`, `scratch.py`, `associative_memory.py`, `run_gpt_prompt.py`, plus 3 tests. Two tests that read evidence outside the repo now use `tests/fixtures/`.
- New `reverie/backend_server/pilot_actions.py`: strict JSON contracts for sector, arena, object, event triples, `decide_to_talk` and `decide_to_react`. It reuses `pilot_tasks.request_contract` (3 attempts, feedback on retry, `PilotScheduleError` on exhaustion, no silent defaults). A single option skips the model call.
- Hooks under `NCN_PILOT=1` in `persona/prompt_template/run_gpt_prompt.py`. Also fixed the non-pilot `event_triple` fail-safe re-wrapping bug.
- `run_midnight_pilot.py`: `location_preflight()` (real-API check that Isabella's cafe shift resolves to Hobbs Cafe) and `--stop-at-step N` (multiple of 180, an operational limit outside the config lock).
- Tests: `tests/test_pilot_actions.py` (27 tests) and `test_stop_at_step_*` in `test_pilot_engine.py`. Full suite: `PYTHONHASHSEED=104729 python3 -m pytest -q reverie/backend_server/tests` gives 117 passed.
- `MIDNIGHT_PILOT.md` has a new section "Action contract v4".

## 3. Smoke-run evidence (the fix works)

Run: `pilot_runs/smoke_mobility_v4`, seed 104729, repeats 2, `--defer-surveys`, paused at step 5040 (Day 1 14:00). `verify` reports `snapshot_valid` and `manifest_matches` true.

| Metric | Old run | Smoke run |
|---|---|---|
| Agents visiting a public place | 0/25 | 20/25 (all 6 public sectors) |
| Cumulative edges | 3 | 30 |
| Nodes with betweenness above 0 | 0 | 16 (top: Sam Moore 0.27, Carlos Gomez 0.22, Eddy Lin 0.19) |
| Largest component | 2 | 21 of 25 |

Command to re-run the smoke test or resume it:

```bash
cd /Users/mac/Documents/PythonNote/SocResearch/generative_agents-main
/usr/bin/caffeinate -i -s /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -u \
  reverie/backend_server/run_midnight_pilot.py run \
  --output "$PWD/pilot_runs/smoke_mobility_v4" --seed 104729 --repeats 2 --defer-surveys --stop-at-step 5040
```

Resuming the same directory is safe only while the source digest is unchanged. After you edit any `pilot_*.py` or `persona/` file, use a new output directory.

## 4. THE OPEN PROBLEM: schedule-revision contract

Now that agents leave home, meet, and react to each other, the existing schedule-revision contract is reached much more often. It failed with an uncaught `PilotScheduleError`, which aborts the whole run.

Code: `reverie/backend_server/pilot_tasks.py`
- `reschedule_reaction()` builds the prompt and calls `request_contract(..., "Schedule revision")`.
- `parse_replanned_suffix(raw, source, budget)` validates the reply.
- Called from `plan.py` `_create_react` (pilot branch) when a conversation or wait reaction is inserted.

Rejections recorded in `pilot_runs/smoke_mobility_v4/schedule_diagnostics.jsonl` (step is the engine tick):

| Sim time | Step | Attempts rejected | Error |
|---|---|---|---|
| 08:14:00 | 2964 | 3 of 3 (run aborted) | "Replanned suffix must fill the exact remaining interval" |
| 08:14:00 | 2964 (same tick, after resume) | 2 | same error |
| 08:30:10 | 3061 | 1 | "Replanning cannot lengthen a source interval" |
| 13:01:10 | 4687 | 1 | "Replanned suffix must fill the exact remaining interval" |

Totals: 111 accepted, 7 attempts rejected, and exactly one run abort. The aborting tick involved Carmen Ortiz. Resuming got past that tick (model output is nondeterministic).

Also seen: 12 rejected event-triple attempts and 1 rejected task-decomposition attempt. Both recovered via feedback retry.

What the contract asks (see `reschedule_reaction`): the model gets a list of original activities, each with an index and minutes, plus `remaining_minutes`. It must return `{"parts":[{"source":i,"minutes":m},...]}` with increasing unique indices, each `m` no larger than that source's minutes, and a total of exactly `remaining_minutes`.

Why this is hard (hypotheses, not verified):
- The model has to do exact integer arithmetic under an "omit or shorten, never lengthen" constraint. Replies are about 240-300 characters, so they are valid JSON that simply does not sum correctly.
- The retry gets feedback but only the error string and the rejected reply, with no running total.
- A feasible answer always exists: the source window spans at least `budget` minutes. So the failure is a model-arithmetic problem, not an impossible request.

### Evidence gap
I could not capture the actual failing prompts and replies. `schedule_diagnostics.jsonl` stores only hashes and lengths. The observer `/Users/mac/Documents/ChatGPT/Social Network Research/resume_20261021/observe_runner.py` logged nothing, probably because `run_midnight_pilot.configure()` re-execs the process for `PYTHONHASHSEED` and the profile hook is lost. First step: capture real failures. Either run the observer with `PYTHONHASHSEED=104729` already set in the environment so no re-exec occurs, or temporarily log `prompt`, `raw` and `error` inside `request_contract` when `error` is set (use a debug env var, keep it out of committed behavior).

### Recommended approach (decide after seeing real failures)
1. Prefer deterministic computation over asking the model to add. For example, ask the model only for a priority order or a keep/drop choice per source activity, then let code assign minutes. Keep order, never lengthen, fill the budget exactly (truncate the last kept row, or distribute any remainder).
2. Keep the pilot rule: no silent default replacing a model failure. If a deterministic fill is used, record it explicitly in the contract log. The scientific content is which activities survive the interruption, not the arithmetic.
3. Alternatively keep the current format and strengthen the repair prompt with the actual running total and the shortfall or excess, so the retry knows how many minutes to add or remove.
4. Whatever is chosen, add offline tests in `test_contract_repair.py` / `test_pilot_tasks.py`. Real failure replies from step 1 make the best fixtures.
5. Do not change unrelated schedule contracts (wake hour, daily plan, hourly schedule, task decomposition); they are accepted at normal rates.

### Acceptance
- Offline: full suite green.
- Real: rerun a smoke trajectory in a NEW output directory (the source digest will change), e.g. `--output pilot_runs/smoke_mobility_v5 --seed 104729 --repeats 2 --defer-surveys --stop-at-step 5040` (about 5 wall-clock hours and a few dollars; Day 1 morning is where reactions concentrate, so `--stop-at-step 3600` is a cheaper first pass). Require zero aborts, and no schedule-revision attempts exhausted.
- Re-check the metrics in section 3 (public-place visits, edges, nonzero betweenness) to confirm the change did not disturb mobility.

## 5. After that (not done, not required for this handoff)

- A full 64-hour fresh trajectory in a new directory, following `MAIN_10202026_RUN_GUIDE.md` (the old run guide's timing windows are expired; ask the user for a new window).
- The 3-day analysis must be redone on that new trajectory.
- Out of scope so far: about 14.6% of survey probability items missing, and the alphabetical-order effect in degree rankings.

## 6. Housekeeping

- Nothing is committed to git; all changes are in the working tree.
- `pilot_runs/smoke_mobility_v4` is evidence, so leave it as is.
- `/tmp/smoke_check.py` computes the section 3 metrics, but `/tmp` may be cleared. Metrics come from `worlds/control/movement/*.json`, `network_history.json`, and `pilot_protocol.graph_from_chats` plus `networkx.betweenness_centrality`.
- Never print `.env`. Use full-suite tests before and after any source change.
