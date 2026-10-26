# Handoff: schedule-revision contract (next agent starts here)

Date: 2026-10-24/25. Workspace: `/Users/mac/Documents/PythonNote/SocResearch/generative_agents-main`.
Python: `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3` (the `.env` here sets a non-Luna model; the runner re-pins `gpt-6-luna`, and tests must pin it too).

## Update: priority-fill-v1 implementation (2026-10-25)

**2026-10-26 repository cleanup:** at the user's request, pre-August experiment outputs and logs were removed from the working tree. Current v4/v5 evidence, templates, assets and frozen runtime source were verified unchanged; all 162 offline tests and the step 5040 snapshot verification passed. See `REPOSITORY_CLEANUP_20261026.md`. Git history is retained; the user has authorized committing and pushing the implementation and cleanup.

**2026-10-26 execution/measurement decision:** the user selected three independent survey repeats and will continue on the current machine. The alternate-machine handoff, installation list and private transfer archive have been removed at the user's request. `measure_saved_pilot.py` remains necessary: it supports an independent repeats=3 measurement output without rewriting the frozen repeats=2 trajectory. Use the local commands in `README.md`.

The step 5040 snapshot has been verified (25 agents, snapshot and manifest match). Initial, 10:00 and 14:00 snapshots exist, but there are zero completed survey waves and zero respondent files; no paid questionnaires have run for this trajectory. The three-repeat wrapper adds four offline regression tests; the full suite passed **162 tests** under Python 3.11 with `PYTHONHASHSEED=104729` and `OPENAI_MODEL=gpt-6-luna`. Simulation runtime source digest is unchanged.

The sections below describe the original v4 diagnosis. The v4 action repair is now committed as `4d0ec88fc`, rather than uncommitted. Schedule revision now asks for a complete priority permutation and deterministically fills the remaining interval: preserve higher priorities at their original duration, shorten only the boundary activity, omit the rest, then restore original execution order. This removes model arithmetic from this contract. The schedule state stays v3; the manifest separately locks `reaction_revision_version=priority-fill-v1`.

Each revision has its actual recipient and an independent revision ID. `schedule_revision_audit.jsonl` records the source, priority and allocation. Opt-in `NCN_REACTION_DEBUG_LOG` captures full attempt prompts/replies/errors; normal attempt logs remain hashes and metadata. A recorded allocation is not proof that its entire engine tick committed. Three malformed rankings still stop the run; no automatic priority completion or failed-run retry is used.

Run `run_schedule_smoke.py` as documented in `MIDNIGHT_PILOT.md`: a new directory, first step 3600, then 5040 only after verification succeeds. `smoke_status.json` and `acceptance_*.json` are the source of truth for actual validation progress; implementation alone is not evidence of a successful real run. Preserve all v4 evidence.

Metric correction: the old `/tmp/smoke_check.py` used action destination descriptions. Actual coordinates show **19/25** public-place visitors at step 5040, versus 20/25 public destinations (Francisco Lopez had not arrived). All six public sectors were reached. The 30 edges, 16 nonzero betweenness nodes and largest component of 21 were independently verified. New validation reports both destination and arrival metrics.

The exact v4 failing replies remain unavailable. Synthetic ranking tests are labeled synthetic. Any future observed failures can be extracted from the new local debug log after review.

Implementation validation: **158 tests passed** with Python 3.11 and `PYTHONHASHSEED=104729`; `git diff --check` passed. The real `smoke_mobility_v5` supervisor started on 2026-10-25 at 06:24 America/Los_Angeles. Its API/location preflight and **10:00 stage passed**. The original 14:00 attempt **failed at 11:05 (step 3990)** on the unchanged **Task decomposition** contract; a later explicitly user-requested recovery **reached 14:00 and passed post-recovery validation**, detailed below. The original zero-abort criterion remains unmet. Frozen runtime source digest remains `29cfd946e0d44df4cddc0f3938bb74a3ac12875ec3e81f5d27c379c8b1cd2813`; v4 evidence and runtime source were preserved.

### v5 real-validation outcome (2026-10-25 10:21 America/Los_Angeles)

- `acceptance_003600.json`: 34 model schedule revisions, all accepted on attempt 1; allocation audits, full debug traces, snapshot and manifest all passed. At 10:00, actual public visitors were 14/25 (v4: 17/25), public destinations 15/25 (v4: 17/25), all six public sectors were reached in both, edges 10 (v4: 8), nonzero betweenness nodes 4 (v4: 4), largest component 5 (v4: 4).
- Across the entire attempted run, **44 schedule revisions passed on attempt 1, zero rejected or exhausted**. Independent post-failure audit confirmed all 44 allocation records and debug traces are valid. There are no rejected priority replies to turn into new fixtures.
- The fatal failure was `Task decomposition contract failed after 3 attempts: Subtask durations must sum exactly to the parent duration`. All three replies have the same SHA-256 and length (760 characters). The API calls themselves succeeded. Task-decomposition prompt/reply bodies were not captured because the approved debug scope was schedule revision only; do not invent the exact arithmetic error or attribute a persona from interleaved stdout.
- Last completed live tick: step 3990 / 11:05:00. Last durable recovery snapshot: **step 3973 / 11:02:10**; `snapshot_valid` and `manifest_matches` are true. Its manifest retains the pre-failure status `running`; terminal process status is **failed** in `smoke_status.json`.
- Evidence: `pilot_runs/smoke_mobility_v5/failure_review.json`, `smoke_status.json`, `acceptance_003600.json`, `schedule_diagnostics.jsonl`, `schedule_revision_audit.jsonl`, `schedule_revision_debug.jsonl`, and `smoke_launcher.log`. No `acceptance_005040.json` exists: do not claim full smoke acceptance or 14:00 network results.
- Remaining work requires a separately scoped task-decomposition arithmetic/feedback repair and a fresh trajectory to complete 14:00 acceptance. Task decomposition was explicitly excluded from this implementation. The temporary follow-up stops after reporting this failure; it does not silently expand scope or resume this failed trajectory.

### User-authorized continuation (2026-10-25 18:12 America/Los_Angeles)

After the failure explanation, the user explicitly requested continuing the unfinished simulation validation. A **single recovery attempt** ran from the verified step 3973 snapshot to step 5040, using the identical runtime source digest. This superseded the earlier no-resume operational instruction for this specific user-requested attempt; it does not erase the prior abort or establish uninterrupted-run acceptance.

- Current continuation status: `pilot_runs/smoke_mobility_v5/resume_attempt_01/status.json`. The old root `smoke_status.json` deliberately remains the historical failed result.
- The pre-resume snapshot, manifest, progress, failure report, and movement/environment frames that recovery may overwrite are saved in `resume_attempt_01/before_resume/`; original diagnostic log prefixes are hashed and will be checked after completion.
- The external observer in `resume_attempt_01/supervise_resume.py` captures full task-decomposition attempts (including actual persona, parent task and duration) to `contract_trace.jsonl`, plus other rejected contracts. Hash seed is set before process startup so re-exec does not discard the observer. An offline check confirmed capture works and neither simulation RNG nor runtime source digest changes. No task-decomposition behavior or model settings were changed.
- On completion, validation results go to `resume_attempt_01/acceptance_005040_resumed.json`, explicitly labeled **completed after user-requested resume**, with one prior abort and `original_zero_abort_acceptance_passed=false`. Replayed API usage remains accounted for.
- The continuation supervisor stops on any new failure without another automatic retry. A temporary follow-up will report this attempt's actual result and then stop.

### Completed recovery validation (2026-10-25 21:58 America/Los_Angeles)

The continuation exited with code 0 and is **completed_after_user_requested_resume**. The final snapshot is paused at **step 5040 / Day 1 14:00**, with `snapshot_valid=true`, `manifest_matches=true`, validated schedules for all 25 agents, valid allocation audits, complete revision debug traces, and unchanged pre-resume log prefixes. The original failure remains in the historical status and failure review. The current result is `resume_attempt_01/acceptance_005040_resumed.json`: `validation_passed=true`, `prior_abort_count=1`, and `original_zero_abort_acceptance_passed=false`.

Across the recorded run and continuation, **101 model-priority revisions passed on attempt 1**, with zero rejected or exhausted revision attempts; one additional allocation used the logged single-source shortcut. Audit lineage was checked against the pre-resume byte boundary: 44 allocations preceded recovery, 58 followed, and none of the 44 fell in the discarded interval at/after step 3973. Thus the 102 allocation records do not double-count rolled-back schedule revisions. This does not generalize to API usage: replayed requests remain in the usage ledger as actual billed work.

| Metric at Day 1 14:00 | v4 baseline | v5 after authorized recovery |
|---|---:|---:|
| Actual public-place visitors | 19/25 | 21/25 |
| Agents selecting public destinations | 20/25 | 21/25 |
| Public sectors actually reached | 6 | 6 |
| Cumulative edges | 30 | 24 |
| Nodes with nonzero betweenness | 16 | 10 |
| Largest connected component | 21 | 18 |

The predeclared mobility/network functional checks passed. These are different stochastic trajectories; this comparison is descriptive and does not establish that the allocation policy increases or decreases connectivity.

The recovery observer captured 57 task-decomposition attempts: 56 accepted responses and one rejected first attempt that recovered on attempt 2. At step 4698, Mei Lin's 60-minute task initially returned a single 60-minute subtask (violating the minimum of two subtasks); the retry returned four parts totaling 60. Four event-triple rejections also occurred without aborting the recovery. No task-decomposition sum error recurred in the captured continuation, but its original arithmetic failure was **not repaired**, and this recovery does not prove uninterrupted reliability. No rejected priority responses exist to add as ranking fixtures.

Final evidence: `pilot_runs/smoke_mobility_v5/resume_attempt_01/status.json`, `acceptance_005040_resumed.json`, `contract_trace.jsonl`, and `before_resume/`; the original 10:00 acceptance remains at the run root. The requested continuation and 14:00 post-recovery checks are complete. No further simulation was started, and the temporary follow-up is now stopped.

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
