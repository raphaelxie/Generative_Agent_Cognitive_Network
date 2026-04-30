# Option A Diagnostic Note (n15 salvage pass)

## 1) What froze

The network evolution froze after step 900 (10:30 AM simulated time). In `network_summary_over_time.csv`, `n_new_chats_since_prev` is:

- 3 at step 600
- 1 at step 900
- **0 at every checkpoint from step 1200 through 4200**

So the entire 11h40m run produced only 4 cumulative chats, and no new social interactions were added after late morning.

The planner-side forensic table (`planner_freeze_audit.csv`) is consistent with this freeze:

- 15/15 personas have `act_event_at_save = sleep`
- 15/15 personas have `chatting_at_save = 0`
- Across personas, there are **42 prompt-leak schedule slots** (`n_prompt_leak_slots`) and **167 zero-duration sleep filler slots** (`n_zero_duration_sleep_slots`)
- `daily_req` remains non-trivial (103 total requested tasks across personas), but realized schedule quality degrades into degenerate/filler patterns

Interpretation: this run exhibits a planner/schedule degeneration that suppresses social activity well before afternoon checkpoints.

## 2) Why this matters methodologically

This is a methodological validity issue, not just a noisy run.

- If social interaction generation freezes, later checkpoints are no longer sampling cognition under evolving network structure; they are sampling cognition under a nearly static, low-information environment.
- Internal persona state can appear superficially “alive” (time advancing, active event text present) while behaviorally the system is frozen (`sleep`, no chatting, no new ties).
- Therefore, **external behavioral checks** (new chats/ties over time, schedule degeneration flags) are required before treating any checkpoint series as valid pre/post intervention evidence.

Without these checks, one can mistakenly interpret downstream survey changes as cognitive effects when the causal substrate (network dynamics) has already collapsed.

## 3) What can still be learned from t600

Even though the full checkpoint trajectory is frozen, t600 remains usable as a baseline diagnostic wave.

From `analysis_micro_tie_t600_summary.csv` (`respondent=OVERALL`):

- precision = **0.023026**
- recall = **0.162791**
- pair_accuracy = **0.784047**
- n_fail_safe = **294**

This indicates sparse true-positive detection with high apparent accuracy driven by many true negatives (expected in sparse graphs), and non-trivial fail-safe burden.

From `self_position_calibration_t600.csv`:

- self-position responses are numeric (1–5 scale)
- mean absolute rank error = **0.933**
- mean signed rank error = **0.0**

So there is meaningful individual miscalibration, but little aggregate directional bias (no strong overall over/under-estimation direction).

From `retrieval_diagnostics_t600_summary.json` (excluding self-queries):

- n_calls = **225**
- pct_calls_with_any_target_mention = **4.44%**
- pct_calls_all_self_mention = **26.67%**
- mean_target_mention_fraction = **0.0119**

This suggests many retrieval batches do not surface target-relevant social memories, which plausibly contributes to weak micro-tie recall.

## 4) Why this does not replace the advisor’s main pre/post shock agenda

This pass is a diagnostic salvage artifact only.

- The intended broker-vs-hub contrast is structurally infeasible in this run after t900: `hub_eq_broker = 1` at every checkpoint where betweenness is non-zero (`network_summary_over_time.csv`).
- The run does not deliver the advisor’s main agenda yet (true pre/post shock identification on a probe-capable network, meso-level cognition tasks, separate construct prompts, hard-isolation condition, richer network-level tracking).
- Therefore, these outputs should be treated as **quality-control and baseline documentation**, not as a substitute for the planned pre/post shock research program.

Operationally: this note justifies why this run is informative as a failure-mode diagnosis and why the next research sprint must restart from a configuration that maintains active social dynamics long enough for valid intervention testing.