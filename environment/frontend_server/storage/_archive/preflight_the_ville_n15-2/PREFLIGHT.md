# n=15 pre-flight (`preflight_the_ville_n15-1`)

Forked from `base_the_ville_n15`. Use this folder only for the cheap plumbing check before a full pilot.

## Operator sequence

1. **Environment + sim servers** (see repo [README.md](../../../../README.md)): Django `runserver`, browser on simulator home, `python reverie.py`.
2. **Load this sim** (both prompts): `preflight_the_ville_n15-1` → `preflight_the_ville_n15-1`.
3. **Burn-in**: `run 600` (requires the frontend to advance `environment/<step>.json` each step).
4. **Save state**: `save` (or `fin` if you are done with the session) so persona memory and `reverie/meta.json` persist.
5. **Measurement batch** (from `reverie/backend_server`, after save):

   ```bash
   python preflight_n15_batch.py
   ```

   This runs `survey pre` plus hub/broker shock probe (`shock isolate-hub` → `unshock` → `shock isolate-broker` → `unshock`).

6. **Analysis + verify**:

   ```bash
   python analyze_survey.py ../../environment/frontend_server/storage/preflight_the_ville_n15-1/survey
   python verify_preflight_n15.py ../../environment/frontend_server/storage/preflight_the_ville_n15-1/survey
   ```

## Interactive alternative

After burn-in, instead of step 5 you can type: `survey pre`, then the four shock commands from the plan.

## Pass / not-ready

See exit code and messages from `verify_preflight_n15.py` (matches the pre-flight plan criteria).
