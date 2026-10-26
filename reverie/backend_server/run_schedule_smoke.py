#!/usr/bin/env python3
"""Run one fresh, two-stage schedule smoke; never automatically retry a failed run."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from pilot_session import terminate_group
from pilot_state import atomic_json

BACKEND = Path(__file__).resolve().parent
PROJECT = BACKEND.parent.parent


def run_stages(output, baseline):
    from run_midnight_pilot import source_digest
    from validate_schedule_smoke import validate_stage
    # Refuse even an empty existing directory: this command always means fresh evidence.
    output.mkdir(parents=True, exist_ok=False)
    environment = {**os.environ, "PYTHONHASHSEED": "104729", "OPENAI_MODEL": "gpt-6-luna",
                   "NCN_REACTION_DEBUG_LOG": str(output/"schedule_revision_debug.jsonl")}
    record = {"status": "starting", "supervisor_pid": os.getpid(), "stages": [],
              "source_digest": source_digest(), "started_at": datetime.now(timezone.utc).isoformat()}
    process = None
    atomic_json(output/"smoke_status.json", record)
    try:
        for step in (3600, 5040):
            if source_digest() != record["source_digest"]:
                raise ValueError("Source changed between smoke stages")
            stage = {"target_step": step, "status": "running"}
            record["stages"].append(stage)
            record["status"] = "running"
            command = [sys.executable, "-u", str(BACKEND/"run_midnight_pilot.py"), "run",
                       "--output", str(output), "--seed", "104729", "--repeats", "2",
                       "--defer-surveys", "--stop-at-step", str(step)]
            with (output/"smoke_launcher.log").open("a") as log:
                process = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                stage["runner_pid"] = process.pid
                atomic_json(output/"smoke_status.json", record)
                stage["exit_code"] = process.wait()
            if stage["exit_code"]:
                raise RuntimeError(f"Runner failed at stage {step}; evidence retained, no automatic retry")
            result = validate_stage(output, baseline, step)
            atomic_json(output/f"acceptance_{step:06d}.json", result)
            stage["status"] = "passed"
            stage["finished_at"] = datetime.now(timezone.utc).isoformat()
            atomic_json(output/"smoke_status.json", record)
            print(json.dumps({"stage": step, **result}, ensure_ascii=False), flush=True)
        record["status"] = "passed"
        return 0
    except BaseException as exc:
        record["status"] = "failed"
        # Don't copy arbitrary provider exception messages or credentials to summaries.
        record["error_type"] = type(exc).__name__
        if record["stages"] and record["stages"][-1]["status"] == "running":
            record["stages"][-1]["status"] = "failed"
        raise
    finally:
        if process is not None:
            terminate_group(process)
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output/"smoke_status.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT/"pilot_runs/smoke_mobility_v5")
    parser.add_argument("--baseline", type=Path, default=PROJECT/"pilot_runs/smoke_mobility_v4")
    args = parser.parse_args()
    return run_stages(args.output.resolve(), args.baseline.resolve())


if __name__ == "__main__":
    sys.exit(main())
