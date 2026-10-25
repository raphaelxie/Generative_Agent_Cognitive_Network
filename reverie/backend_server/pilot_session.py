#!/usr/bin/env python3
"""Supervise one explicitly dated pilot session, with macOS idle-sleep prevention.

No scheduler is installed. This process must remain running for the session.
The simulation runner handles graceful pauses; the supervisor bounds a stuck tick.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from pilot_control import aware_time
from pilot_state import atomic_json

BACKEND = Path(__file__).resolve().parent
PROJECT = BACKEND.parent.parent


def terminate_group(process):
    """Only terminate the fresh process group created by this supervisor."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def validate_deadlines(cutoff, stop, hard, now):
    if not now < cutoff <= stop < hard:
        raise ValueError("Require now < survey cutoff <= graceful stop < hard stop; use today's explicit dates")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--survey-cutoff", required=True)
    parser.add_argument("--stop-at", required=True)
    parser.add_argument("--hard-stop-at", required=True)
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    cutoff, stop, hard = map(aware_time, (args.survey_cutoff, args.stop_at, args.hard_stop_at))
    validate_deadlines(cutoff, stop, hard, datetime.now(timezone.utc))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/"STOP_REQUESTED").exists():
        raise ValueError("STOP_REQUESTED exists; inspect the previous session before explicitly removing it")

    import fcntl
    with (output/".session.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Refuse to supervise an output already owned by another runner.
        with (output/".pilot.lock").open("a") as pilot_lock:
            fcntl.flock(pilot_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(pilot_lock.fileno(), fcntl.LOCK_UN)
        record = {"started_at": datetime.now(timezone.utc).isoformat(),
                  "supervisor_pid": os.getpid(), "output": str(output),
                  "survey_cutoff": args.survey_cutoff, "stop_at": args.stop_at,
                  "hard_stop_at": args.hard_stop_at, "status": "starting"}
        command = [sys.executable, "-u", str(BACKEND/"run_midnight_pilot.py"),
                   "run", "--output", str(output), "--seed", str(args.seed),
                   "--repeats", str(args.repeats), "--survey-cutoff", args.survey_cutoff,
                   "--stop-at", args.stop_at, "--stop-file", str(output/"STOP_REQUESTED")]
        caffeinate = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-s", "-w", str(os.getpid())])
        process = None
        hard_stopped = False
        try:
            with (output/"run.log").open("a") as log:
                process = subprocess.Popen(command, cwd=PROJECT, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                record.update(status="running", runner_pid=process.pid, caffeinate_pid=caffeinate.pid)
                atomic_json(output/"session.json", record)
                while process.poll() is None:
                    if caffeinate.poll() is not None:
                        # No silent loss of the requested sleep protection.
                        (output/"STOP_REQUESTED").touch()
                        record["caffeinate_ended_early"] = True
                    if datetime.now(timezone.utc) >= hard:
                        hard_stopped = True
                        terminate_group(process)
                        break
                    time.sleep(1)
            record.update(runner_exit_code=process.returncode, hard_stopped=hard_stopped,
                          runner_ended_at=datetime.now(timezone.utc).isoformat())
            # Verify the actual on-disk checkpoint, not merely a "saved" log line.
            verification = subprocess.run(
                [sys.executable, str(BACKEND/"run_midnight_pilot.py"), "verify",
                 "--output", str(output), "--seed", str(args.seed), "--repeats", str(args.repeats)],
                cwd=PROJECT, capture_output=True, text=True, timeout=120)
            record["verification_exit_code"] = verification.returncode
            if verification.returncode == 0:
                record["verified_progress"] = json.loads(verification.stdout)
                record["status"] = ("recovery_verified_after_interruption"
                                    if hard_stopped or process.returncode else "saved_and_verified")
                if not record["verified_progress"]["manifest_matches"]:
                    record["status"] = "recovery_verified_manifest_stale"
            else:
                record["status"] = "verification_failed"
                # Provider credentials and exception bodies never enter the handoff.
                record["verification_error"] = "Inspect latest.pkl.gz and manifest.json; do not claim save succeeded"
        except BaseException:
            if process is not None:
                terminate_group(process)
            record["status"] = "supervisor_error"
            raise
        finally:
            if caffeinate.poll() is None:
                caffeinate.terminate()
                caffeinate.wait(timeout=5)
            record["ended_at"] = datetime.now(timezone.utc).isoformat()
            atomic_json(output/"session.json", record)
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0 if record["status"] == "saved_and_verified" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, BlockingIOError) as error:
        print(f"Session did not start: {error}", file=sys.stderr)
        sys.exit(2)
