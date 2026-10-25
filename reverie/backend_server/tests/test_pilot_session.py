"""Wall-clock supervision checks; no paid API calls and no real long-running jobs."""
from datetime import datetime
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from pilot_control import SessionControl, SessionPaused, aware_time
from pilot_protocol import next_boundary
from pilot_session import validate_deadlines


def test_deadlines_use_wall_time_and_cutoff_only_stops_surveys(tmp_path):
    clock = [aware_time("2026-10-20T08:59:59-07:00")]
    control = SessionControl("2026-10-20T09:20:00-07:00", "2026-10-20T09:00:00-07:00",
                             tmp_path/"STOP_REQUESTED", now=lambda: clock[0])
    control.check("survey")
    clock[0] = aware_time("2026-10-20T16:00:00+00:00")
    control.check("simulation")
    with pytest.raises(SessionPaused, match="survey_cutoff"):
        control.check("survey")
    clock[0] = aware_time("2026-10-20T09:20:00-07:00")
    with pytest.raises(SessionPaused, match="wall_clock_stop"):
        control.check("simulation")
    (tmp_path/"STOP_REQUESTED").touch()
    with pytest.raises(SessionPaused, match="stop_file"):
        control.check("simulation")


def test_refuse_ambiguous_expired_or_reversed_session_times():
    with pytest.raises(ValueError, match="UTC offset"):
        aware_time("2026-10-20T09:00:00")
    now, cutoff, stop, hard = map(aware_time, [
        "2026-10-20T01:00:00-07:00", "2026-10-20T09:00:00-07:00",
        "2026-10-20T09:20:00-07:00", "2026-10-20T09:40:00-07:00"])
    validate_deadlines(cutoff, stop, hard, now)
    with pytest.raises(ValueError):
        validate_deadlines(cutoff, stop, hard, hard)
    with pytest.raises(ValueError):
        validate_deadlines(stop, cutoff, hard, now)


def test_resume_between_boundaries_preserves_original_grid():
    assert next_boundary(1) == 180
    assert next_boundary(179) == 180
    assert next_boundary(7199) == 7200
    assert next_boundary(7381, 7380) == 7560


def test_survey_cutoff_terminates_worker_and_keeps_immutable_source(monkeypatch, tmp_path):
    import run_midnight_pilot as runner
    path = tmp_path/"snapshot.pkl.gz"
    path.write_bytes(b"untouched snapshot")
    monkeypatch.setenv("GA_SEED", "104729")
    checks = []
    class Control:
        def check(self, phase):
            checks.append(phase)
            if len(checks) == 2:
                raise SessionPaused("survey_cutoff")
    class Worker:
        returncode = None
        terminated = False
        def poll(self):
            return self.returncode
        def terminate(self):
            self.terminated = True
            self.returncode = -15
        def wait(self, **kwargs):
            return self.returncode
    worker = Worker()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: worker)
    with pytest.raises(SessionPaused, match="survey_cutoff"):
        runner.measure_snapshot(tmp_path, path, 0, 3, control=Control())
    assert worker.terminated
    assert path.read_bytes() == b"untouched snapshot"
    assert not (tmp_path/"surveys/step_000000/isolation_check.json").exists()


def test_cutoff_before_survey_never_launches_child(monkeypatch, tmp_path):
    import run_midnight_pilot as runner
    path = tmp_path/"snapshot.pkl.gz"
    path.write_bytes(b"snapshot")
    control = SessionControl(survey_cutoff="2026-10-20T09:00:00-07:00",
                             now=lambda: aware_time("2026-10-20T09:01:00-07:00"))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Launched late survey"))
    with pytest.raises(SessionPaused):
        runner.measure_snapshot(tmp_path, path, 0, 3, control=control)


def test_supervisor_hard_stop_targets_own_runner_and_verifies_recovery(monkeypatch, tmp_path):
    import pilot_session as session
    times = iter([aware_time("2026-10-20T01:00:00-07:00"),
                  aware_time("2026-10-20T01:00:00-07:00"),
                  aware_time("2026-10-20T09:40:00-07:00")])
    class Clock:
        @staticmethod
        def now(tz):
            return next(times, aware_time("2026-10-20T09:40:01-07:00"))
    class Child:
        def __init__(self, pid):
            self.pid, self.returncode = pid, None
        def poll(self):
            return self.returncode
        def terminate(self):
            self.returncode = -15
        def wait(self, **kwargs):
            return self.returncode
    caffeinate, runner = Child(99001), Child(99002)
    commands, terminated = [], []
    def launch(command, **kwargs):
        commands.append(command)
        return caffeinate if "caffeinate" in command[0] else runner
    def stop(child):
        terminated.append(child.pid)
        child.returncode = -15
    monkeypatch.setattr(session, "datetime", Clock)
    monkeypatch.setattr(session.subprocess, "Popen", launch)
    monkeypatch.setattr(session, "terminate_group", stop)
    monkeypatch.setattr(session.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=json.dumps({"step": 42, "snapshot_valid": True, "manifest_matches": True})))
    monkeypatch.setattr(sys, "argv", ["pilot_session.py", "--output", str(tmp_path),
        "--survey-cutoff", "2026-10-20T09:00:00-07:00",
        "--stop-at", "2026-10-20T09:20:00-07:00",
        "--hard-stop-at", "2026-10-20T09:40:00-07:00"])
    assert session.main() == 1
    record = json.loads((tmp_path/"session.json").read_text())
    assert record["status"] == "recovery_verified_after_interruption"
    assert record["verified_progress"]["step"] == 42
    assert terminated == [runner.pid] and caffeinate.returncode == -15
    assert commands[0][1:4] == ["-i", "-s", "-w"]
    assert "--stop-file" in commands[1]
