"""Named 16:00 isolation branches stay outside the frozen runtime digest."""
from datetime import datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

BACKEND = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("OPENAI_API_KEY", "test-only-not-a-key")
os.environ["NCN_PILOT"] = "1"
os.environ["OPENAI_MODEL"] = "gpt-6-luna"

import run_isolation_branch as branch
import run_midnight_pilot as runner
from pilot_protocol import CHECK_INTERVAL
from pilot_state import load_snapshot, save_snapshot, state_digest
from test_pilot_engine import engine, replay_move  # noqa: F401


EXPECTED = "29cfd946e0d44df4cddc0f3938bb74a3ac12875ec3e81f5d27c379c8b1cd2813"
JM, CG = branch.JM, branch.CG


def test_branch_entry_is_outside_the_frozen_digest_and_does_not_change_it():
    files = sorted(set(BACKEND.glob("pilot_*.py")) | {BACKEND / "run_midnight_pilot.py",
                   BACKEND / "reverie.py", BACKEND / "utils.py"}
                   | set((BACKEND / "persona").rglob("*.py"))
                   | set((BACKEND / "persona/prompt_template").rglob("*.txt")))
    relative = {str(path.relative_to(BACKEND)) for path in files}
    assert "run_isolation_branch.py" not in relative
    assert "css_cumulative_survey.py" not in relative
    assert "measure_css_survey.py" not in relative
    assert runner.source_digest() == EXPECTED == branch.EXPECTED_DIGEST


def test_named_targets_are_the_unique_degree_pair_and_unique_betweenness_leader():
    degrees = {JM: 8, CG: 8, "Ada": 3, "Bo": 2}
    between = {JM: 0.24, CG: 0.20, "Ada": 0.01, "Bo": 0.0}
    report = branch.evaluate_targets(degrees, between)
    assert report["ok"] and report["degree_leaders"] == [CG, JM]
    between[CG] = 0.24
    assert not branch.evaluate_targets(degrees, between)["jennifer_unique_max_betweenness"]
    degrees["Ada"] = 8
    assert not branch.evaluate_targets(degrees, {**between, CG: 0.20})["degree_leaders_are_jennifer_and_carlos"]
    degrees["Ada"] = 9
    assert not branch.evaluate_targets(degrees, {JM: 0.24, CG: 0.20, "Ada": 0.0, "Bo": 0.0})["ok"]


def _persona(name, isolated=False, hour=15):
    scratch = SimpleNamespace(
        act_address="the Ville:Hobbs Cafe:cafe", act_description="serving coffee",
        chatting_with=None, chatting_with_buffer={}, planned_path=["x"],
        curr_time=datetime(2023, 2, 13, hour))
    return SimpleNamespace(name=name, isolated=isolated, scratch=scratch)


def test_block_recorder_logs_the_first_isolation_gate_without_calling_the_model(tmp_path):
    import persona.cognitive_modules.plan as plan_mod
    original = plan_mod._should_react
    initiator, target = _persona(JM, isolated=True), _persona(CG)
    personas = {JM: initiator, CG: target}
    retrieved = {"curr_event": SimpleNamespace(subject=CG)}
    protocol = {"intervention": {"first_blocked": None}}
    server = SimpleNamespace(step=5761, curr_time=datetime(2023, 2, 13, 16, 0, 10))
    called = []

    def forbidden(*args, **kwargs):
        raise AssertionError("isolation must not ask the model whether to talk")

    context = {"server": server, "protocol": protocol, "output": tmp_path}
    try:
        with patch.object(plan_mod, "generate_decide_to_talk", forbidden), \
             patch.object(plan_mod, "generate_decide_to_react", forbidden):
            wrapped = branch.install_block_recorder(context)
            assert wrapped(initiator, retrieved, personas) is False
            record = protocol["intervention"]["first_blocked"]
            assert record["kind"] == "talk" and record["initiator"] == JM and record["target"] == CG
            assert json.loads((tmp_path / "intervention.json").read_text())["first_blocked"] == record
            called.append(record)
            wrapped(initiator, retrieved, personas)
            assert protocol["intervention"]["first_blocked"] is called[0]
    finally:
        plan_mod._should_react = original


def _prepare_origin(engine_bundle, monkeypatch):
    import pilot_report
    from persona.persona import Persona
    server, protocol, output = engine_bundle
    monkeypatch.setattr(Persona, "move", replay_move)
    monkeypatch.setattr(pilot_report, "write_report", lambda *args: None)
    config = protocol["config"]
    (output / "preflight.json").write_text(json.dumps({"config": config}))
    origin = output / "snapshots" / "step_000000.pkl.gz"
    save_snapshot(origin, server, protocol)
    protocol["snapshots"] = {"0": {"path": "snapshots/step_000000.pkl.gz", "labels": ["initial"]}}
    runner.save_progress(output, server, protocol)
    return server, protocol, output, config


def test_replay_then_named_fork_keeps_origin_unmeasured_and_control_unisolated(engine, tmp_path, monkeypatch):
    server, protocol, source, config = _prepare_origin(engine, monkeypatch)
    replay = tmp_path / "replay"
    branch.seed_from_snapshot(source, replay, config, step=0)
    payload = load_snapshot(replay / "latest.pkl.gz", restore_random=False)
    assert payload["server"].step == 0
    assert json.loads((replay / "manifest.json").read_text())["snapshots"]["0"]["labels"] == ["replay_origin"]
    args = SimpleNamespace(stop_at=None, survey_cutoff=None, stop_file=None, stop_at_step=None)
    branch.advance(args, replay, config, CHECK_INTERVAL, {CHECK_INTERVAL: ["checkpoint"]},
                   allow_isolation=False, complete_status="checkpoint_ready")
    manifest = json.loads((replay / "manifest.json").read_text())
    assert manifest["status"] == "checkpoint_ready"
    assert manifest["step"] == CHECK_INTERVAL
    assert manifest["snapshots"][str(CHECK_INTERVAL)]["labels"] == ["checkpoint"]
    assert "scheduled" not in manifest["snapshots"]["0"]["labels"]
    jm = tmp_path / "JM"
    intervention = branch.fork_arm(replay, jm, "JM", config, step=CHECK_INTERVAL)
    loaded = load_snapshot(jm / "latest.pkl.gz", restore_random=False)
    assert intervention["target"] == JM
    assert loaded["server"].personas[JM].isolated
    assert not loaded["server"].personas["Isabella Rodriguez"].isolated
    assert json.loads((jm / "manifest.json").read_text())["snapshots"][str(CHECK_INTERVAL)]["labels"] == ["branch_origin"]
    control = tmp_path / "control"
    branch.fork_arm(replay, control, "control", config, step=CHECK_INTERVAL)
    control_state = load_snapshot(control / "latest.pkl.gz", restore_random=False)
    assert branch.isolated_names(control_state["server"]) == []
    import persona.cognitive_modules.plan as plan_mod
    original_react = plan_mod._should_react
    try:
        branch.advance(args, jm, config, CHECK_INTERVAL + 1, {CHECK_INTERVAL: ["branch_origin"]},
                       allow_isolation=True, complete_status="simulation_complete_surveys_pending")
    finally:
        plan_mod._should_react = original_react
    resumed = load_snapshot(jm / "latest.pkl.gz", restore_random=False)
    assert resumed["server"].personas[JM].isolated
    assert resumed["server"].step == CHECK_INTERVAL + 1


def test_official_runner_still_refuses_an_isolated_state(engine, monkeypatch):
    import pilot_report
    from persona.persona import Persona
    server, protocol, output = engine
    monkeypatch.setattr(Persona, "move", replay_move)
    monkeypatch.setattr(pilot_report, "write_report", lambda *args: None)
    config = protocol["config"]
    (output / "preflight.json").write_text(json.dumps({"config": config}))
    server.personas[JM].isolated = True
    save_snapshot(output / "latest.pkl.gz", server, protocol)
    with pytest.raises(ValueError, match="unperturbed"):
        runner.run(SimpleNamespace(defer_surveys=True, stop_at=None, survey_cutoff=None,
                                   stop_file=None, stop_at_step=None), output, config)
    assert load_snapshot(output / "latest.pkl.gz", restore_random=False)["server"].personas[JM].isolated


def test_resume_does_not_reseed_an_existing_replay(engine, tmp_path, monkeypatch):
    server, protocol, source, config = _prepare_origin(engine, monkeypatch)
    replay = tmp_path / "replay"
    branch.seed_from_snapshot(source, replay, config, step=0)
    before = (replay / "latest.pkl.gz").read_bytes()
    branch.seed_from_snapshot(source, replay, config, step=0)
    assert (replay / "latest.pkl.gz").read_bytes() == before
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "stray.txt").write_text("no")
    with pytest.raises(FileExistsError):
        branch.seed_from_snapshot(source, occupied, config, step=0)
    lock_only = tmp_path / "lock_only"
    lock_only.mkdir()
    (lock_only / ".pilot.lock").write_text("")
    branch.seed_from_snapshot(source, lock_only, config, step=0)
    assert (lock_only / "latest.pkl.gz").exists()


def test_fork_ignores_configure_scaffolding_and_refuses_a_real_fork(engine, tmp_path, monkeypatch):
    server, protocol, source, config = _prepare_origin(engine, monkeypatch)
    replay = tmp_path / "replay"
    branch.seed_from_snapshot(source, replay, config, step=0)
    args = SimpleNamespace(stop_at=None, survey_cutoff=None, stop_file=None, stop_at_step=None)
    branch.advance(args, replay, config, CHECK_INTERVAL, {CHECK_INTERVAL: ["checkpoint"]},
                   allow_isolation=False, complete_status="checkpoint_ready")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "stray.txt").write_text("no")
    with pytest.raises(FileExistsError):
        branch.fork_arm(replay, occupied, "JM", config, step=CHECK_INTERVAL)
    scaffolded = tmp_path / "scaffolded"
    scaffolded.mkdir()
    (scaffolded / ".pilot.lock").write_text("")
    (scaffolded / "frontend_temp").mkdir()
    intervention = branch.fork_arm(replay, scaffolded, "JM", config, step=CHECK_INTERVAL)
    assert intervention["target"] == JM
    assert (scaffolded / "latest.pkl.gz").exists() and (scaffolded / "branch.pkl.gz").exists()
    with pytest.raises(FileExistsError):
        branch.fork_arm(replay, scaffolded, "JM", config, step=CHECK_INTERVAL)
