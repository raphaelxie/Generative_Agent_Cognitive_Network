"""Integration checks against the real 25-agent template; no paid API calls."""
import copy
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import random
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("OPENAI_API_KEY", "test-only-not-a-key")
os.environ["NCN_PILOT"] = "1"
os.environ["OPENAI_MODEL"] = "gpt-6-luna"
os.environ["GA_DEBUG"] = "0"


@pytest.fixture
def engine(tmp_path, monkeypatch):
    project = Path(os.environ.get("NCN_TEST_PROJECT", str(BACKEND.parent.parent)))
    template = project/"environment/frontend_server/storage/base_the_ville_n25"
    if not template.exists():
        pytest.skip("Real template assets unavailable; set NCN_TEST_PROJECT")
    monkeypatch.chdir(BACKEND)
    import reverie
    import maze
    from persona.prompt_template import gpt_structure
    # utils.py loads the project .env with override=True; the real runner re-pins Luna.
    monkeypatch.setattr(gpt_structure, "DEFAULT_CHAT_MODEL", "gpt-6-luna")
    from run_midnight_pilot import new_server
    monkeypatch.setattr(reverie, "fs_storage", str(tmp_path/"worlds"))
    monkeypatch.setattr(reverie, "fs_temp_storage", str(tmp_path/"frontend_temp"))
    monkeypatch.setattr(maze, "env_matrix", str(project/"environment/frontend_server/static_dirs/assets/the_ville/matrix"))
    (tmp_path/"frontend_temp").mkdir()
    args = SimpleNamespace(template=template, seed=42)
    server, protocol = new_server(args, tmp_path, {"seed": 42, "model": "gpt-6-luna", "survey_repeats": 3})
    return server, protocol, tmp_path


def test_real_midnight_initialization_and_full_snapshot(engine):
    from pilot_state import save_snapshot, load_snapshot, state_digest
    server, protocol, output = engine
    assert len(server.personas) == 25 and server.curr_time.hour == 0
    assert server.step == 0 and all(not p.a_mem.seq_chat for p in server.personas.values())
    assert all(not p.a_mem.id_to_node for p in server.personas.values())
    server.personas["Hailey Johnson"].isolated = True
    server.game_obj_cleanup = {("test:bed", "is", "used", "used"): (10, 10)}
    path = output/"snapshot.pkl.gz"
    expected = save_snapshot(path, server, protocol)
    loaded = load_snapshot(path)
    assert state_digest(loaded["server"]) == expected
    assert loaded["server"].personas["Hailey Johnson"].isolated
    assert loaded["server"].game_obj_cleanup == server.game_obj_cleanup


def replay_move(self, maze, personas, tile, now):
    """Fixed cognitive response fixture; exercise real world stepping and RNG."""
    self.scratch.curr_time = now
    self.scratch.curr_tile = tile
    self.scratch.act_description = "resting"
    self.scratch.thought_count = random.randrange(1, 99) + int(np.random.randint(1, 99))
    return tile, "z", "resting"


def test_real_engine_continuous_chunked_and_restored_equivalence(engine, monkeypatch):
    from persona.persona import Persona
    from pilot_state import save_snapshot, load_snapshot, state_digest, materialize_environment, rng_state, restore_rng
    server, protocol, output = engine
    monkeypatch.setattr(Persona, "move", replay_move)
    initial = output/"initial.pkl.gz"
    save_snapshot(initial, server, protocol)
    server.start_server(3)
    expected = state_digest(server)
    resumed = load_snapshot(initial)["server"]
    materialize_environment(resumed, output/"worlds/control")
    resumed.start_server(1)
    mid = output/"mid.pkl.gz"
    save_snapshot(mid, resumed, protocol)
    resumed = load_snapshot(mid)["server"]
    materialize_environment(resumed, output/"worlds/control")
    resumed.start_server(2)
    assert state_digest(resumed) == expected


def test_operational_pause_saves_complete_tick_and_resumes_exactly(engine, monkeypatch):
    import run_midnight_pilot as runner
    import pilot_report
    from persona.persona import Persona
    from pilot_control import SessionPaused
    from pilot_state import load_snapshot, state_digest
    server, protocol, output = engine
    config = protocol["config"]
    (output/"preflight.json").write_text(json.dumps({"config": config}))
    monkeypatch.setattr(Persona, "move", replay_move)
    monkeypatch.setattr(runner, "new_server", lambda *args: (server, protocol))
    monkeypatch.setattr(pilot_report, "write_report", lambda *args: None)
    calls = []
    original_start = type(server).start_server
    def complete_one_tick(self, count):
        calls.append(count)
        return original_start(self, count)
    monkeypatch.setattr(type(server), "start_server", complete_one_tick)
    class Control:
        def check(self, phase):
            if calls:
                raise SessionPaused("test_pause")
    monkeypatch.setattr(runner, "SessionControl", lambda *args: Control())
    args = SimpleNamespace(defer_surveys=True)
    runner.run(args, output, config)
    verified = runner.verify_progress(output, config)
    assert verified["step"] == 1 and verified["status"] == "paused"
    assert verified["snapshot_valid"] and verified["manifest_matches"]
    payload = load_snapshot(output/"latest.pkl.gz")
    assert list(payload["protocol"]["snapshots"]) == ["0"]
    assert [entry["step"] for entry in payload["protocol"]["network_history"]] == [0]
    expected_server = payload["server"]
    original_start(expected_server, 1)
    expected = state_digest(expected_server)
    calls.clear()
    runner.run(args, output, config)
    payload = load_snapshot(output/"latest.pkl.gz")
    assert payload["server"].step == 2 and calls == [1]
    assert state_digest(payload["server"]) == expected


def test_real_persona_detects_midnight_date_change(engine, monkeypatch):
    from persona.persona import Persona
    server, protocol, output = engine
    seen = []
    monkeypatch.setattr(Persona, "perceive", lambda *args: [])
    monkeypatch.setattr(Persona, "retrieve", lambda *args: {})
    monkeypatch.setattr(Persona, "plan", lambda self, maze, personas, new_day, retrieved: seen.append(new_day))
    monkeypatch.setattr(Persona, "reflect", lambda *args: None)
    monkeypatch.setattr(Persona, "execute", lambda self, *args: (self.scratch.curr_tile, "z", "resting"))
    persona = server.personas["Hailey Johnson"]
    persona.move(server.maze, server.personas, (10, 10), datetime(2023, 2, 13, 0, 0))
    persona.move(server.maze, server.personas, (10, 10), datetime(2023, 2, 13, 23, 59, 50))
    persona.move(server.maze, server.personas, (10, 10), datetime(2023, 2, 14, 0, 0))
    assert seen == ["First day", False, "New day"]


def test_actual_survey_worker_in_fresh_process_with_fixed_api_responses(engine, monkeypatch):
    import subprocess
    from pilot_state import save_snapshot, state_digest
    server, protocol, output = engine
    path = output/"initial.pkl.gz"
    save_snapshot(path, server, protocol)
    before = state_digest(server)
    source = path.read_bytes()
    code = '''import sys, json, random
from pilot_state import load_snapshot
payload=load_snapshot(sys.argv[1])
roster=sorted(payload['server'].personas)
from persona.prompt_template import gpt_structure
import perception_survey
def fake_chat(prompt, **kwargs):
    if 'Pairs: ' in prompt:
        return json.dumps({'p': [.5]*len(json.loads(prompt.split('Pairs: ')[-1]))})
    if 'Partition everyone' in prompt:
        return json.dumps({'groups': [[name] for name in roster]})
    return json.dumps({'ranking': roster})
def malicious_retrieval(persona, *args, **kwargs):
    assert not hasattr(persona.scratch, 'survey_answer')
    persona.scratch.survey_answer='This must not enter a later survey'
    persona.scratch.daily_req.append('survey reminder')
    random.random()
    return '(No specific memories found.)', {'nodes': []}
gpt_structure._chat=fake_chat
perception_survey._retrieve_with_diagnostics=malicious_retrieval
from pilot_survey import survey_worker
survey_worker(sys.argv[1],sys.argv[2],int(sys.argv[3]))
'''
    for repeat in (1, 2):
        destination = output/f"survey_{repeat}"
        subprocess.run([sys.executable, "-c", code, str(path), str(destination), str(repeat)],
                       cwd=BACKEND, check=True, env=os.environ.copy())
        assert (destination/"complete.json").exists()
        assert len(list(destination.glob("respondent_*.json"))) == 25
        answer = json.loads((destination/"respondent_00.json").read_text())
        assert len(answer["interaction"]) == 300
    assert state_digest(server) == before
    assert path.read_bytes() == source


def test_stop_at_step_pauses_at_a_saved_boundary_without_touching_the_config(engine, monkeypatch):
    import run_midnight_pilot as runner
    import pilot_report
    from persona.persona import Persona
    server, protocol, output = engine
    config = protocol["config"]
    (output/"preflight.json").write_text(json.dumps({"config": config}))
    monkeypatch.setattr(Persona, "move", replay_move)
    monkeypatch.setattr(runner, "new_server", lambda *args: (server, protocol))
    monkeypatch.setattr(pilot_report, "write_report", lambda *args: None)
    runner.run(SimpleNamespace(defer_surveys=True, stop_at_step=runner.CHECK_INTERVAL), output, config)
    verified = runner.verify_progress(output, config)
    assert verified["step"] == runner.CHECK_INTERVAL and verified["status"] == "paused"
    assert verified["snapshot_valid"] and verified["manifest_matches"]
    assert "stop_at_step" not in json.dumps(config)
    # The same configuration resumes without a stop limit.
    assert runner.verify_progress(output, config)["status"] == "paused"
