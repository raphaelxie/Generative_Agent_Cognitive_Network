"""Priority allocation regressions. Synthetic cases are not historical API replies."""
import copy
from datetime import datetime
from itertools import permutations, product
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot_tasks import (REACTION_REVISION_VERSION, allocate_revision_minutes,
                         parse_revision_priority, reschedule_reaction, slice_rows, state_for)
from pilot_schedule import PilotScheduleError
from test_pilot_engine import engine
from test_pilot_tasks import setup_persona, suffix_response


def sources(durations):
    return [{"activity": f"activity {i}", "minutes": d, "terminal": bool(i % 2),
             "origin": f"origin:{i}"} for i, d in enumerate(durations)]


@pytest.mark.parametrize("value", [None, [], {}, {"priority": None}, {"priority": []},
    {"priority": [0]}, {"priority": [0, 0]}, {"priority": [0, 2]},
    {"priority": [-1, 0]}, {"priority": [False, 1]}, {"priority": [0, 1.0]},
    {"priority": ["0", 1]}, {"priority": [[0], 1]},
    {"priority": [0, 1], "extra": 0}])
def test_priority_rejects_invalid_json_shapes(value):
    with pytest.raises(ValueError):
        parse_revision_priority(json.dumps(value), 2)


def test_priority_is_not_execution_order_and_preserves_input():
    source = sources([30, 30, 30])
    before = copy.deepcopy(source)
    priority = parse_revision_priority('{"priority":[2,0,1]}', 3)
    assert allocate_revision_minutes(source, priority, 70) == [30, 10, 30]
    assert allocate_revision_minutes(source, priority, 30) == [0, 0, 30]
    assert source == before and priority == [2, 0, 1]


def test_exhaustive_small_allocations_preserve_capacity_and_priority():
    for n in range(1, 5):
        for caps in product(range(1, 5), repeat=n):
            source = sources(caps)
            for order in permutations(range(n)):
                for budget in range(sum(caps) + 1):
                    allocated = allocate_revision_minutes(source, list(order), budget)
                    assert sum(allocated) == budget
                    assert all(0 <= a <= cap for a, cap in zip(allocated, caps))
                    assert sum(0 < a < cap for a, cap in zip(allocated, caps)) <= 1
                    unfinished = False
                    for i in order:
                        if unfinished:
                            assert allocated[i] == 0
                        if allocated[i] < caps[i]:
                            unfinished = True
    assert allocate_revision_minutes([], [], 0) == []


@pytest.mark.parametrize("budget", [-1, True, 1.5, 61, None])
def test_invalid_local_budget_is_fatal(budget):
    with pytest.raises(PilotScheduleError, match="Invalid revision interval"):
        allocate_revision_minutes(sources([30, 30]), [0, 1], budget)


@pytest.mark.parametrize("caps", [[0], [True], [-1], [1441], [1000, 1000]])
def test_invalid_local_capacity_is_fatal(caps):
    with pytest.raises(PilotScheduleError):
        allocate_revision_minutes(sources(caps), list(range(len(caps))), 1)


def setup_two(engine, name="Latoya Williams", now=None):
    return setup_persona(engine, name=name, now=now, schedule=[
        ["sleeping", 480], ["painting", 60], ["reading", 60], ["resting", 840]])


def diagnostic_paths(engine, monkeypatch, debug=False):
    paths = [engine[2]/name for name in ("contracts.jsonl", "audit.jsonl", "debug.jsonl")]
    monkeypatch.setenv("NCN_SCHEDULE_LOG", str(paths[0]))
    monkeypatch.setenv("NCN_REACTION_AUDIT_LOG", str(paths[1]))
    monkeypatch.delenv("NCN_REACTION_DEBUG_LOG", raising=False)
    if debug:
        monkeypatch.setenv("NCN_REACTION_DEBUG_LOG", str(paths[2]))
    return paths


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_revision_trace_replays_allocation_without_affecting_rng(engine, monkeypatch):
    from persona.prompt_template import gpt_structure
    from pilot_state import rng_state, _canonical
    p = setup_two(engine, now=datetime(2023, 2, 13, 8, 5, 10))
    paths = diagnostic_paths(engine, monkeypatch, debug=True)
    replies = iter(['{"priority":[0,0]}', '{"priority":[1,0]}'])
    monkeypatch.setattr(gpt_structure, "_chat", lambda *a, **k: next(replies))
    before = copy.deepcopy(state_for(p.scratch)["rows"])
    rng = _canonical(rng_state())
    reschedule_reaction(p, "talk with a neighbor", 5)
    assert _canonical(rng_state()) == rng
    contracts, audit, debug = map(read_records, paths)
    assert [r["status"] for r in contracts] == ["rejected", "accepted"]
    assert len(audit) == 1 and len(debug) == 2
    assert len({r["revision_id"] for r in contracts + audit + debug}) == 1
    assert all(r["persona"] == p.name for r in contracts + audit + debug)
    assert all(r["reaction_revision_version"] == REACTION_REVISION_VERSION for r in contracts + audit + debug)
    assert all("prompt" not in r and "raw" not in r for r in contracts)
    assert "missing=[1]" in debug[0]["error"]
    assert debug[1]["raw"] == '{"priority":[1,0]}'
    row = audit[0]
    assert row["start"] == 486 and row["remaining_minutes"] == 109
    assert row["allocated_minutes"] == allocate_revision_minutes(row["source"], row["priority"], 109)
    assert row["allocated_minutes"] == [49, 60]
    after = state_for(p.scratch)["rows"]
    assert slice_rows(after, 0, 486) == slice_rows(before, 0, 486)
    assert slice_rows(after, 600, 1440) == slice_rows(before, 600, 1440)
    assert slice_rows(after, 491, 600) == row["suffix"]
    assert [{k: v for k, v in r.items() if k != "minutes"} for r in row["suffix"]] == [
        {k: v for k, v in r.items() if k != "minutes"} for r in row["source"]]


def test_single_source_is_logged_without_model_or_debug_text(engine, monkeypatch):
    from persona.prompt_template import gpt_structure
    p = setup_persona(engine)
    paths = diagnostic_paths(engine, monkeypatch)
    monkeypatch.setattr(gpt_structure, "_chat", lambda *a, **k: pytest.fail("No model choice"))
    reschedule_reaction(p, "talk with a neighbor", 5)
    row = read_records(paths[1])[0]
    assert row["decision"] == "only_feasible_choice"
    assert row["priority"] == [0] and row["allocated_minutes"] == [115]
    assert not paths[0].exists() and not paths[2].exists()


def test_debug_is_off_by_default_for_model_decision(engine, monkeypatch):
    from persona.prompt_template import gpt_structure
    p = setup_two(engine)
    paths = diagnostic_paths(engine, monkeypatch)
    monkeypatch.setattr(gpt_structure, "_chat", suffix_response)
    reschedule_reaction(p, "talk with a neighbor", 5)
    assert paths[0].exists() and paths[1].exists() and not paths[2].exists()


@pytest.mark.parametrize("failure", ["format", "transport"])
def test_revision_failures_preserve_scratch_and_bound_calls(engine, monkeypatch, failure):
    from persona.prompt_template import gpt_structure
    from pilot_api import PilotAPIError
    p = setup_two(engine)
    paths = diagnostic_paths(engine, monkeypatch, debug=True)
    before, calls = copy.deepcopy(vars(p.scratch)), []
    def bad(*args, **kwargs):
        calls.append(args)
        if failure == "transport":
            raise PilotAPIError("injected")
        return '{"priority":[0]}'
    monkeypatch.setattr(gpt_structure, "_chat", bad)
    with pytest.raises((PilotScheduleError, PilotAPIError)):
        reschedule_reaction(p, "talk with a neighbor", 5)
    assert vars(p.scratch) == before
    assert len(calls) == (3 if failure == "format" else 1)
    assert not paths[1].exists()
    if failure == "format":
        assert len(read_records(paths[2])) == 3


def test_bad_local_interval_never_calls_model(engine, monkeypatch):
    import pilot_tasks
    from persona.prompt_template import gpt_structure
    p = setup_two(engine)
    before = copy.deepcopy(vars(p.scratch))
    monkeypatch.setattr(pilot_tasks, "slice_rows", lambda *a: sources([1]))
    monkeypatch.setattr(gpt_structure, "_chat", lambda *a, **k: pytest.fail("Invalid local budget"))
    with pytest.raises(PilotScheduleError, match="Invalid revision interval"):
        reschedule_reaction(p, "talk with a neighbor", 5)
    assert vars(p.scratch) == before


def test_actual_chat_logs_each_recipient_and_wait_uses_same_contract(engine, monkeypatch):
    from persona.cognitive_modules import plan
    from persona.prompt_template import gpt_structure
    p = setup_two(engine)
    q = setup_two(engine, "Rajiv Patel")
    paths = diagnostic_paths(engine, monkeypatch)
    monkeypatch.setattr(gpt_structure, "_chat", suffix_response)
    monkeypatch.setattr(plan, "generate_convo", lambda *a: ([[p.name, "Hello"]], 5))
    monkeypatch.setattr(plan, "generate_convo_summary", lambda *a: "talk about painting")
    plan._chat_react(engine[0].maze, p, {}, f"chat with {q.name}", engine[0].personas)
    assert {r["persona"] for r in read_records(paths[0])} == {p.name, q.name}
    assert len({r["revision_id"] for r in read_records(paths[1])}) == 2
    p.scratch.curr_time = datetime(2023, 2, 13, 8, 10)
    plan._wait_react(p, "wait: February 13, 2023, 08:15:00")
    last = read_records(paths[1])[-1]
    assert last["persona"] == p.name and last["inserted_minutes"] == 5
    assert last["inserted_activity"].startswith("waiting to start")


def test_replanned_snapshot_restore_has_identical_next_allocation(engine, monkeypatch):
    from persona.prompt_template import gpt_structure
    from pilot_state import load_snapshot, save_snapshot
    monkeypatch.setattr(gpt_structure, "_chat", suffix_response)
    p = setup_two(engine)
    reschedule_reaction(p, "talk with a neighbor", 5)
    path = engine[2]/"replanned.pkl.gz"
    save_snapshot(path, engine[0], engine[1])
    q = load_snapshot(path)["server"].personas[p.name]
    for persona in (p, q):
        persona.scratch.curr_time = datetime(2023, 2, 13, 8, 10)
        reschedule_reaction(persona, "wait for a neighbor", 3)
    assert vars(p.scratch) == vars(q.scratch)


def test_stage_auditor_rejects_tampering_and_unresolved_attempts(engine, monkeypatch):
    from persona.prompt_template import gpt_structure
    from validate_schedule_smoke import validate_audit
    p = setup_two(engine)
    paths = diagnostic_paths(engine, monkeypatch, debug=True)
    monkeypatch.setattr(gpt_structure, "_chat", suffix_response)
    reschedule_reaction(p, "talk with a neighbor", 5)
    records = list(map(read_records, paths))
    assert validate_audit(*records)["audit_valid"]
    for field, value in [("allocated_minutes", [60, 60]), ("priority", [1, 0]),
                         ("persona", "different recipient"), ("suffix", [])]:
        tampered = copy.deepcopy(records)
        tampered[1][0][field] = value
        with pytest.raises(ValueError):
            validate_audit(*tampered)
    with pytest.raises(ValueError, match="Incomplete debug trace"):
        validate_audit(records[0], records[1], [])
    records[0][0]["status"] = "rejected"
    with pytest.raises(ValueError, match="Exhausted or unresolved"):
        validate_audit(*records)


@pytest.mark.parametrize("failure", [None, "runner", "gate", "source"])
def test_smoke_supervisor_only_advances_after_success(tmp_path, monkeypatch, failure):
    import run_schedule_smoke as smoke
    import run_midnight_pilot as runner
    import validate_schedule_smoke as validator
    calls, checks, digests = [], [], []
    def digest():
        digests.append(1)
        return "changed" if failure == "source" and len(digests) > 2 else "fixed"
    monkeypatch.setattr(runner, "source_digest", digest)
    class Process:
        pid = 123
        def __init__(self, command, **kwargs):
            calls.append(int(command[-1]))
            assert kwargs["env"]["PYTHONHASHSEED"] == "104729"
            assert kwargs["env"]["NCN_REACTION_DEBUG_LOG"].endswith("schedule_revision_debug.jsonl")
        def wait(self):
            return 1 if failure == "runner" else 0
    monkeypatch.setattr(smoke.subprocess, "Popen", Process)
    monkeypatch.setattr(smoke, "terminate_group", lambda *a: None)
    def validate(output, baseline, step):
        checks.append(step)
        if failure == "gate":
            raise ValueError("injected gate failure")
        return {"passed": True}
    monkeypatch.setattr(validator, "validate_stage", validate)
    output = tmp_path/"fresh"
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            smoke.run_stages(output, tmp_path/"baseline")
        assert calls == [3600]
    else:
        assert smoke.run_stages(output, tmp_path/"baseline") == 0
        assert calls == checks == [3600, 5040]
    status = json.loads((output/"smoke_status.json").read_text())
    assert status["status"] == ("failed" if failure else "passed")
    with pytest.raises(FileExistsError):
        smoke.run_stages(output, tmp_path/"baseline")


def test_configuration_locks_revision_version(monkeypatch):
    from types import SimpleNamespace
    import run_midnight_pilot as runner
    config = runner.configuration(SimpleNamespace(seed=104729, repeats=2))
    assert config["schedule_version"] == "standalone-tasks-v3"
    assert config["reaction_revision_version"] == REACTION_REVISION_VERSION


def test_mobility_counts_arrival_separately_from_destination(tmp_path):
    from types import SimpleNamespace
    from validate_schedule_smoke import mobility_metrics
    folder = tmp_path/"worlds/control/movement"
    folder.mkdir(parents=True)
    (folder/"0.json").write_text(json.dumps({"persona": {
        "Ada": {"movement": [1, 1], "description": "walk @ the Ville:home:room"},
        "Bo": {"movement": [2, 2], "description": "work @ the Ville:Hobbs Cafe:cafe"}}}))
    server = SimpleNamespace(step=1, sim_code="control", start_time=datetime(2023, 2, 13),
                             curr_time=datetime(2023, 2, 13, 0, 0, 10),
                             personas={n: SimpleNamespace(a_mem=SimpleNamespace(seq_chat=[]))
                                       for n in ("Ada", "Bo")},
                             maze=SimpleNamespace(access_tile=lambda xy: {
                                 "sector": "Hobbs Cafe" if xy == (1, 1) else "home"}))
    metrics = mobility_metrics(tmp_path, 1, server)
    assert metrics["actual_public_visitors"] == metrics["public_destination_agents"] == 1
    assert metrics["destination_without_arrival"] == ["Bo"]
    assert metrics["public_sectors_reached"] == ["Hobbs Cafe"]
