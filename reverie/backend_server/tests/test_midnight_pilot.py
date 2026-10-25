import copy
from datetime import datetime, timedelta
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("OPENAI_API_KEY", "test-only-not-a-key")
os.environ["NCN_PILOT"] = "1"
os.environ["OPENAI_MODEL"] = "gpt-6-luna"
os.environ["GA_DEBUG"] = "0"

from pilot_protocol import (CHECK_START, CHECK_END, CHECK_INTERVAL, CHECK_STEPS,
    END_STEP, SURVEY_STEPS, CheckpointSelector, graph_from_chats,
    priority_order, labels_at, next_boundary)
from pilot_sleep import bedtime_hour, prewake_sleep
from pilot_state import (save_snapshot, load_snapshot, state_digest, rng_state,
                         restore_rng, materialize_environment)
from pilot_survey import parse_answer
from pilot_report import score_respondent
from pilot_api import chat_arguments, estimated_cost, PilotAPIError


def separated_graph():
    graph = nx.barbell_graph(5, 2)
    graph.add_edges_from([(0, 12), (0, 13)])
    nx.set_edge_attributes(graph, 1, "weight")
    return nx.relabel_nodes(graph, {n: str(n) for n in graph})


def selector_for(graph):
    return CheckpointSelector(priority_order(list(graph), 104729))


def test_exact_timing_and_no_skipped_ticks():
    assert END_STEP*10 == 64*3600
    assert len(CHECK_STEPS) == 25
    assert CHECK_STEPS[0]*10 == 20*3600
    assert CHECK_STEPS[-1]*10 == 32*3600
    step, boundaries = 0, {0}
    while step < END_STEP:
        new = next_boundary(step, CHECK_START + CHECK_INTERVAL)
        assert 0 < new-step <= CHECK_INTERVAL
        step = new
        boundaries.add(step)
    assert set(SURVEY_STEPS) <= boundaries
    assert set(CHECK_STEPS) <= boundaries
    assert labels_at(END_STEP) == ["final"]
    # A selected 22:00 checkpoint has +12? Here +24h coincides with a normal wave.
    assert labels_at(34*360, 10*360) == ["scheduled", "checkpoint_plus_24h"]


def test_separation_requires_two_adjacent_matching_checks_and_locks_once():
    graph = separated_graph()
    selector = selector_for(graph)
    first = selector.observe(CHECK_START, graph)
    assert first["eligible"] and not first["locked_now"]
    second = selector.observe(CHECK_START+CHECK_INTERVAL, graph)
    assert second["locked_now"] and second["hub"] != second["broker"]
    locked = copy.deepcopy(selector.selected)
    selector.observe(CHECK_START+2*CHECK_INTERVAL, nx.complete_graph(list(graph)))
    assert selector.selected == locked
    restored = CheckpointSelector(**json.loads(json.dumps(selector.to_dict())))
    assert restored.selected == locked


def test_overlap_is_not_resolved_by_choosing_another_tied_name():
    graph = nx.cycle_graph(["a", "b", "c", "d"])
    nx.set_edge_attributes(graph, 1, "weight")
    record = selector_for(graph).observe(CHECK_START, graph)
    assert len(record["hub_ties"]) == 4 and len(record["broker_ties"]) == 4
    assert not record["eligible"] and record["hub"] is None


def test_zero_betweenness_and_no_eligible_checkpoint():
    graph = nx.complete_graph(["a", "b", "c"])
    nx.set_edge_attributes(graph, 1, "weight")
    selector = selector_for(graph)
    for step in CHECK_STEPS:
        assert not selector.observe(step, graph)["eligible"]
    assert selector.status(CHECK_END) == "no_eligible_checkpoint"
    assert selector.selected is None
    with pytest.raises(ValueError):
        selector.observe(CHECK_END+CHECK_INTERVAL, graph)


def test_gaps_and_ineligible_checks_reset_stability():
    graph = separated_graph()
    selector = selector_for(graph)
    selector.observe(CHECK_START, graph)
    assert not selector.observe(CHECK_START+2*CHECK_INTERVAL, graph)["locked_now"]
    blank = nx.empty_graph(list(graph))
    selector.observe(CHECK_START+3*CHECK_INTERVAL, blank)
    assert not selector.observe(CHECK_START+4*CHECK_INTERVAL, graph)["locked_now"]


def test_lock_at_last_window_boundary():
    graph = separated_graph()
    selector = selector_for(graph)
    selector.observe(CHECK_END-CHECK_INTERVAL, graph)
    assert selector.observe(CHECK_END, graph)["locked_now"]


def test_selection_does_not_consume_simulation_randomness():
    random.seed(15)
    before = random.getstate()
    graph = separated_graph()
    selector_for(graph).observe(CHECK_START, graph)
    assert before == random.getstate()


def chat(a, b, time):
    return SimpleNamespace(subject=a, object=b, created=time)


def test_truth_deduplicates_and_excludes_old_and_future_conversations():
    start = datetime(2023, 2, 13)
    now = start + timedelta(hours=4)
    nodes = [chat("a", "b", start-timedelta(seconds=1)),
             chat("a", "b", start+timedelta(hours=1)),
             chat("a", "b", start+timedelta(hours=2)),
             chat("a", "b", now), chat("a", "b", now+timedelta(seconds=10))]
    personas = {name: SimpleNamespace(a_mem=SimpleNamespace(seq_chat=nodes)) for name in ["a", "b", "c"]}
    cumulative = graph_from_chats(personas, start, now)
    recent = graph_from_chats(personas, start, now, 120)
    assert cumulative["a"]["b"]["weight"] == 2
    assert recent["a"]["b"]["weight"] == 1
    assert set(recent) == {"a", "b", "c"}


@pytest.mark.parametrize("lifestyle,wake,expected", [
    ("Adam Smith goes to bed around 8pm, awakes up around 4am.", 4, range(4)),
    ("Hailey Johnson goes to bed around 2am, awakes up around 10am.", 10, range(2, 10)),
    ("Ryan Park goes to bed around 1am, awakes up around 9am.", 9, range(1, 9)),
    ("Rajiv Patel goes to bed around midnight, awakes up around 9am.", 9, range(9)),
])
def test_persona_specific_sleep_prefix(lifestyle, wake, expected):
    assert [h for h in range(24) if prewake_sleep(h, wake, lifestyle)] == list(expected)


def test_generated_schedule_preserves_late_night_activity(monkeypatch, tmp_path):
    monkeypatch.chdir(BACKEND)
    from persona.cognitive_modules import plan
    persona = SimpleNamespace(scratch=SimpleNamespace(
        lifestyle="Hailey Johnson goes to bed around 2am, awakes up around 10am."))
    monkeypatch.setattr(plan, "run_gpt_prompt_generate_hourly_schedule",
                        lambda persona, hour, previous, hours, **kwargs: ["awake at " + hour])
    schedule = plan.generate_hourly_schedule(persona, 10)
    assert schedule[0] == ["awake at 00:00 AM", 60]
    assert schedule[1] == ["awake at 01:00 AM", 60]
    assert schedule[2] == ["sleeping", 480]
    assert sum(duration for _, duration in schedule) == 1440


def test_schedule_retry_preserves_sleep_prefix(monkeypatch):
    monkeypatch.chdir(BACKEND)
    from persona.cognitive_modules import plan
    persona = SimpleNamespace(scratch=SimpleNamespace(
        lifestyle="Adam Smith goes to bed around 8pm, awakes up around 4am."))
    monkeypatch.setattr(plan, "run_gpt_prompt_generate_hourly_schedule", lambda *args, **kwargs: ["awake"])
    schedule = plan.generate_hourly_schedule(persona, 4)
    assert schedule[0] == ["sleeping", 240]
    assert sum(duration for _, duration in schedule) == 1440


def test_action_completion_across_midnight():
    from persona.memory_structures.scratch import Scratch
    scratch = Scratch.__new__(Scratch)
    scratch.act_address = "bed"
    scratch.chatting_with = None
    scratch.act_start_time = datetime(2023, 2, 13, 23, 30)
    scratch.act_duration = 120
    scratch.curr_time = datetime(2023, 2, 14, 1, 29)
    assert not scratch.act_check_finished()
    scratch.curr_time = datetime(2023, 2, 14, 1, 30, 10)
    assert scratch.act_check_finished()
    scratch.curr_time = datetime(2023, 2, 15, 1, 29)
    assert scratch.act_check_finished()


def fake_server():
    return SimpleNamespace(curr_time=datetime(2023, 2, 13), step=0,
        personas={"Ada": SimpleNamespace(isolated=False, scratch={"plan": [1, 2]},
            memory={"last_accessed": datetime(2023, 2, 13), "embedding": [0.2]})},
        maze={"events": {("bed", None, None, None)}},
        game_obj_cleanup={("bed", "is", "used", "used"): (2, 3)})


def advance_replay(server):
    server.personas["Ada"].scratch["plan"].append(random.random() + float(np.random.random()))
    server.curr_time += timedelta(seconds=10)
    server.step += 1


def test_complete_snapshot_restores_rng_objects_and_replayed_progress(tmp_path):
    random.seed(4)
    np.random.seed(5)
    live = fake_server()
    path = tmp_path/"state.pkl.gz"
    digest = save_snapshot(path, live, {"selected": "Ada"})
    assert state_digest(live) == digest
    advance_replay(live)
    expected = state_digest(live)
    restored = load_snapshot(path)
    assert restored["state_digest"] == digest
    assert restored["server"].game_obj_cleanup == fake_server().game_obj_cleanup
    advance_replay(restored["server"])
    assert state_digest(restored["server"]) == expected


def test_disposable_process_cannot_leak_survey_or_rng_into_next_survey(tmp_path):
    live = fake_server()
    path = tmp_path/"state.pkl.gz"
    save_snapshot(path, live, {})
    before = state_digest(live)
    original = path.read_bytes()
    code = """import sys, random
from pilot_state import load_snapshot
state = load_snapshot(sys.argv[1])
agent = state['server'].personas['Ada']
assert 'survey' not in agent.memory
agent.memory['survey'] = 'previous answers'
agent.scratch['plan'].append('survey')
random.random()
"""
    for _ in range(2):
        subprocess.run([sys.executable, "-c", code, str(path)], check=True, cwd=BACKEND)
    assert path.read_bytes() == original
    assert state_digest(live) == before


def test_materialized_environment_uses_next_movement_not_old_position(tmp_path):
    (tmp_path/"movement").mkdir()
    (tmp_path/"movement/0.json").write_text(json.dumps({"persona": {"Ada": {"movement": [4, 5]}}}))
    server = SimpleNamespace(step=1, personas_tile={"Ada": (2, 3)})
    materialize_environment(server, tmp_path)
    assert json.loads((tmp_path/"environment/1.json").read_text()) == {"Ada": {"x": 4, "y": 5}}


def test_four_instrument_validation_and_missing_initial_scores():
    roster = ["Ada", "Bo"]
    assert parse_answer('{"p": [0.2]}', "interaction", roster, 1) == [0.2]
    for raw in ('{"p": [true]}', '{"p": [2]}', '{"p": [NaN]}'):
        with pytest.raises(ValueError):
            parse_answer(raw, "interaction", roster, 1)
    with pytest.raises(ValueError):
        parse_answer('{"ranking": ["Ada", "Ada"]}', "degree", roster)
    with pytest.raises(ValueError):
        parse_answer('{"groups": [["Ada"], ["Ada"]]}', "community", roster)
    metadata = {"roster": roster, "truth_edges": [], "step": 0, "repeat": 1, "initial": True}
    answer = {"respondent": "Ada", "interaction": [["Ada", "Bo", .2]],
              "degree": roster, "broker": roster, "community": [roster], "invalid_attempts": {}}
    scores = score_respondent(metadata, answer)
    assert all(scores[key] is None for key in ("brier", "degree_tau_b", "broker_tau_b", "community_ari"))
    metadata.update(step=3600, initial=False)
    scores = score_respondent(metadata, answer)
    assert scores["brier"] == pytest.approx(.04)
    assert scores["degree_tau_b"] is None and scores["broker_tau_b"] is None


def test_api_model_lock_and_cost_accounting():
    kwargs = chat_arguments("gpt-6-luna", "Hi", 15)
    assert kwargs["reasoning_effort"] == "none"
    assert kwargs["max_completion_tokens"] == 15
    assert "temperature" not in kwargs and "max_tokens" not in kwargs
    with pytest.raises(PilotAPIError):
        chat_arguments("gpt-4o-mini", "Hi")
    record = {"kind": "chat", "usage": {"prompt_tokens": 1000000, "completion_tokens": 1000000,
              "prompt_tokens_details": {"cached_tokens": 100000}}}
    assert estimated_cost(record) == pytest.approx(.591)


def test_64_hour_protocol_end_to_end_without_api(tmp_path):
    from run_midnight_pilot import process_boundary, save_progress
    graph = separated_graph()
    start = datetime(2023, 2, 13)
    chats = [chat(a, b, start+timedelta(hours=1)) for a, b in graph.edges()]
    personas = {name: SimpleNamespace(a_mem=SimpleNamespace(seq_chat=chats),
                isolated=False, scratch=SimpleNamespace(act_description="resting")) for name in graph}
    server = SimpleNamespace(start_time=start, curr_time=start, step=0, personas=personas,
                             personas_tile={name: (1, 2) for name in graph}, sim_code="control")
    protocol = {"config": {"survey_repeats": 3}, "selector": selector_for(graph).to_dict(),
                "network_history": [], "snapshots": {}, "status": "running", "arm": "control", "template": "fixture"}
    while True:
        selected = process_boundary(tmp_path, server, protocol, defer_surveys=True)
        if server.step == END_STEP:
            break
        server.step = next_boundary(server.step, selected)
        server.curr_time = start+timedelta(seconds=10*server.step)
    assert selected == CHECK_START+CHECK_INTERVAL
    snapshots = protocol["snapshots"]
    assert set(map(str, SURVEY_STEPS)) <= set(snapshots)
    for hours in [2, 4, 24]:
        assert str(selected+hours*360) in snapshots
    assert len(protocol["selector"]["records"]) == 25
    assert protocol["network_history"][-1]["sim_time"] == "2023-02-15T16:00:00"
    # Resume and reprocess the same boundary: no duplicated candidate or history.
    old_len = len(protocol["network_history"])
    payload = load_snapshot(tmp_path/"latest.pkl.gz")
    process_boundary(tmp_path, payload["server"], payload["protocol"], defer_surveys=True)
    assert len(payload["protocol"]["network_history"]) == old_len


def test_api_error_cannot_be_hidden_by_a_legacy_bare_except(tmp_path, monkeypatch):
    import pilot_api
    monkeypatch.setenv("NCN_USAGE_LOG", str(tmp_path/"usage.jsonl"))
    monkeypatch.setattr(pilot_api, "_FAILURE", None)
    pilot_api.record_usage("chat", "gpt-6-luna", error=ConnectionError())
    try:
        pilot_api.raise_if_failed()
    except BaseException:
        pass
    with pytest.raises(PilotAPIError):
        pilot_api.raise_if_failed()
