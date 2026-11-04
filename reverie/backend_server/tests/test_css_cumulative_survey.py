"""The cumulative CSS questionnaire asks about connections, with no time window; no paid API calls."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import css_cumulative_survey as css
import measure_css_survey as measurement
from pilot_protocol import graph_from_chats
from test_pilot_engine import engine  # noqa: F401

WINDOW_WORDS = ("120", "minute", "interval", "recent", "STARTED", "preceding", "two hours")


def all_prompts(roster):
    context = css.survey_context(roster[0], {"name": roster[0]}, "(No specific memories found.)", roster)
    pairs = [[roster[0], roster[1]]]
    return [css.pair_prompt(context, pairs)] + [context + q for q in css.QUESTIONS.values()]


def test_questions_do_not_ask_about_a_recent_window():
    for prompt in all_prompts(["Ada", "Bo", "Cy"]):
        for word in WINDOW_WORDS:
            assert word not in prompt
    prompts = all_prompts(["Ada", "Bo"])
    assert "have a connection with each other" in prompts[0]
    assert '"tie"' in prompts[0] and '"uncertain"' in prompts[0]
    assert "give your probability" not in prompts[0]
    assert not any("conversation" in prompt for prompt in prompts)


def test_truth_counts_every_conversation_since_the_start():
    def chat(a, b, hour):
        return SimpleNamespace(subject=a, object=b, created=datetime(2023, 2, 13, hour))
    personas = {"Ada": SimpleNamespace(a_mem=SimpleNamespace(seq_chat=[chat("Ada", "Bo", 1)])),
                "Bo": SimpleNamespace(a_mem=SimpleNamespace(seq_chat=[chat("Bo", "Cy", 15)])),
                "Cy": SimpleNamespace(a_mem=SimpleNamespace(seq_chat=[]))}
    server = SimpleNamespace(personas=personas, start_time=datetime(2023, 2, 13),
                             curr_time=datetime(2023, 2, 13, 16))
    cumulative = css.truth_graph(server)
    recent = graph_from_chats(personas, server.start_time, server.curr_time, 120)
    assert sorted(cumulative.edges()) == [("Ada", "Bo"), ("Bo", "Cy")]
    assert sorted(recent.edges()) == [("Bo", "Cy")]


def test_worker_in_fresh_process_records_cumulative_version(engine, tmp_path):
    from pilot_state import save_snapshot, state_digest
    server, protocol, output = engine
    path = output / "initial.pkl.gz"
    save_snapshot(path, server, protocol)
    before, source = state_digest(server), path.read_bytes()
    prompts = tmp_path / "prompts.jsonl"
    code = '''import sys, json
from pilot_state import load_snapshot
roster = sorted(load_snapshot(sys.argv[1])['server'].personas)
from persona.prompt_template import gpt_structure
import perception_survey
def fake_chat(prompt, **kwargs):
    with open(sys.argv[4], 'a') as out:
        out.write(json.dumps(prompt) + '\\n')
    if 'Pairs: ' in prompt:
        count = len(json.loads(prompt.split('Pairs: ')[-1]))
        return json.dumps({'tie': [i % 2 for i in range(count)],
                           'uncertain': [i % 3 == 0 for i in range(count)]})
    if 'Partition everyone' in prompt:
        return json.dumps({'groups': [[name] for name in roster]})
    return json.dumps({'ranking': roster})
gpt_structure._chat = fake_chat
perception_survey._retrieve_with_diagnostics = lambda *a, **k: ('(No specific memories found.)', {'nodes': []})
from css_cumulative_survey import survey_worker
survey_worker(sys.argv[1], sys.argv[2], int(sys.argv[3]))
'''
    destination = output / "css_survey"
    subprocess.run([sys.executable, "-c", code, str(path), str(destination), "1", str(prompts)],
                   cwd=BACKEND, check=True, env=os.environ.copy())
    complete = json.loads((destination / "complete.json").read_text())
    metadata = json.loads((destination / "metadata.json").read_text())
    assert complete["survey_version"] == metadata["survey_version"] == css.SURVEY_VERSION
    assert metadata["recent_minutes"] is None
    assert metadata["window_start"] == server.start_time.isoformat()
    assert len(list(destination.glob("respondent_*.json"))) == 25
    answer = json.loads((destination / "respondent_00.json").read_text())
    assert len(answer["interaction"]) == len(answer["interaction_uncertainty"]) == 300
    assert answer["survey_version"] == css.SURVEY_VERSION and answer["repeat"] == 1
    assert answer["interaction"][0][2] == 0
    assert answer["interaction_uncertainty"][0][2] is True
    assert [r[:2] for r in answer["interaction"]] == [r[:2] for r in answer["interaction_uncertainty"]]
    sent = [json.loads(line) for line in prompts.read_text().splitlines()]
    assert len(sent) == 25 * (6 + 3)
    # Biographies are persona text; check only the instrument wording that follows them.
    instrument = [prompt.split("For this questionnaire", 1)[1] for prompt in sent]
    assert not any(word in text for text in instrument for word in WINDOW_WORDS)
    assert state_digest(server) == before and path.read_bytes() == source
    # Resuming a complete wave makes no calls and preserves all answers.
    original = (destination / "respondent_00.json").read_bytes()
    command = [sys.executable, "-c", code, str(path), str(destination), "1", str(prompts)]
    subprocess.run(command, cwd=BACKEND, check=True, env=os.environ.copy())
    assert len(prompts.read_text().splitlines()) == 25 * 9
    # Simulate interruption before the final respondent was saved.
    (destination / "complete.json").unlink()
    (destination / "respondent_24.json").unlink()
    subprocess.run(command, cwd=BACKEND, check=True, env=os.environ.copy())
    assert len(prompts.read_text().splitlines()) == 26 * 9
    assert (destination / "respondent_00.json").read_bytes() == original
    assert state_digest(server) == before and path.read_bytes() == source


def test_measurement_keeps_source_and_labels_the_new_instrument(tmp_path, monkeypatch):
    source, output = tmp_path / "trajectory", tmp_path / "css_measurements"
    (source / "snapshots").mkdir(parents=True)
    manifest = {"run_name": "trajectory", "config": {"survey_repeats": 2, "seed": 104729,
                "source_digest": "frozen", "survey_version": "recent-network-v1",
                "recent_minutes": 120}, "step": 23040, "checkpoint_status": "no_eligible_checkpoint",
                "snapshots": {"3600": {"path": "snapshots/3600.gz", "labels": ["scheduled"]}}}
    (source / "manifest.json").write_text(json.dumps(manifest))
    (source / "latest.pkl.gz").write_bytes(b"unchanged recovery bytes")
    (source / "snapshots/3600.gz").write_bytes(b"unchanged snapshot bytes")
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}

    def configure(args, config):
        args.seed = config["seed"]
        return {**config, "survey_repeats": args.repeats}

    def measure(out, snapshot, step, repeats, seed, env_file):
        for repeat in range(1, repeats + 1):
            directory = out / "surveys" / f"step_{step:06d}" / f"repeat_{repeat:02d}"
            directory.mkdir(parents=True)
            (directory / "complete.json").write_text("{}")

    def report(run_dir):
        return {"reported": True}

    monkeypatch.setattr(measurement, "environment_configuration", configure)
    monkeypatch.setattr(measurement, "measure_snapshot", measure)
    monkeypatch.setattr(measurement.css_report, "write_report", report)
    args = SimpleNamespace(source_run=source, output=output, repeats=3, env_file=tmp_path / ".env")
    assert measurement.run_measurement(args) == {"reported": True}
    assert all(p.read_bytes() == data for p, data in before.items())
    plan = json.loads((output / "measurement_plan.json").read_text())
    assert plan["survey_version"] == plan["measurement_config"]["survey_version"] == css.SURVEY_VERSION
    assert plan["measurement_config"]["recent_minutes"] is None
    assert plan["source_runtime_config"]["survey_version"] == "recent-network-v1"
    assert plan["scoring_version"] == measurement.css_report.SCORING_VERSION
    assert len(plan["scorer_sha256"]) == 64
    assert json.loads((output / "manifest.json").read_text())["status"] == "saved_snapshots_measured"
    old = {**plan, "survey_version": "cumulative-connection-v1"}
    with pytest.raises(ValueError, match="new measurement output"):
        measurement.validate_plan_extension(old, plan)


@pytest.mark.parametrize("raw", [
    '{"p": [0.5]}', '{"tie": [0.5], "uncertain": [true]}',
    '{"tie": [true], "uncertain": [false]}', '{"tie": [1.0], "uncertain": [true]}',
    '{"tie": [2], "uncertain": [false]}', '{"tie": [1], "uncertain": [1]}',
    '{"tie": [0], "uncertain": ["false"]}', '{"tie": [0], "uncertain": []}',
    '{"tie": [], "uncertain": [true]}', '{"tie": [null], "uncertain": [null]}',
    '[]', 'null', 'invalid JSON',
])
def test_binary_contract_rejects_probabilities_booleans_as_ties_and_misalignment(raw):
    with pytest.raises((ValueError, TypeError, KeyError)):
        css.parse_pair_answer(raw, 1)


def test_binary_contract_preserves_judgment_and_uncertainty_independently():
    raw = '```json\n{"tie": [0, 1, 0, 1], "uncertain": [true, true, false, false]}\n```'
    assert css.parse_pair_answer(raw, 4) == ([0, 1, 0, 1], [True, True, False, False])


def test_retry_preserves_failures_and_never_fills_in_uncertainty(monkeypatch):
    from persona.prompt_template import gpt_structure
    replies = iter(['{"p": [0.5]}', '{"tie": [0], "uncertain": [true]}'])
    monkeypatch.setattr(gpt_structure, "_chat", lambda *a, **k: next(replies))
    answer, failures = css.ask_pairs("test", 1)
    assert answer == ([0], [True]) and len(failures) == 1
    monkeypatch.setattr(gpt_structure, "_chat", lambda *a, **k: '{"p": [0.5]}')
    answer, failures = css.ask_pairs("test", 1)
    assert answer is None and len(failures) == 3


def test_worker_persists_failed_batch_as_two_missing_fields(tmp_path, monkeypatch):
    import perception_survey
    import pilot_api
    roster = ["Ada", "Bo"]
    server = SimpleNamespace(personas={n: SimpleNamespace(scratch=SimpleNamespace(name=n)) for n in roster},
                             step=1, start_time=datetime(2023, 2, 13), curr_time=datetime(2023, 2, 13, 1))
    snapshot = tmp_path / "snapshot.gz"
    snapshot.write_bytes(b"stub loaded by test")
    monkeypatch.setattr(css, "load_snapshot", lambda _: {"server": server, "state_digest": "state",
                        "protocol": {"config": {"model": "test"}}})
    import networkx as nx
    monkeypatch.setattr(css, "truth_graph", lambda _: nx.empty_graph(roster))
    monkeypatch.setattr(perception_survey, "_retrieve_with_diagnostics", lambda *a, **k: ("", {}))
    monkeypatch.setattr(pilot_api, "set_context", lambda *a, **k: None)
    monkeypatch.setattr(css, "ask_pairs", lambda *a: (None, [{"error": "ValueError", "response": "bad"}] * 3))
    monkeypatch.setattr(css, "ask", lambda prompt, kind, names: ([names] if kind == "community" else names, []))
    output = tmp_path / "survey"
    css.survey_worker(snapshot, output, 1)
    answers = css.read_survey_results(output, require_complete=True)
    answer = answers["respondent_00.json"]
    assert answer["interaction"] == answer["interaction_uncertainty"] == [["Ada", "Bo", None]]
    assert len(answer["invalid_attempts"]["interaction_0"]) == 3
    # Old answers are rejected before any metadata is overwritten, even at the same snapshot.
    meta_before = (output / "metadata.json").read_bytes()
    answer["survey_version"] = "cumulative-connection-v1"
    (output / "respondent_00.json").write_text(json.dumps(answer))
    with pytest.raises(ValueError, match="different survey"):
        css.survey_worker(snapshot, output, 1)
    assert (output / "metadata.json").read_bytes() == meta_before
