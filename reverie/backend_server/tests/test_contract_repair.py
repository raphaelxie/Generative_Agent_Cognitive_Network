"""Regressions from the six real single-part failures, with no API access."""
import json
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot_tasks import (request_contract, repair_prompt, parse_task_parts,
                         parse_revision_priority, allocate_revision_minutes,
                         REVISION_REPAIR_INSTRUCTIONS, MAX_REPAIR_RESPONSE, MAX_REPAIR_ERROR)
from pilot_schedule import PilotScheduleError


def fixture():
    path = Path(__file__).resolve().parent/"fixtures/failed_decompositions.json"
    return json.loads(path.read_text())


def valid_parts():
    return json.dumps({"parts": [
        {"activity": "Review the music course assignment", "minutes": 20},
        {"activity": "Complete the assigned composition exercise", "minutes": 40}]})


@pytest.mark.parametrize("index", range(6))
def test_real_invalid_response_is_corrected_only_after_feedback(monkeypatch, index):
    from persona.prompt_template import gpt_structure
    record = fixture()[index]
    prompts, logs = [], []
    def chat(prompt, **kwargs):
        prompts.append(prompt)
        if prompt == record["prompt"]:
            return record["raw"]
        assert "Expected 2 to 48 subtasks" in prompt
        assert record["raw"] == json.loads(prompt.rsplit("\n", 1)[1])["rejected_response"]
        return valid_parts()
    monkeypatch.setattr(gpt_structure, "_chat", chat)
    monkeypatch.setattr("pilot_tasks.record_contract", lambda *args: logs.append(args))
    before = random.getstate()
    result = request_contract(record["prompt"], lambda x: parse_task_parts(x, 60), "Task decomposition")
    assert result == [["Review the music course assignment", 20],
                      ["Complete the assigned composition exercise", 40]]
    assert len(prompts) == 2 and prompts[0] == record["prompt"]
    assert logs[0][2] == prompts[0] and logs[1][2] == prompts[1]
    assert random.getstate() == before


def test_valid_first_attempt_is_unchanged(monkeypatch):
    from persona.prompt_template import gpt_structure
    calls = []
    def chat(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return valid_parts()
    monkeypatch.setattr(gpt_structure, "_chat", chat)
    result = request_contract("original request", lambda x: parse_task_parts(x, 60), "Task decomposition", 1234)
    assert sum(d for _, d in result) == 60
    assert calls == [("original request", {"max_tokens": 1234})]


def test_exhaustion_stays_bounded_and_never_accepts_one_part(monkeypatch):
    from persona.prompt_template import gpt_structure
    record = fixture()[-1]
    calls = []
    def chat(prompt, **kwargs):
        calls.append(prompt)
        return record["raw"]
    monkeypatch.setattr(gpt_structure, "_chat", chat)
    with pytest.raises(PilotScheduleError, match="after 3 attempts"):
        request_contract(record["prompt"], lambda x: parse_task_parts(x, 60), "Task decomposition")
    assert len(calls) == 3
    assert calls[1] == calls[2] and calls[0] != calls[1]


def test_retry_uses_latest_error_without_accumulating_responses(monkeypatch):
    from persona.prompt_template import gpt_structure
    record = fixture()[-1]
    wrong_sum = valid_parts().replace('"minutes": 40', '"minutes": 39')
    responses = iter([record["raw"], wrong_sum, valid_parts()])
    calls = []
    def chat(prompt, **kwargs):
        calls.append(prompt)
        return next(responses)
    monkeypatch.setattr(gpt_structure, "_chat", chat)
    request_contract(record["prompt"], lambda x: parse_task_parts(x, 60), "Task decomposition")
    feedback = json.loads(calls[2].rsplit("\n", 1)[1])
    assert feedback["rejected_response"] == wrong_sum
    assert feedback["validation_error"] == "Subtask durations must sum exactly to the parent duration"
    assert "Expected 2 to 48 subtasks" not in calls[2]


def test_feedback_is_size_bounded_even_for_pathological_output():
    text = repair_prompt("base", "x"*100000, "y"*10000)
    feedback = json.loads(text.rsplit("\n", 1)[1])
    assert len(feedback["rejected_response"]) == MAX_REPAIR_RESPONSE
    assert len(feedback["validation_error"]) == MAX_REPAIR_ERROR
    assert feedback["response_truncated"] is True
    assert len(text) < 3500


def test_transport_error_does_not_enter_format_retry(monkeypatch):
    from persona.prompt_template import gpt_structure
    from pilot_api import PilotAPIError
    calls = []
    def chat(*args, **kwargs):
        calls.append(args)
        raise PilotAPIError("transport failure")
    monkeypatch.setattr(gpt_structure, "_chat", chat)
    with pytest.raises(PilotAPIError):
        request_contract("original", lambda x: parse_task_parts(x, 60), "Task decomposition")
    assert len(calls) == 1


def test_suffix_retry_preserves_order_and_budget(monkeypatch):
    from persona.prompt_template import gpt_structure
    source = [{"activity": "Study notes", "minutes": 30, "terminal": True, "origin": "a"},
              {"activity": "Write an essay", "minutes": 30, "terminal": True, "origin": "b"}]
    # Artificial invalid ranking; the historical minute-error replies were not captured.
    responses = iter(['{"priority":[0,0]}', '{"priority":[1,0]}'])
    calls = []
    def chat(prompt, **kwargs):
        calls.append(prompt)
        return next(responses)
    monkeypatch.setattr(gpt_structure, "_chat", chat)
    priority = request_contract("original", lambda x: parse_revision_priority(x, len(source)),
                                "Schedule revision", repair_instructions=REVISION_REPAIR_INSTRUCTIONS)
    allocated = allocate_revision_minutes(source, priority, 55)
    result = [{**r, "minutes": allocated[i]} for i, r in enumerate(source) if allocated[i]]
    assert [r["origin"] for r in result] == ["a", "b"]
    assert [r["minutes"] for r in result] == [25, 30]
    assert "missing=[1]" in calls[1] and "repeated=[0]" in calls[1]
    assert "exact total minutes" not in calls[1]
