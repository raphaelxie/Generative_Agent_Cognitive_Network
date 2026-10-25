"""Regression: a midnight timetable row must never become an afternoon activity."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot_schedule import (parse_hourly_activity, build_hourly_prompt,
                            generate_hourly_activity, parse_daily_plan, PilotScheduleError)


def persona():
    return SimpleNamespace(scratch=SimpleNamespace(
        get_str_iss=lambda: "Latoya is a painter. She sleeps at 23:00 and wakes at 08:00.",
        get_str_curr_date_str=lambda: "Monday February 13",
        daily_req=["Paint at the studio at 15:00", "Eat dinner at 18:00"]))


def test_actual_failed_midnight_row_is_rejected_for_afternoon():
    raw = "[Monday February 13 -- 00:00 AM] Activity: Latoya is sleeping"
    with pytest.raises(ValueError):
        parse_hourly_activity(raw, "03:00 PM")
    with pytest.raises(ValueError, match="different hour"):
        parse_hourly_activity('{"hour":"00:00 AM","activity":"sleeping"}', "03:00 PM")
    assert parse_hourly_activity('{"hour":"03:00 PM","activity":"painting at the studio"}',
                                 "03:00 PM") == "painting at the studio"


def test_prompt_asks_one_hour_and_retains_identity_daily_plan_and_history():
    prompt = build_hourly_prompt(persona(), "03:00 PM", ["sleeping"], ["00:00 AM"])
    context = json.loads(prompt.split("\n", 1)[1])
    assert context["requested_hour"] == "03:00 PM"
    assert context["previous_hours"] == [{"hour":"00:00 AM", "activity":"sleeping"}]
    assert context["daily_plan"] == persona().scratch.daily_req
    assert "ONE hour" in prompt and "[Fill in]" not in prompt


def test_invalid_hour_output_never_falls_back_to_asleep(monkeypatch):
    from persona.prompt_template import gpt_structure
    calls = []
    def wrong_hour(prompt, **kwargs):
        calls.append(kwargs)
        return '{"hour":"00:00 AM","activity":"sleeping"}'
    monkeypatch.setattr(gpt_structure, "_chat", wrong_hour)
    with pytest.raises(PilotScheduleError):
        generate_hourly_activity(persona(), "03:00 PM", [], [])
    assert len(calls) == 3 and all("stop" not in call for call in calls)


def test_pilot_adapter_uses_validated_activity(monkeypatch):
    from persona.prompt_template import run_gpt_prompt
    from persona.prompt_template import gpt_structure
    monkeypatch.setenv("NCN_PILOT", "1")
    monkeypatch.setattr(gpt_structure, "_chat", lambda *args, **kwargs:
                        '{"hour":"03:00 PM","activity":"painting at the studio"}')
    result = run_gpt_prompt.run_gpt_prompt_generate_hourly_schedule(persona(), "03:00 PM", [], [])
    assert result[0] == "painting at the studio"


def test_empty_or_wake_only_daily_plan_is_rejected():
    for raw in ('{"activities":[]}', '{"activities":["wake up at 08:00"]}'):
        with pytest.raises(ValueError):
            parse_daily_plan(raw)
    assert len(parse_daily_plan('{"activities":["wake at 08:00","paint at 15:00","bed at 23:00"]}')) == 3
