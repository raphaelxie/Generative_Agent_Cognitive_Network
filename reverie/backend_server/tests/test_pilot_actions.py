"""Action-path contracts (v4): agents must be able to leave home, and a model
formatting problem must never be silently converted into "stay home"/"yes"."""
from datetime import datetime
import json
from pathlib import Path
import re
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot_actions import (choose_sector, choose_arena, choose_object, generate_triple,
                           decide_yes_no, decide_option, match_option, parse_choice,
                           parse_triple, parse_yes_no, parse_option_number)
from pilot_schedule import PilotScheduleError
from test_pilot_engine import engine  # noqa: F401  (also pins NCN_PILOT=1)
from test_pilot_tasks import setup_persona

SECTORS = ["Sam Kim's house", "The Rose and Crown Pub", "Hobbs Cafe", "Johnson Park"]
SECTOR_PROMPT = ("Task -- choose an appropriate area.\nArea options: {Sam Kim's house, Hobbs Cafe}. "
                 "For serving coffee, Sam Kim should go to the following area: {")


# ---- parsers --------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    '{"area": "Hobbs Cafe"}',
    '```json\n{"area": "Hobbs Cafe"}\n```',
    '{"area": "hobbs cafe"}',
    '{"area": "{Hobbs Cafe}"}',
    '{"area": "  Hobbs   Cafe. "}',
    '{"area": "\\"Hobbs Cafe\\""}',
])
def test_choice_accepts_realistic_luna_shapes(raw):
    assert parse_choice(raw, SECTORS, "area") == "Hobbs Cafe"


@pytest.mark.parametrize("raw", [
    'Hobbs Cafe',                              # not JSON
    '{"area": "The Library"}',                 # not an option
    '{"area": ""}',
    '{"area": ["Hobbs Cafe"]}',
    '{"area": "Hobbs Cafe", "why": "coffee"}',  # extra field
    '{"place": "Hobbs Cafe"}',                 # wrong field
    '["Hobbs Cafe"]',
])
def test_choice_rejects_everything_else(raw):
    with pytest.raises(ValueError):
        parse_choice(raw, SECTORS, "area")


def test_match_option_is_exact_not_substring():
    with pytest.raises(ValueError):
        match_option("Hobbs", SECTORS)
    with pytest.raises(ValueError):
        match_option("Hobbs Cafe and Johnson Park", SECTORS)
    assert match_option("johnson park", SECTORS) == "Johnson Park"


def test_triple_parser():
    assert parse_triple('{"predicate": "paint", "object": "canvas"}') == ("paint", "canvas")
    for raw in ['{"predicate": "paint"}', '{"predicate": "", "object": "x"}',
                '{"predicate": "a\\nb", "object": "x"}', '{"predicate": "..", "object": "x"}',
                '{"predicate": "paint", "object": "x", "subject": "y"}', "(a, b, c)"]:
        with pytest.raises(ValueError):
            parse_triple(raw)


def test_yes_no_and_option_parsers():
    assert parse_yes_no('{"reasoning": "They are friends.", "answer": "Yes"}') == "yes"
    assert parse_yes_no('{"answer": "no"}') == "no"
    for raw in ['{"answer": "maybe"}', '{"answer": true}', 'yes', '{"reasoning": "x"}',
                '{"answer": "yes", "extra": 1}']:
        with pytest.raises(ValueError):
            parse_yes_no(raw)
    assert parse_option_number('{"reasoning": "wait", "option": 1}', {1, 2}) == "1"
    assert parse_option_number('{"option": "2"}', {1, 2}) == "2"
    for raw in ['{"option": 3}', '{"option": 1.5}', '{"option": true}', '{"option": "one"}']:
        with pytest.raises(ValueError):
            parse_option_number(raw, {1, 2})


# ---- contract behaviour with a fake chat ---------------------------------

def fake_chat(monkeypatch, replies):
    from persona.prompt_template import gpt_structure
    prompts = []
    queue = list(replies)
    def chat(prompt, **kwargs):
        prompts.append(prompt)
        return queue.pop(0) if len(queue) > 1 else queue[0]
    monkeypatch.setattr(gpt_structure, "_chat", chat)
    return prompts


def test_sector_prompt_keeps_template_and_lists_exact_options(monkeypatch):
    prompts = fake_chat(monkeypatch, ['{"area": "Hobbs Cafe"}'])
    assert choose_sector(SECTOR_PROMPT, SECTORS) == "Hobbs Cafe"
    assert "Task -- choose an appropriate area." in prompts[0]
    assert not prompts[0].rstrip().endswith("{")      # completion stub removed
    assert json.dumps(SECTORS) in prompts[0]


def test_bad_first_reply_is_repaired_with_feedback(monkeypatch):
    prompts = fake_chat(monkeypatch, ["Hobbs Cafe", '{"area": "Hobbs Cafe"}'])
    assert choose_sector(SECTOR_PROMPT, SECTORS) == "Hobbs Cafe"
    assert len(prompts) == 2
    assert "failed validation" in prompts[1] and "Hobbs Cafe" in prompts[1]


def test_exhausted_attempts_raise_instead_of_sending_agent_home(monkeypatch):
    prompts = fake_chat(monkeypatch, ["Hobbs Cafe"])   # plain text, never valid JSON
    with pytest.raises(PilotScheduleError, match="after 3 attempts"):
        choose_sector(SECTOR_PROMPT, SECTORS)
    assert len(prompts) == 3
    prompts.clear()
    fake_chat(monkeypatch, ['{"answer": "perhaps"}'])
    with pytest.raises(PilotScheduleError, match="after 3 attempts"):
        decide_yes_no("Question: Would A talk to B?\n\nReasoning: Let's think step by step.")


def test_single_option_needs_no_model_call(monkeypatch):
    prompts = fake_chat(monkeypatch, ["unused"])
    assert choose_arena("Answer: {", ["cafe"]) == "cafe"
    assert prompts == []


def test_object_choice_and_triples(monkeypatch):
    fake_chat(monkeypatch, ['{"object": "easel"}'])
    assert choose_object("Current activity: painting\nPick ONE:", ["easel", "closet"]) == "easel"
    fake_chat(monkeypatch, ['{"predicate": "paint", "object": "canvas"}'])
    assert generate_triple("Input: Ann is painting.\nOutput: (Ann,") == ("paint", "canvas")


def test_decide_prompts_replace_completion_stub(monkeypatch):
    prompts = fake_chat(monkeypatch, ['{"reasoning": "friends", "answer": "no"}'])
    talk = "Question: Would Ann initiate a conversation with Bo?\n\nReasoning: Let's think step by step."
    assert decide_yes_no(talk) == "no"
    assert not prompts[0].rstrip().startswith("Reasoning") and "Let's think step by step." not in prompts[0]
    fake_chat(monkeypatch, ['{"reasoning": "wait", "option": 1}'])
    assert decide_option("Option 1: Wait\nOption 2: Go\nReasoning: ") == "1"


# ---- engine level: real maze, spatial memory and prompt templates --------

def luna_like_chat(prefer, calls):
    """Answers every action-path prompt as JSON, preferring ``prefer`` sectors."""
    def chat(prompt, **kwargs):
        calls.append(prompt)
        if '{"area"' in prompt:
            options = json.loads(re.search(r"verbatim: (\[.*?\])\.", prompt, re.S).group(1))
            pick = next((o for o in options if o in prefer), options[0])
            return json.dumps({"area": pick})
        if '{"object"' in prompt:
            options = json.loads(re.search(r"verbatim: (\[.*?\])\.", prompt, re.S).group(1))
            return json.dumps({"object": options[0]})
        if '"predicate"' in prompt:
            return json.dumps({"predicate": "serve", "object": "coffee"})
        return json.dumps({"output": "ok"})        # emoji / object-state prompts
    return chat


def determine(engine, monkeypatch, name, task, prefer, calls):
    from persona.cognitive_modules import plan
    from persona.prompt_template import gpt_structure, run_gpt_prompt

    def legacy_forbidden(*args, **kwargs):
        raise AssertionError("legacy completion-style call used under NCN_PILOT")
    monkeypatch.setattr(run_gpt_prompt, "safe_generate_response", legacy_forbidden)
    monkeypatch.setattr(gpt_structure, "_chat", luna_like_chat(prefer, calls))
    monkeypatch.setattr(plan, "generate_task_decomp", lambda p, t, d: [[f"{t} part one", d // 2], [f"{t} part two", d - d // 2]])
    server = engine[0]
    p = setup_persona(engine, name, datetime(2023, 2, 13, 8),
                      [["sleeping", 480], [task, 120], ["resting", 840]])
    plan._determine_action(p, server.maze)
    return p


def test_agent_goes_to_public_place_when_the_task_needs_it(engine, monkeypatch):
    calls = []
    p = determine(engine, monkeypatch, "Isabella Rodriguez",
                  "work the morning shift at Hobbs Cafe", {"Hobbs Cafe"}, calls)
    assert p.scratch.act_address.startswith("the Ville:Hobbs Cafe:cafe:")
    assert p.scratch.act_event[0] == p.name and p.scratch.act_event[1:] == ("serve", "coffee")
    assert any("Hobbs Cafe" in c and '{"area"' in c for c in calls)


def test_agent_can_choose_a_shared_room_outside_own_bedroom(engine, monkeypatch):
    calls = []
    p = determine(engine, monkeypatch, "Latoya Williams", "make lunch", {"kitchen", "common room"},
                  calls)
    parts = p.scratch.act_address.split(":")
    assert parts[1] == "artist's co-living space" and parts[2] in ("kitchen", "common room")


def test_model_that_never_answers_valid_json_stops_the_run(engine, monkeypatch):
    from persona.prompt_template import gpt_structure
    from persona.cognitive_modules import plan
    monkeypatch.setattr(gpt_structure, "_chat", lambda prompt, **kw: "Hobbs Cafe")
    monkeypatch.setattr(plan, "generate_task_decomp", lambda p, t, d: [["a part", d // 2], ["b part", d - d // 2]])
    p = setup_persona(engine, "Isabella Rodriguez", datetime(2023, 2, 13, 8),
                      [["sleeping", 480], ["open the cafe", 120], ["resting", 840]])
    with pytest.raises(PilotScheduleError):
        plan._determine_action(p, engine[0].maze)


# ---- preflight guard ------------------------------------------------------

def test_location_preflight_passes_with_a_conforming_model_and_fills_every_template_slot(monkeypatch):
    import run_midnight_pilot as runner
    monkeypatch.chdir(Path(__file__).resolve().parents[1])  # the runner chdirs here
    seen = []
    def chat(prompt, **kwargs):
        seen.append(prompt)
        assert "!<INPUT" not in prompt, "unfilled template slot"
        if '{"area"' in prompt:
            options = json.loads(re.search(r"verbatim: (\[.*?\])\.", prompt, re.S).group(1))
            return json.dumps({"area": "Hobbs Cafe" if "Hobbs Cafe" in options else options[0]})
        if '"predicate"' in prompt:
            return '{"predicate": "serve", "object": "coffee"}'
        return '{"reasoning": "They know each other.", "answer": "yes"}'
    from persona.prompt_template import gpt_structure
    monkeypatch.setattr(gpt_structure, "_chat", chat)
    result = runner.location_preflight()
    assert result["sector_for_cafe_shift"] == "Hobbs Cafe"
    assert result["arena_for_dinner"] in ("kitchen", "common room", "garden")
    assert result["decide_to_talk_parsed"] == "yes" and len(seen) == 4


def test_location_preflight_fails_when_the_cafe_shift_resolves_elsewhere(monkeypatch):
    import run_midnight_pilot as runner
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    from persona.prompt_template import gpt_structure
    monkeypatch.setattr(gpt_structure, "_chat",
                        lambda prompt, **kw: '{"area": "Isabella Rodriguez\'s apartment"}')
    with pytest.raises(ValueError, match="not Hobbs Cafe"):
        runner.location_preflight()
