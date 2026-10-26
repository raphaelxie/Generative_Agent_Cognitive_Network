"""Offline regressions for expansion, reaction intervals and day rollover."""
import copy
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot_schedule import PilotScheduleError, parse_daily_plan, parse_hourly_activity, parse_wake_hour
from pilot_tasks import (initialize_day, prepare_actions, reschedule_reaction, state_for,
                         slice_rows, projection, parse_task_parts, parse_revision_priority,
                         generate_task_parts, current_action_end)
from test_pilot_engine import engine


def setup_persona(engine, name="Latoya Williams", now=None, schedule=None):
    server, _, _ = engine
    p = server.personas[name]
    p.scratch.curr_time = now or datetime(2023, 2, 13, 8)
    p.scratch.curr_tile = server.personas_tile[name]
    schedule = schedule or [["sleeping", 480], ["painting", 120], ["reading", 120], ["resting", 720]]
    p.scratch.f_daily_schedule_hourly_org = copy.deepcopy(schedule)
    initialize_day(p.scratch, schedule)
    return p


def suffix_response(prompt, **kwargs):
    context = json.loads(prompt.split("\n", 1)[1])
    return json.dumps({"priority": [row["source"] for row in context["source"]]})


def mock_location(monkeypatch):
    from persona.cognitive_modules import plan
    monkeypatch.setattr(plan,"generate_action_sector",lambda *a:"home")
    monkeypatch.setattr(plan,"generate_action_arena",lambda *a:"room")
    monkeypatch.setattr(plan,"generate_action_game_object",lambda *a:"desk")
    monkeypatch.setattr(plan,"generate_action_pronunciatio",lambda *a:"x")
    monkeypatch.setattr(plan,"generate_action_event_triple",lambda a,p:(p.name,"does",a))
    monkeypatch.setattr(plan,"generate_act_obj_desc",lambda *a:"being used")
    monkeypatch.setattr(plan,"generate_act_obj_event_triple",lambda *a:("desk","is","used"))


def test_real_adapter_never_adds_parent_wrapper(engine,monkeypatch):
    p=setup_persona(engine)
    from persona.prompt_template import gpt_structure, run_gpt_prompt
    monkeypatch.setattr(gpt_structure,"_chat",lambda *a,**k:json.dumps({"parts":[
        {"activity":"mix pigments","minutes":60},{"activity":"paint the canvas","minutes":60}]}))
    result=run_gpt_prompt.run_gpt_prompt_task_decomp(p,"painting",120)[0]
    assert result==[["mix pigments",60],["paint the canvas",60]]


def test_repeated_actual_action_selection_does_not_reexpand_60_minute_leaves(engine,monkeypatch):
    p=setup_persona(engine)
    from persona.cognitive_modules import plan
    mock_location(monkeypatch)
    calls=[]
    def parts(persona, task, minutes):
        calls.append(task)
        return [["mix pigments",60],["paint the canvas",60]]
    monkeypatch.setattr(plan,"generate_task_decomp",parts)
    maze=SimpleNamespace(access_tile=lambda xy:{"world":"ville"})
    for _ in range(20):
        plan._determine_action(p,maze)
        p.scratch.curr_time+=timedelta(seconds=10)
    assert calls==["painting", "reading"]
    assert p.scratch.f_daily_schedule[1:3]==[["mix pigments",60],["paint the canvas",60]]
    assert p.scratch.pilot_action_end_time==datetime(2023,2,13,9)
    p.scratch.curr_time=datetime(2023,2,13,9)
    assert p.scratch.act_check_finished()


def test_repeated_replans_keep_leaf_flags_text_order_and_elapsed_prefix(engine,monkeypatch):
    p=setup_persona(engine)
    from persona.prompt_template import gpt_structure
    monkeypatch.setattr(gpt_structure,"_chat",suffix_response)
    calls=[]
    def parts(persona,task,minutes):
        calls.append(task)
        return [["mix pigments",60],["paint the canvas",60]]
    prepare_actions(p,parts)
    for minute in range(5,55,5):
        p.scratch.curr_time=datetime(2023,2,13,8,minute)
        old=copy.deepcopy(state_for(p.scratch)["rows"])
        reschedule_reaction(p,"talk with a neighbor",1)
        rows=state_for(p.scratch)["rows"]
        assert slice_rows(rows,0,480+minute)==slice_rows(old,0,480+minute)
        assert {r["activity"] for r in rows} <= {"sleeping","mix pigments","paint the canvas","reading","resting","talk with a neighbor","read a chapter","take notes"}
        # Don't let advancing the horizon bring unrelated coarse rows into this assertion.
        prepare_actions(p,lambda *a: [["read a chapter",60],["take notes",60]])
        assert sum(d for a,d in p.scratch.f_daily_schedule)==1440
        assert max(len(a) for a,d in p.scratch.f_daily_schedule)<30
    assert calls==["painting"]


@pytest.mark.parametrize("raw",[
    "1. **08:00–09:00** Paint", '```json\n{"parts": []}\n```',
    '{"parts":[{"activity":"mix pigments","minutes":60}]}',
    '{"parts":[{"activity":"mix pigments","minutes":true},{"activity":"paint","minutes":59}]}',
    '{"parts":[{"activity":"mix pigments","minutes":0},{"activity":"paint","minutes":60}]}',
    '{"parts":[{"activity":"mix pigments","minutes":30},{"activity":"paint","minutes":29}]}',
    '{"parts":[{"activity":"...","minutes":30},{"activity":"paint","minutes":30}]}',
])
def test_malformed_decomposition_is_rejected(raw):
    with pytest.raises((ValueError,TypeError)):
        parse_task_parts(raw,60)


def test_parse_retry_exhaustion_aborts_without_schedule_mutation(engine,monkeypatch):
    p=setup_persona(engine)
    before=copy.deepcopy(vars(p.scratch))
    from persona.prompt_template import gpt_structure
    calls=[]
    def bad(*a,**k): calls.append(1); return "8:00–9:00 paint"
    monkeypatch.setattr(gpt_structure,"_chat",bad)
    with pytest.raises(PilotScheduleError,match="after 3 attempts"):
        prepare_actions(p,lambda p,t,d:generate_task_parts(p,t,d)[0])
    assert len(calls)==3 and vars(p.scratch)==before


def test_valid_retry_preserves_model_order_and_exact_minutes(engine,monkeypatch):
    p=setup_persona(engine)
    from persona.prompt_template import gpt_structure
    responses=iter(["bad",json.dumps({"parts":[{"activity":"mix pigments","minutes":5},
                                                           {"activity":"paint canvas","minutes":55}]})])
    monkeypatch.setattr(gpt_structure,"_chat",lambda *a,**k:next(responses))
    assert generate_task_parts(p,"painting",60)[0]==[["mix pigments",5],["paint canvas",55]]


def test_replanning_rejects_minutes_and_activity_text_instead_of_priority():
    for value in [{"parts": [{"source": 0, "minutes": 20}]},
                  {"priority": [0, 1], "activity": "sleeping"},
                  {"priority": [0, 1], "minutes": [20, 30]}]:
        with pytest.raises(ValueError):
            parse_revision_priority(json.dumps(value), 2)


def test_failed_revision_does_not_change_past_or_any_state(engine,monkeypatch):
    p=setup_persona(engine,schedule=[["sleeping",480],["painting",60],["reading",60],["resting",840]])
    before=copy.deepcopy(vars(p.scratch))
    from persona.prompt_template import gpt_structure
    calls=[]
    def bad(*a,**k):calls.append(1);return '{"parts":[]}'
    monkeypatch.setattr(gpt_structure,"_chat",bad)
    with pytest.raises(PilotScheduleError):reschedule_reaction(p,"talk with a neighbor",5)
    assert vars(p.scratch)==before and len(calls)==3


@pytest.mark.parametrize("now,duration,day_minutes,next_minutes",[
    (datetime(2023,2,13,23,55),10,5,5),
    (datetime(2023,2,13,23,59,50),5,0,5),
    (datetime(2023,2,13,23,55),5,5,0)])
def test_reaction_at_midnight_preserves_day_total_and_carries_only_actual_overlap(engine,monkeypatch,now,duration,day_minutes,next_minutes):
    p=setup_persona(engine,now=now,schedule=[["reading",60] for _ in range(24)])
    from persona.prompt_template import gpt_structure
    monkeypatch.setattr(gpt_structure,"_chat",lambda *a,**k:pytest.fail("No suffix to replan at midnight"))
    reschedule_reaction(p,"talk with a neighbor",duration)
    assert sum(d for a,d in p.scratch.f_daily_schedule)==1440
    assert sum(d for a,d in p.scratch.f_daily_schedule if a=="talk with a neighbor")==day_minutes
    p.scratch.curr_time=datetime(2023,2,14)
    initialize_day(p.scratch,[["sleeping",480],["painting",960]])
    assert sum(d for a,d in p.scratch.f_daily_schedule)==1440
    assert sum(d for a,d in p.scratch.f_daily_schedule if a=="talk with a neighbor")==next_minutes
    assert all(d>0 for a,d in p.scratch.f_daily_schedule)


def test_replan_keeps_suffix_after_window_and_handles_final_hour(engine,monkeypatch):
    p=setup_persona(engine,now=datetime(2023,2,13,23,10),schedule=[["reading",60] for _ in range(24)])
    from persona.prompt_template import gpt_structure
    monkeypatch.setattr(gpt_structure,"_chat",suffix_response)
    reschedule_reaction(p,"talk with a neighbor",5)
    assert p.scratch.f_daily_schedule[-2:]==[["talk with a neighbor",5],["reading",45]]
    assert current_action_end(p.scratch)==datetime(2023,2,13,23,15)


@pytest.mark.parametrize("name,wakes,sleep_ranges",[
    ("Hailey Johnson",[10,11],[range(2,10),range(2,11)]),
    ("Adam Smith",[4,5],[range(4),range(5)])])
def test_day1_day2_refresh_daily_plan_and_generated_wake_without_losing_sleep(engine,monkeypatch,name,wakes,sleep_ranges):
    server,_,_=engine
    p=server.personas[name]
    from persona.cognitive_modules import plan
    wake_values=iter(wakes);calls=[]
    monkeypatch.setattr(plan,"generate_wake_up_hour",lambda p:next(wake_values))
    def daily(p,wake):
        calls.append((p.scratch.curr_time.date(),wake))
        return [f"wake at {wake}:00",f"work on {p.scratch.curr_time.day}","have dinner at 18:00"]
    monkeypatch.setattr(plan,"generate_first_daily_plan",daily)
    monkeypatch.setattr(plan,"revise_identity",lambda p:None)
    monkeypatch.setattr(plan,"get_embedding",lambda *a:[0.1,0.2])
    monkeypatch.setattr(plan,"run_gpt_prompt_generate_hourly_schedule",lambda p,h,*a,**k:["awake at "+h])
    for day,new_day in [(13,"First day"),(14,"New day")]:
        p.scratch.curr_time=datetime(2023,2,day)
        plan._long_term_planning(p,new_day)
        rows=state_for(p.scratch)["rows"]
        minute_text=[r["activity"] for r in rows for _ in range(r["minutes"])]
        assert [h for h in range(24) if minute_text[h*60]=="sleeping"]==list(sleep_ranges[day-13])
        assert p.scratch.daily_req[1]==f"work on {day}"
        assert len(minute_text)==1440
    assert len(calls)==2 and calls[0][1]!=calls[1][1]


def test_full_snapshot_preserves_terminal_state_and_no_reexpansion_after_restore(engine):
    server,protocol,output=engine
    p=setup_persona(engine)
    prepare_actions(p,lambda *a:[["mix pigments",60],["paint the canvas",60]])
    from pilot_state import save_snapshot,load_snapshot,state_digest
    before=state_digest(server)
    save_snapshot(output/"v3.pkl.gz",server,protocol)
    restored=load_snapshot(output/"v3.pkl.gz")["server"]
    assert state_digest(restored)==before
    p=restored.personas[p.name]
    prepare_actions(p,lambda *a:pytest.fail("Restored leaves must remain terminal"))
    assert state_digest(restored)==before


def test_mismatched_or_absent_provenance_fails_instead_of_cleaning_old_state(engine):
    p=setup_persona(engine)
    p.scratch.f_daily_schedule[1][0]="painting (painting)"
    with pytest.raises(PilotScheduleError,match="diverged"):state_for(p.scratch)
    del p.scratch.pilot_schedule_state
    with pytest.raises(PilotScheduleError,match="new trajectory"):state_for(p.scratch)


@pytest.mark.parametrize("raw",['{"wake_hour":true}','{"wake_hour":24}','{"wake_hour":-1}','8am'])
def test_wake_contract_rejects_out_of_range_and_malformed_values(raw):
    with pytest.raises(ValueError):parse_wake_hour(raw)


def test_daily_hourly_contracts_reject_punctuation_only_and_wrong_fields():
    with pytest.raises(ValueError):parse_hourly_activity('{"hour":"03:00 PM","activity":"..."}',"03:00 PM")
    with pytest.raises(ValueError):parse_daily_plan('{"activities":["...","...","..."]}')
    with pytest.raises(ValueError):parse_daily_plan('{"activities":["wake at 4","work at 8","bed at 20"],"extra":1}')


def test_new_source_cannot_resume_or_verify_old_evidence(monkeypatch):
    import run_midnight_pilot as runner
    evidence=Path(__file__).resolve().parent/"fixtures/main_10202026"
    old=json.loads((evidence/"manifest.json").read_text())["config"]
    new={**old,"source_digest":runner.source_digest()}
    assert new!=old
    with pytest.raises(ValueError,match="configuration"):
        runner.verify_progress(evidence,new)
    monkeypatch.setattr(runner,"preflight",lambda *a:pytest.fail("No network before rejection"))
    with pytest.raises(ValueError,match="configuration changed"):
        runner.run(SimpleNamespace(),evidence,new)


def test_transport_error_is_not_retried_as_bad_json(engine,monkeypatch):
    p=setup_persona(engine)
    from persona.prompt_template import gpt_structure
    from pilot_api import PilotAPIError
    calls=[]
    def fail(*a,**k):calls.append(1);raise PilotAPIError("injected connection failure")
    monkeypatch.setattr(gpt_structure,"_chat",fail)
    with pytest.raises(PilotAPIError):generate_task_parts(p,"painting",60)
    assert len(calls)==1


def test_literal_parentheses_survive_prompts_and_memory(engine,monkeypatch):
    p=setup_persona(engine)
    from persona.prompt_template import run_gpt_prompt
    from persona.prompt_template import gpt_structure
    seen=[]
    def response(prompt,*args,**kwargs):
        seen.append(prompt)
        return json.dumps({"predicate":"mixes","object":"pigments"})
    monkeypatch.setattr(gpt_structure,"_chat",response)
    description="mix pigments (blue and red) for the canvas"
    triple=run_gpt_prompt.run_gpt_prompt_event_triple(description,p)[0]
    assert triple==(p.name,"mixes","pigments")
    assert description in seen[0]
    node=p.a_mem.add_event(p.scratch.curr_time,None,p.name,"mixes","pigments",description,
                          {"pigments"},1,(description,[0.1,0.2]),[])
    assert node.description==description


def test_scratch_json_roundtrip_retains_provenance_and_absolute_deadline(engine):
    p=setup_persona(engine)
    from persona.memory_structures.scratch import Scratch
    prepare_actions(p,lambda *a:[["mix pigments",60],["paint canvas",60]])
    p.scratch.pilot_action_end_time=datetime(2023,2,13,9)
    path=engine[2]/"scratch.json"
    p.scratch.save(path)
    restored=Scratch(path)
    assert restored.pilot_schedule_state==p.scratch.pilot_schedule_state
    assert restored.pilot_action_end_time==p.scratch.pilot_action_end_time
    state_for(restored)


def test_daily_and_wake_failures_exhaust_three_attempts_without_defaults(engine,monkeypatch):
    p=setup_persona(engine)
    from persona.prompt_template import gpt_structure
    from pilot_schedule import generate_daily_plan,generate_wake_hour
    for call in [lambda:generate_daily_plan(p,8),lambda:generate_wake_hour(p)]:
        calls=[]
        def bad(*a,**k):calls.append(1);return "not JSON"
        monkeypatch.setattr(gpt_structure,"_chat",bad)
        with pytest.raises(PilotScheduleError,match="after 3 attempts"):call()
        assert len(calls)==3


def test_actual_persona_loop_runs_every_ten_seconds_through_day2_routines(engine,monkeypatch):
    server,_,_=engine
    from persona.persona import Persona
    from persona.cognitive_modules import plan
    monkeypatch.setattr(Persona,"perceive",lambda *a:[])
    monkeypatch.setattr(Persona,"retrieve",lambda *a:{})
    monkeypatch.setattr(Persona,"reflect",lambda *a:None)
    monkeypatch.setattr(Persona,"execute",lambda self,*a:(self.scratch.curr_tile,"x",self.scratch.act_description))
    mock_location(monkeypatch)
    daily_calls=[]
    def wake(p):return (10 if p.name=="Hailey Johnson" else 4)+(p.scratch.curr_time.day-13)
    monkeypatch.setattr(plan,"generate_wake_up_hour",wake)
    def daily(p,w):
        daily_calls.append((p.name,p.scratch.curr_time.date().isoformat()))
        return [f"wake at {w}:00",f"work on day {p.scratch.curr_time.day}","have dinner at 18:00"]
    monkeypatch.setattr(plan,"generate_first_daily_plan",daily)
    monkeypatch.setattr(plan,"revise_identity",lambda p:None)
    monkeypatch.setattr(plan,"get_embedding",lambda *a:[0.1,0.2])
    def hourly(p,h,previous,hours,**kw):
        index=hours.index(h)
        return ["sleeping" if p.name=="Adam Smith" and index>=20 else f"read chapter at hour {index}"]
    monkeypatch.setattr(plan,"run_gpt_prompt_generate_hourly_schedule",hourly)
    decomp_calls=[]
    def parts(p,task,d):
        decomp_calls.append((p.name,p.scratch.curr_time.isoformat(),task))
        return [["read the first passage",d//2],["read the next passage",d-d//2]]
    monkeypatch.setattr(plan,"generate_task_decomp",parts)
    times={}
    start=datetime(2023,2,13)
    for tick in range(36*360+1):
        now=start+timedelta(seconds=tick*10)
        for name in ["Hailey Johnson","Adam Smith"]:
            p=server.personas[name]
            p.move(server.maze,server.personas,server.personas_tile[name],now)
            assert sum(d for a,d in p.scratch.f_daily_schedule)==1440
            if now.minute==0 and now.second==0:
                times[name,now]=p.scratch.act_description
    assert len(daily_calls)==4
    for day,wake_h,wake_a in [(13,10,4),(14,11,5)]:
        assert times["Hailey Johnson",datetime(2023,2,day,0)]!="sleeping"
        assert times["Hailey Johnson",datetime(2023,2,day,1)]!="sleeping"
        assert times["Hailey Johnson",datetime(2023,2,day,2)]=="sleeping"
        assert times["Hailey Johnson",datetime(2023,2,day,wake_h)]!="sleeping"
        assert times["Adam Smith",datetime(2023,2,day,wake_a-1)]=="sleeping"
        assert times["Adam Smith",datetime(2023,2,day,wake_a)]!="sleeping"
    assert all("(" not in task for name,t,task in decomp_calls)


def test_lookahead_recomputes_interval_positions_after_splice(engine):
    p=setup_persona(engine,now=datetime(2023,2,13),schedule=[
        ["painting",120],["reading",60],["cleaning",60],["sleeping",1200]])
    calls=[]
    def parts(p,t,d):
        calls.append(t)
        return [[t+" first step",d//2],[t+" next step",d-d//2]]
    prepare_actions(p,parts)
    assert calls==["painting"]
    p.scratch.curr_time=datetime(2023,2,13,1)
    prepare_actions(p,parts)
    assert calls==["painting","reading"]
    assert p.scratch.f_daily_schedule[:4]==[["painting first step",60],["painting next step",60],
                                           ["reading first step",30],["reading next step",30]]


def test_source_lock_includes_prompt_templates(monkeypatch,tmp_path):
    import run_midnight_pilot as runner
    root=tmp_path/"source"
    prompt=root/"persona/prompt_template/v2/prompt.txt"
    prompt.parent.mkdir(parents=True)
    for name in ["run_midnight_pilot.py","reverie.py","utils.py"]:(root/name).write_text("# fixture")
    monkeypatch.setattr(runner,"BACKEND",root)
    monkeypatch.setattr(runner,"SCRIPT",root/"run_midnight_pilot.py")
    prompt.write_text("version one")
    before=runner.source_digest()
    prompt.write_text("version two")
    assert before!=runner.source_digest()


def test_contract_diagnostics_record_bounded_rejection_and_acceptance(engine,monkeypatch):
    p=setup_persona(engine)
    from persona.prompt_template import gpt_structure
    from pilot_schedule import generate_wake_hour
    path=engine[2]/"schedule_diagnostics.jsonl"
    monkeypatch.setenv("NCN_SCHEDULE_LOG",str(path))
    responses=iter(["not JSON",'{"wake_hour":8}'])
    monkeypatch.setattr(gpt_structure,"_chat",lambda *a,**k:next(responses))
    assert generate_wake_hour(p)[0]==8
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert [r["status"] for r in rows]==["rejected","accepted"]
    assert [r["attempt"] for r in rows]==[1,2]
    assert all("response_sha256" in r and "response" not in r for r in rows)
