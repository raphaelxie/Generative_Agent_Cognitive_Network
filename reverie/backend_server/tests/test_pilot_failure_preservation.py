"""Injected connection/contract failures must not persist a partial tick."""
import json
from types import SimpleNamespace

import pytest
from test_pilot_engine import engine, replay_move


@pytest.mark.parametrize("failure",["api","schedule"])
def test_partial_tick_failure_leaves_last_full_snapshot_and_manifest(engine,monkeypatch,failure):
    import run_midnight_pilot as runner
    from persona.persona import Persona
    from pilot_api import PilotAPIError
    from pilot_schedule import PilotScheduleError
    from pilot_state import load_snapshot
    server,protocol,output=engine
    config=protocol["config"]
    (output/"preflight.json").write_text(json.dumps({"config":config}))
    monkeypatch.setattr(runner,"new_server",lambda *a:(server,protocol))
    count=[]
    def move(self,*args):
        count.append(1)
        if len(count)==2:
            self.scratch.daily_req.append("partial tick marker")
            raise (PilotAPIError if failure=="api" else PilotScheduleError)("injected")
        return replay_move(self,*args)
    monkeypatch.setattr(Persona,"move",move)
    with pytest.raises((PilotAPIError,PilotScheduleError)):
        runner.run(SimpleNamespace(defer_surveys=True),output,config)
    restored=load_snapshot(output/"latest.pkl.gz")
    assert server.step==restored["server"].step==0
    assert not (output/"live_progress.json").exists()
    assert all("partial tick marker" not in p.scratch.daily_req for p in restored["server"].personas.values())
    assert runner.verify_progress(output,config)["manifest_matches"]


def test_actual_chat_wrapper_records_transport_failure_once_and_latches(monkeypatch,tmp_path):
    import httpx
    from openai import APIConnectionError
    import pilot_api
    from persona.prompt_template import gpt_structure as gpt
    calls=[]
    def create(**kwargs):
        calls.append(kwargs)
        try:
            raise gpt._HardTimeout("injected deadline")
        except gpt._HardTimeout as cause:
            raise APIConnectionError(request=httpx.Request("POST","https://example.invalid")) from cause
    monkeypatch.setattr(gpt.client.chat.completions,"create",create)
    monkeypatch.setattr(gpt,"temp_sleep",lambda *a:None)
    # utils.py loads the project .env with override=True; pin what the runner pins.
    monkeypatch.setenv("NCN_PILOT","1")
    monkeypatch.setattr(gpt,"DEFAULT_CHAT_MODEL","gpt-6-luna")
    monkeypatch.setattr(pilot_api,"_FAILURE",None)
    path=tmp_path/"usage.jsonl"
    monkeypatch.setenv("NCN_USAGE_LOG",str(path))
    for _ in range(2):
        with pytest.raises(pilot_api.PilotAPIError):gpt._chat("short fixture")
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(calls)==len(rows)==1
    assert rows[0]["error_chain_types"]==["APIConnectionError","_HardTimeout"]
    assert rows[0]["usage"] is None
