#!/usr/bin/env python3
"""64-hour pilot with immutable snapshots and isolated survey subprocesses.

Run --help for preflight, execution, deferred measurements, report and fork.
This runner never imports or resumes the historical GPT-4o experiment.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

from pilot_control import SessionControl, SessionPaused

from pilot_protocol import (MODEL, PROTOCOL_VERSION, END_STEP, CHECK_STEPS,
                            SURVEY_STEPS, RECENT_MINUTES, CHECK_INTERVAL, CheckpointSelector,
                            graph_from_chats, graph_summary, priority_order,
                            labels_at, next_boundary)
from pilot_state import (atomic_json, load_snapshot, materialize_environment,
                         save_snapshot, state_digest)

SCRIPT = Path(__file__).resolve()
BACKEND = SCRIPT.parent
PROJECT = BACKEND.parent.parent


@contextmanager
def run_lock(output):
    import fcntl
    with (output/".pilot.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process is already writing this pilot output")
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def source_digest():
    digest = hashlib.sha256()
    files = sorted(set(BACKEND.glob("pilot_*.py")) | {SCRIPT, BACKEND/"reverie.py", BACKEND/"utils.py"}
                   | set((BACKEND/"persona").rglob("*.py"))
                   | set((BACKEND/"persona/prompt_template").rglob("*.txt")))
    for file in files:
        digest.update(str(file.relative_to(BACKEND)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def configure(args):
    # Import the user's ignored local settings before enforcing this run's
    # contract. Some existing utils.py files load .env with override=True.
    # Do not modify that untracked, user-specific configuration file.
    import utils
    from dotenv import load_dotenv
    load_dotenv(args.env_file, override=True)
    output = args.output.resolve()
    os.environ.update(NCN_PILOT="1", OPENAI_MODEL=MODEL, GA_SEED=str(args.seed),
                      GA_DEBUG="0", GA_RUN_TAG=output.name,
                      NCN_STORAGE_ROOT=str(output/"worlds"),
                      NCN_TEMP_ROOT=str(output/"frontend_temp"),
                      NCN_SCHEDULE_LOG=str(output/"schedule_diagnostics.jsonl"),
                      NCN_USAGE_LOG=str(output/"usage.jsonl"))
    utils.fs_storage = str(output/"worlds")
    utils.fs_temp_storage = str(output/"frontend_temp")
    utils.debug = False
    if os.environ.get("GA_MAZE_ASSETS"):
        utils.maze_assets_loc = os.environ["GA_MAZE_ASSETS"]
        utils.env_matrix = str(Path(utils.maze_assets_loc)/"the_ville/matrix")
        utils.env_visuals = str(Path(utils.maze_assets_loc)/"the_ville/visuals")
    # Hash iteration order is part of engine reproducibility. Re-exec before imports.
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.execv(sys.executable, [sys.executable, str(SCRIPT), *sys.argv[1:]])
    os.chdir(BACKEND)
    output.mkdir(parents=True, exist_ok=True)
    (output/"frontend_temp").mkdir(exist_ok=True)
    return output


def configuration(args):
    from pilot_tasks import VERSION as schedule_version
    return {"protocol_version": PROTOCOL_VERSION, "model": MODEL,
            "schedule_version": schedule_version,
            "embedding_model": os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            "chat_endpoint": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "embedding_endpoint": os.environ.get("OPENAI_EMBED_BASE_URL", "https://api.openai.com/v1"),
            "reasoning_effort": "none", "seed": args.seed, "survey_repeats": args.repeats,
            "source_digest": source_digest(), "python": sys.version.split()[0],
            "dependencies": {name: importlib.metadata.version(name) for name in
                             ("openai", "numpy", "networkx", "scipy", "scikit-learn")},
            "recent_minutes": RECENT_MINUTES, "seconds_per_step": 10,
            "end_step": END_STEP, "survey_version": "recent-network-v1"}


def location_preflight():
    """Real-model check of the action-path contracts with the real templates.

    The first pilot trajectory passed its interface preflight yet kept every agent
    at home, because the sector/arena prompts were never exercised. This fails
    before any simulation starts if the model cannot pick a public place.
    """
    from persona.prompt_template.gpt_structure import generate_prompt
    from pilot_actions import (choose_sector, choose_arena, generate_triple, decide_yes_no)
    sectors = ["Isabella Rodriguez's apartment", "The Rose and Crown Pub", "Hobbs Cafe",
               "Harvey Oak Supply Store", "The Willows Market and Pharmacy", "Johnson Park"]
    home_arenas = "main room, bathroom"
    name = "Isabella Rodriguez"
    task = "work the morning shift serving coffee and pastries at Hobbs Cafe"
    sector_input = [name, "Isabella Rodriguez's apartment", home_arenas, name,
                    "Isabella Rodriguez's apartment", home_arenas, "", ", ".join(sectors),
                    name, task, task, name]
    sector_prompt = generate_prompt(sector_input, "persona/prompt_template/v1/action_location_sector_v1.txt")
    chosen = choose_sector(sector_prompt, sectors)
    if chosen != "Hobbs Cafe":
        raise ValueError(f"Location preflight: the cafe shift resolved to {chosen!r}, not Hobbs Cafe")
    arenas = ["kitchen", "common room", "garden"]
    meal = "cook dinner"
    arena_input = [name, "Isabella Rodriguez's apartment", ", ".join(arenas), name, meal, meal, name,
                   "Isabella Rodriguez's apartment", ", ".join(arenas)]
    arena_prompt = generate_prompt(arena_input, "persona/prompt_template/v1/action_location_object_vMar11.txt")
    arena = choose_arena(arena_prompt, arenas)
    triple = generate_triple(generate_prompt([name, "serving coffee at the cafe counter", name],
                                             "persona/prompt_template/v2/generate_event_triple_v1.txt"))
    talk = decide_yes_no(
        "Task -- given context, determine whether the subject will initiate a conversation with another.\n"
        "Context: Isabella Rodriguez is serving coffee. Klaus Mueller just walked into Hobbs Cafe.\n"
        "Question: Would Isabella Rodriguez initiate a conversation with Klaus Mueller?\n\n"
        "Reasoning: Let's think step by step.")
    return {"sector_for_cafe_shift": chosen, "arena_for_dinner": arena,
            "triple": list(triple), "decide_to_talk_parsed": talk}


def preflight(output, config):
    from persona.prompt_template.gpt_structure import _chat, get_embedding, DEFAULT_CHAT_MODEL
    from pilot_api import set_context
    from pilot_survey import parse_answer
    if DEFAULT_CHAT_MODEL != MODEL:
        raise ValueError("Environment overrode the pilot model")
    if config["embedding_model"] != "text-embedding-3-small":
        raise ValueError("Embedding model must remain text-embedding-3-small")
    set_context("preflight", None, None)
    fixtures = [
        ("interaction", 'Return exactly {"p": [0.25, 0.75]}.', 2),
        ("degree", 'Return exactly {"ranking": ["Ada", "Bo"]}.', None),
        ("broker", 'Return exactly {"ranking": ["Bo", "Ada"]}.', None),
        ("community", 'Return exactly {"groups": [["Ada"], ["Bo"]]}.', None)]
    for instrument, prompt, count in fixtures:
        parse_answer(_chat(prompt, max_tokens=128), instrument, ["Ada", "Bo"], count)
    wake = _chat("Return only the integer 4.", max_tokens=15, stop=["\n"])
    if wake.strip() != "4":
        raise ValueError("Short generation compatibility check failed")
    vector = get_embedding("Pilot interface compatibility check.")
    if not vector or not all(isinstance(x, (int, float)) for x in vector):
        raise ValueError("Embedding preflight failed")
    location = location_preflight()
    evidence = {"status": "passed", "config": config, "location_checks": location,
                "embedding_dimensions": len(vector), "checked_at": datetime.now().isoformat()}
    atomic_json(output/"preflight.json", evidence)
    return evidence


def verify_template(template):
    from pilot_sleep import bedtime_hour
    meta = json.loads((template/"reverie/meta.json").read_text())
    if meta["step"] != 0 or len(meta["persona_names"]) != 25 or meta["sec_per_step"] != 10:
        raise ValueError("Expected a fresh 25-agent, 10-second template")
    for name in meta["persona_names"]:
        memory = template/"personas"/name/"bootstrap_memory"
        nodes = json.loads((memory/"associative_memory/nodes.json").read_text())
        if nodes:
            raise ValueError(f"Template has generated history: {name}")
        scratch = json.loads((memory/"scratch.json").read_text())
        if scratch.get("curr_time") or scratch.get("f_daily_schedule") or scratch.get("act_address"):
            raise ValueError(f"Template has live state: {name}")
        bedtime_hour(scratch["lifestyle"])
    return meta


def new_server(args, output, config):
    from reverie import ReverieServer
    template = args.template.resolve()
    meta = verify_template(template)
    world = output/"worlds/control"
    if world.exists():
        raise FileExistsError("World exists without a recoverable checkpoint; choose a new output directory")
    shutil.copytree(template, world)
    meta["curr_time"] = meta["start_date"] + ", 00:00:00"
    meta["step"] = 0
    atomic_json(world/"reverie/meta.json", meta)
    server = ReverieServer("control", "control")
    server.server_sleep = 0
    server.game_obj_cleanup = {}
    random.seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    protocol = {"config": config, "selector": CheckpointSelector(
        priority_order(server.personas, args.seed)).to_dict(),
        "network_history": [], "snapshots": {}, "status": "running",
        "arm": "control", "template": str(template)}
    return server, protocol


def write_manifest(output, server, protocol):
    selector = CheckpointSelector(**protocol["selector"])
    manifest = {"run_name": output.name, "config": protocol["config"], "status": protocol["status"],
                "arm": protocol["arm"], "step": server.step,
                "sim_time": server.curr_time.isoformat(),
                "checkpoint_status": selector.status(server.step),
                "selected_checkpoint": selector.selected,
                "snapshots": protocol["snapshots"],
                "template": protocol["template"],
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "pause": protocol.get("pause")}
    atomic_json(output/"manifest.json", manifest)
    atomic_json(output/"checkpoint_candidates.json", selector.records)
    atomic_json(output/"network_history.json", protocol["network_history"])


def measure_snapshot(output, path, step, repeats, server=None, control=None):
    before = state_digest(server) if server is not None else None
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    for repeat in range(1, repeats + 1):
        destination = output/"surveys"/f"step_{step:06d}"/f"repeat_{repeat:02d}"
        if (destination/"complete.json").exists():
            continue
        if control:
            control.check("survey")
        command = [sys.executable, str(SCRIPT), "_survey", "--output", str(output),
                   "--snapshot", str(path), "--survey-output", str(destination),
                   "--repeat", str(repeat), "--seed", os.environ["GA_SEED"]]
        if control is None:
            subprocess.run(command, check=True, cwd=BACKEND, env=os.environ.copy())
        else:
            child = subprocess.Popen(command, cwd=BACKEND, env=os.environ.copy())
            try:
                while child.poll() is None:
                    control.check("survey")
                    time.sleep(0.5)
                if child.returncode:
                    raise subprocess.CalledProcessError(child.returncode, command)
                control.check("survey")
            finally:
                # Survey is disposable. Its immutable source was saved before launch.
                # Completed respondents remain available; a partial respondent is retried.
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                if hashlib.sha256(path.read_bytes()).hexdigest() != source_hash:
                    raise RuntimeError("Survey changed its source snapshot")
                if server is not None and state_digest(server) != before:
                    raise RuntimeError("Survey changed live simulation state or RNG")
    if hashlib.sha256(path.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("Survey changed its source snapshot")
    if server is not None and state_digest(server) != before:
        raise RuntimeError("Survey changed live simulation state or RNG")
    atomic_json(output/"surveys"/f"step_{step:06d}"/"isolation_check.json",
                {"source_unchanged": True, "live_state_unchanged": True if server is not None else None,
                 "live_digest": before, "snapshot_sha256": source_hash})


def save_progress(output, server, protocol):
    world = output/"worlds"/server.sim_code
    materialize_environment(server, world)
    save_snapshot(output/"latest.pkl.gz", server, protocol)
    write_manifest(output, server, protocol)


def process_boundary(output, server, protocol, defer_surveys=False, control=None):
    selector = CheckpointSelector(**protocol["selector"])
    cumulative = graph_from_chats(server.personas, server.start_time, server.curr_time)
    recent = graph_from_chats(server.personas, server.start_time, server.curr_time, RECENT_MINUTES)
    if server.step in CHECK_STEPS and (not selector.records or selector.records[-1]["step"] < server.step):
        selector.observe(server.step, cumulative)
    protocol["selector"] = selector.to_dict()
    if not protocol["network_history"] or protocol["network_history"][-1]["step"] != server.step:
        protocol["network_history"].append({"step": server.step, "sim_time": server.curr_time.isoformat(),
            "cumulative": graph_summary(cumulative), "recent": graph_summary(recent),
            "activities": {name: persona.scratch.act_description for name, persona in server.personas.items()}})
    selected_step = selector.selected["step"] if selector.selected else None
    labels = labels_at(server.step, selected_step)
    materialize_environment(server, output/"worlds"/server.sim_code)
    if labels:
        path = output/"snapshots"/f"step_{server.step:06d}.pkl.gz"
        protocol["snapshots"][str(server.step)] = {"path": str(path.relative_to(output)), "labels": labels}
        if not path.exists():
            save_snapshot(path, server, protocol)
        elif load_snapshot(path, restore_random=False)["state_digest"] != state_digest(server):
            raise RuntimeError("Existing immutable snapshot does not match current state")
    # Persist protocol decisions BEFORE measuring, so an interrupted survey resumes the same state.
    save_progress(output, server, protocol)
    if labels and set(labels) & {"initial", "scheduled", "final", "checkpoint"} and not defer_surveys:
        measure_snapshot(output, path, server.step, protocol["config"]["survey_repeats"], server, control)
    return selected_step


def run(args, output, config):
    control = SessionControl(getattr(args, "stop_at", None), getattr(args, "survey_cutoff", None),
                             getattr(args, "stop_file", None))
    control.check("simulation")
    evidence = output/"preflight.json"
    if evidence.exists():
        if json.loads(evidence.read_text())["config"] != config:
            raise ValueError("Preflight configuration changed; rerun preflight in a new run directory")
    else:
        preflight(output, config)
    latest = output/"latest.pkl.gz"
    if latest.exists():
        payload = load_snapshot(latest)
        server, protocol = payload["server"], payload["protocol"]
        if protocol["config"] != config:
            raise ValueError("Resume requires the original source, dependencies, seed and model configuration")
        materialize_environment(server, output/"worlds"/server.sim_code)
    else:
        server, protocol = new_server(args, output, config)
    if any(persona.isolated for persona in server.personas.values()):
        raise ValueError("The pilot runner only runs an unperturbed trajectory")
    protocol["status"] = "running"
    protocol.pop("pause", None)
    last_save = time.monotonic()
    try:
        while True:
            selected = protocol["selector"]["selected"]
            selected_step = selected["step"] if selected else None
            # A wall-clock pause can land between the 30-minute protocol boundaries.
            # Do not turn that operational recovery point into a new measurement.
            if server.step % CHECK_INTERVAL == 0:
                selected_step = process_boundary(output, server, protocol, args.defer_surveys, control)
                last_save = time.monotonic()
                print(f"Pilot: {server.curr_time.isoformat()} step={server.step}/{END_STEP}", flush=True)
            control.check("simulation")
            # Operational limit (not part of the config lock): pause only here, at
            # a protocol boundary that process_boundary has just persisted.
            stop_step = getattr(args, "stop_at_step", None)
            if stop_step is not None and server.step >= stop_step and server.step != END_STEP:
                raise SessionPaused("stop_at_step")
            if server.step == END_STEP:
                break
            target = next_boundary(server.step, selected_step)
            while server.step < target:
                control.check("simulation")
                expected = server.step + 1
                server.start_server(1)
                if server.step != expected:
                    raise RuntimeError("Engine did not complete its requested tick")
                # Only a fully completed tick can be saved. Never pickle half a tick.
                atomic_json(output/"live_progress.json", {
                    "step": server.step, "sim_time": server.curr_time.isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat()})
                if time.monotonic() - last_save >= 60 and server.step < target:
                    save_progress(output, server, protocol)
                    last_save = time.monotonic()
                # Let the outer loop persist all scientific events at a boundary
                # before stopping there (including a newly selected checkpoint).
                if server.step < target:
                    control.check("simulation")
        # A previous invocation may have deferred earlier waves.
        if not args.defer_surveys:
            measure_saved_snapshots(output, protocol, server, control)
        protocol["status"] = "simulation_complete_surveys_pending" if args.defer_surveys else "complete"
        save_progress(output, server, protocol)
        server.save()
    except SessionPaused as pause:
        pause_progress(output, server, protocol, str(pause))
    from pilot_report import write_report
    write_report(output)


def pause_progress(output, server, protocol, reason):
    protocol["status"] = "paused"
    protocol["pause"] = {"reason": reason, "wall_time": datetime.now(timezone.utc).isoformat()}
    save_progress(output, server, protocol)
    print(f"Paused and saved: {server.curr_time.isoformat()} step={server.step} reason={reason}", flush=True)


def measure_saved_snapshots(output, protocol, server=None, control=None):
    for step, entry in sorted(protocol["snapshots"].items(), key=lambda pair: int(pair[0])):
        if set(entry["labels"]) & {"initial", "scheduled", "final", "checkpoint"}:
            measure_snapshot(output, output/entry["path"], int(step), protocol["config"]["survey_repeats"], server, control)


def measure_all(output, control=None):
    payload = load_snapshot(output/"latest.pkl.gz")
    server, protocol = payload["server"], payload["protocol"]
    try:
        measure_saved_snapshots(output, protocol, server, control)
        if server.step == END_STEP:
            protocol["status"] = "complete"
            protocol.pop("pause", None)
            save_progress(output, server, protocol)
    except SessionPaused as pause:
        pause_progress(output, server, protocol, str(pause))
    from pilot_report import write_report
    write_report(output)


def verify_progress(output, config):
    """Read and validate trusted local recovery state without calling an API."""
    payload = load_snapshot(output/"latest.pkl.gz", restore_random=False)
    server, protocol = payload["server"], payload["protocol"]
    if protocol["config"] != config:
        raise ValueError("Snapshot configuration does not match this runner")
    manifest = json.loads((output/"manifest.json").read_text())
    matches = (manifest["step"] == server.step
               and manifest["sim_time"] == server.curr_time.isoformat()
               and manifest["status"] == protocol["status"]
               and manifest["selected_checkpoint"] == protocol["selector"]["selected"])
    return {"step": server.step, "sim_time": server.curr_time.isoformat(),
            "status": protocol["status"], "state_digest": payload["state_digest"],
            "snapshot_valid": True, "manifest_matches": matches,
            "agents": len(server.personas), "pause": protocol.get("pause"),
            "selected_checkpoint": protocol["selector"]["selected"]}


def fork_checkpoint(output, destination, arm):
    """Prepare a future branch only; never advance the world or call an API."""
    manifest = json.loads((output/"manifest.json").read_text())
    selected = manifest["selected_checkpoint"]
    if not selected:
        raise ValueError("This trajectory has no eligible checkpoint")
    entry = manifest["snapshots"][str(selected["step"])]
    payload = load_snapshot(output/entry["path"])
    server, protocol = payload["server"], payload["protocol"]
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    target = selected[arm] if arm in ("hub", "broker") else None
    for name, persona in server.personas.items():
        persona.isolated = name == target
    protocol["arm"] = arm
    protocol["status"] = "branch_prepared"
    protocol["intervention"] = {"arm": arm, "target": target, "step": server.step,
        "sim_time": server.curr_time.isoformat(), "source_digest": payload["state_digest"],
        "ongoing_chats": {name: p.scratch.chatting_with for name, p in server.personas.items()
                          if p.scratch.chatting_with},
        "definition": "Prevent new interactions in both directions; existing conversations are retained"}
    save_snapshot(destination/"branch.pkl.gz", server, protocol)
    atomic_json(destination/"intervention.json", protocol["intervention"])
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("describe", "preflight", "run", "measure", "report", "verify", "fork", "_survey"))
    parser.add_argument("--output", type=Path, default=PROJECT/"pilot_runs/midnight_luna_seed104729")
    parser.add_argument("--template", type=Path, default=PROJECT/"environment/frontend_server/storage/base_the_ville_n25")
    parser.add_argument("--env-file", type=Path, default=PROJECT/".env")
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--defer-surveys", action="store_true", help="Save survey states now; measure later in isolated workers")
    parser.add_argument("--survey-cutoff", help="Timezone-aware wall time: pause if a survey is active or due at/after this time")
    parser.add_argument("--stop-at", help="Timezone-aware wall time: save and pause after a complete engine tick")
    parser.add_argument("--stop-file", type=Path, help="Pause safely when this file exists; remove explicitly before resuming")
    parser.add_argument("--stop-at-step", type=int, help="Save and pause at this protocol boundary step (multiple of 180); an operational limit, not part of the experiment configuration")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--survey-output", type=Path)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--arm", choices=("control", "hub", "broker"), default="control")
    args = parser.parse_args()
    for field in ("output", "template", "env_file", "snapshot", "survey_output", "destination", "stop_file"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.resolve())
    if not 0 <= args.seed < 2**32 or args.repeats < 2:
        parser.error("Seed must fit uint32 and at least two repeats are needed for noise estimation")
    if args.stop_at_step is not None and (args.command != "run" or args.stop_at_step <= 0
                                          or args.stop_at_step > END_STEP
                                          or args.stop_at_step % CHECK_INTERVAL):
        parser.error(f"--stop-at-step applies to run and must be a positive multiple of {CHECK_INTERVAL} up to {END_STEP}")
    if args.command == "describe":
        print(json.dumps({"protocol": PROTOCOL_VERSION, "model": MODEL,
              "hours": 64, "end_step": END_STEP, "survey_steps": SURVEY_STEPS,
              "checkpoint_checks": CHECK_STEPS, "repeats": args.repeats}, indent=2))
        return
    output = configure(args)
    if args.command == "_survey":
        from pilot_survey import survey_worker
        survey_worker(args.snapshot, args.survey_output, args.repeat)
        return
    if args.command == "report":
        from pilot_report import write_report
        write_report(output)
        return
    if args.command == "fork":
        if args.destination is None:
            parser.error("fork requires --destination")
        with run_lock(output):
            fork_checkpoint(output, args.destination.resolve(), args.arm)
        return
    config = configuration(args)
    with run_lock(output):
        if args.command == "preflight":
            if (output/"latest.pkl.gz").exists():
                raise ValueError("Do not replace preflight evidence after the pilot starts")
            preflight(output, config)
        elif args.command == "run":
            run(args, output, config)
        elif args.command == "verify":
            print(json.dumps(verify_progress(output, config), ensure_ascii=False, indent=2))
        elif args.command == "measure":
            manifest = json.loads((output/"manifest.json").read_text())
            if manifest["config"] != config:
                raise ValueError("Deferred surveys require the original experiment configuration")
            measure_all(output, SessionControl(args.stop_at, args.survey_cutoff, args.stop_file))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Resume from the last completed recovery checkpoint.", file=sys.stderr)
        sys.exit(130)
    except BaseException as error:
        if isinstance(error, SystemExit):
            raise
        # Avoid echoing provider URLs, authorization headers or credentials in exception bodies.
        from pilot_schedule import PilotScheduleError
        detail = str(error) if type(error).__module__ == 'builtins' or isinstance(error, PilotScheduleError) else 'see local diagnostic logs'
        print(f"Pilot stopped: {type(error).__name__}: {detail}", file=sys.stderr)
        sys.exit(1)
