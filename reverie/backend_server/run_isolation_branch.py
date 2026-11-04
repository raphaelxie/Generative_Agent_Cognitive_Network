#!/usr/bin/env python3
"""Day 1 16:00 named soft-isolation branches.

Outside the frozen runtime digest (pilot_*.py, run_midnight_pilot.py,
reverie.py, utils.py, persona/). Replays main's 14:00 snapshot to 16:00,
forks Jennifer Moore / Carlos Gomez / control, and runs each branch to
Day 2 06:00. Surveys stay in measure_css_survey.py.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone

from pilot_control import SessionControl, SessionPaused
from pilot_protocol import (CHECK_INTERVAL, CHECK_STEPS, RECENT_MINUTES,
                            CheckpointSelector, graph_from_chats, graph_summary,
                            next_boundary)
from pilot_state import (atomic_json, load_snapshot, materialize_environment,
                         save_snapshot, state_digest)
import run_midnight_pilot as runner

SCRIPT = Path(__file__).resolve()
BACKEND = SCRIPT.parent
PROJECT = BACKEND.parent.parent
EXPECTED_DIGEST = "29cfd946e0d44df4cddc0f3938bb74a3ac12875ec3e81f5d27c379c8b1cd2813"
JM = "Jennifer Moore"
CG = "Carlos Gomez"
ARMS = {"JM": JM, "CG": CG, "control": None}
ORIGIN_STEP = 14 * 360
CHECKPOINT_STEP = 16 * 360
BRANCH_END_STEP = (24 + 6) * 360
REPLAY_MEASURE_STEPS = {CHECKPOINT_STEP: ["checkpoint"]}
BRANCH_ORIGIN_LABELS = ["branch_origin"]
BRANCH_MEASURE_STEPS = {
    18 * 360: ["scheduled"],
    22 * 360: ["scheduled"],
    (24 + 2) * 360: ["scheduled"],
    BRANCH_END_STEP: ["scheduled"],
}


class TargetsChanged(RuntimeError):
    """Replay 16:00 network no longer matches the named JM/CG assignment."""


def isolated_names(server):
    return sorted(name for name, persona in server.personas.items() if persona.isolated)


def talk_ready_except_isolation(init, target):
    if (not target.scratch.act_address or not target.scratch.act_description
            or not init.scratch.act_address or not init.scratch.act_description):
        return False
    if "sleeping" in target.scratch.act_description or "sleeping" in init.scratch.act_description:
        return False
    if init.scratch.curr_time.hour == 23:
        return False
    if "<waiting>" in target.scratch.act_address:
        return False
    if target.scratch.chatting_with or init.scratch.chatting_with:
        return False
    buffered = init.scratch.chatting_with_buffer.get(target.name, 0)
    return buffered <= 0


def react_ready_except_isolation(init, target):
    if (not target.scratch.act_address or not target.scratch.act_description
            or not init.scratch.act_address or not init.scratch.act_description):
        return False
    if "sleeping" in target.scratch.act_description or "sleeping" in init.scratch.act_description:
        return False
    if init.scratch.curr_time.hour == 23:
        return False
    if "waiting" in target.scratch.act_description:
        return False
    if init.scratch.planned_path == []:
        return False
    return init.scratch.act_address == target.scratch.act_address


def install_block_recorder(context):
    """Record the first isolation gate hit without changing talk/react decisions."""
    import persona.cognitive_modules.plan as plan_mod
    if getattr(plan_mod._should_react, "_ncn_block_recorder", False):
        plan_mod._should_react._ncn_context = context
        return plan_mod._should_react
    original = plan_mod._should_react

    def wrapped(persona, retrieved, personas):
        protocol = context.get("protocol") or {}
        intervention = protocol.get("intervention")
        if intervention is not None and intervention.get("first_blocked") is None:
            event = retrieved.get("curr_event")
            target = personas.get(getattr(event, "subject", None)) if event is not None else None
            if (event is not None and ":" not in getattr(event, "subject", ":")
                    and target is not None and (persona.isolated or target.isolated)):
                kind = None
                if talk_ready_except_isolation(persona, target):
                    kind = "talk"
                elif react_ready_except_isolation(persona, target):
                    kind = "react"
                if kind:
                    server = context["server"]
                    record = {"kind": kind, "initiator": persona.name, "target": target.name,
                              "step": server.step, "sim_time": server.curr_time.isoformat(),
                              "initiator_isolated": bool(persona.isolated),
                              "target_isolated": bool(target.isolated)}
                    intervention["first_blocked"] = record
                    output = context.get("output")
                    if output is not None:
                        atomic_json(Path(output) / "intervention.json", intervention)
        return original(persona, retrieved, personas)

    wrapped._ncn_block_recorder = True
    wrapped._ncn_context = context
    plan_mod._should_react = wrapped
    return wrapped


def structure_metrics(server):
    import networkx as nx
    graph = graph_from_chats(server.personas, server.start_time, server.curr_time)
    degrees = dict(graph.degree())
    between = nx.betweenness_centrality(graph, normalized=True, weight=None)
    return graph, degrees, between


def evaluate_targets(degrees, between):
    if JM not in degrees or CG not in degrees:
        raise KeyError("Jennifer Moore and Carlos Gomez must both be on the roster")
    max_between = max(between.values())
    jm_unique_first = (abs(between[JM] - max_between) <= 1e-12
                       and all(name == JM or abs(value - max_between) > 1e-12
                               for name, value in between.items()))
    ranked = sorted(degrees.values(), reverse=True)
    cutoff = ranked[1] if len(ranked) > 1 else ranked[0]
    leaders = {name for name, value in degrees.items() if value >= cutoff}
    degree_pair = leaders == {JM, CG}
    return {
        "ok": jm_unique_first and degree_pair,
        "jennifer_unique_max_betweenness": jm_unique_first,
        "degree_leaders_are_jennifer_and_carlos": degree_pair,
        "jennifer_degree": degrees[JM], "carlos_degree": degrees[CG],
        "jennifer_betweenness": between[JM], "carlos_betweenness": between[CG],
        "max_betweenness": max_between, "degree_leaders": sorted(leaders),
        "degrees": degrees, "betweenness": between,
    }


def inspect_server(server):
    graph, degrees, between = structure_metrics(server)
    report = evaluate_targets(degrees, between)
    report.update(step=server.step, sim_time=server.curr_time.isoformat(),
                  cumulative=graph_summary(graph),
                  recent=graph_summary(graph_from_chats(
                      server.personas, server.start_time, server.curr_time, RECENT_MINUTES)))
    return report


def require_locked_config(config):
    if config["source_digest"] != EXPECTED_DIGEST or runner.source_digest() != EXPECTED_DIGEST:
        raise RuntimeError("Frozen runtime source digest changed")
    if config["model"] != "gpt-6-luna" or config["seed"] != 104729:
        raise RuntimeError("Model or seed differs from the locked configuration")


def bind_storage(output):
    """Keep utils and a possibly already-imported reverie module on this output."""
    import utils
    output = Path(output)
    utils.fs_storage = str(output / "worlds")
    utils.fs_temp_storage = str(output / "frontend_temp")
    (output / "frontend_temp").mkdir(exist_ok=True)
    (output / "worlds").mkdir(exist_ok=True)
    if "reverie" in sys.modules:
        import reverie
        reverie.fs_storage = utils.fs_storage
        reverie.fs_temp_storage = utils.fs_temp_storage


def copy_preflight(source, destination, config):
    evidence = Path(source) / "preflight.json"
    if not evidence.exists():
        raise FileNotFoundError("Source preflight.json is required; do not rerun preflight")
    payload = json.loads(evidence.read_text())
    if payload["config"] != config:
        raise ValueError("Source preflight configuration does not match this runner")
    shutil.copy2(evidence, Path(destination) / "preflight.json")


def seed_from_snapshot(source, destination, config, step=ORIGIN_STEP):
    source, destination = Path(source), Path(destination)
    snapshot = source / "snapshots" / f"step_{step:06d}.pkl.gz"
    if not snapshot.exists():
        raise FileNotFoundError(snapshot)
    if (destination / "latest.pkl.gz").exists():
        return destination
    scaffolding = {".pilot.lock", ".session.lock", "frontend_temp", "worlds", "preflight.json"}
    leftover = [path for path in destination.iterdir() if path.name not in scaffolding] if destination.exists() else []
    if leftover:
        raise FileExistsError(f"{destination} already exists; choose a new directory or resume it")
    destination.mkdir(parents=True, exist_ok=True)
    payload = load_snapshot(snapshot)
    server, protocol = payload["server"], payload["protocol"]
    if protocol["config"] != config:
        raise ValueError("Source snapshot configuration does not match this runner")
    if isolated_names(server):
        raise ValueError("The 14:00 origin cannot start from an isolated state")
    if server.step != step:
        raise ValueError(f"Origin snapshot step is {server.step}, not {step}")
    copy_preflight(source, destination, config)
    bind_storage(destination)
    protocol = dict(protocol)
    protocol["snapshots"] = {}
    protocol["arm"] = "replay"
    protocol["status"] = "running"
    protocol.pop("pause", None)
    protocol["replay"] = {
        "source_run": str(source), "source_snapshot": str(snapshot.relative_to(source)),
        "source_state_digest": payload["state_digest"], "origin_step": step,
        "checkpoint_step": CHECKPOINT_STEP,
        "note": "Day 1 16:00 reconstructed from Day 1 14:00; not identical to main",
    }
    materialize_environment(server, destination / "worlds" / server.sim_code)
    origin = destination / "snapshots" / f"step_{step:06d}.pkl.gz"
    protocol["snapshots"][str(step)] = {"path": str(origin.relative_to(destination)),
                                        "labels": ["replay_origin"]}
    save_snapshot(origin, server, protocol)
    runner.save_progress(destination, server, protocol)
    return destination


def process_step(output, server, protocol, measure_steps):
    selector = CheckpointSelector(**protocol["selector"])
    cumulative = graph_from_chats(server.personas, server.start_time, server.curr_time)
    recent = graph_from_chats(server.personas, server.start_time, server.curr_time, RECENT_MINUTES)
    if server.step in CHECK_STEPS and (not selector.records or selector.records[-1]["step"] < server.step):
        selector.observe(server.step, cumulative)
    protocol["selector"] = selector.to_dict()
    if not protocol["network_history"] or protocol["network_history"][-1]["step"] != server.step:
        protocol["network_history"].append({
            "step": server.step, "sim_time": server.curr_time.isoformat(),
            "cumulative": graph_summary(cumulative), "recent": graph_summary(recent),
            "activities": {name: persona.scratch.act_description
                           for name, persona in server.personas.items()}})
    labels = list(measure_steps.get(server.step, []))
    materialize_environment(server, output / "worlds" / server.sim_code)
    if labels:
        path = output / "snapshots" / f"step_{server.step:06d}.pkl.gz"
        protocol["snapshots"][str(server.step)] = {"path": str(path.relative_to(output)),
                                                   "labels": labels}
        if not path.exists():
            save_snapshot(path, server, protocol)
        elif load_snapshot(path, restore_random=False)["state_digest"] != state_digest(server):
            raise RuntimeError("Existing immutable snapshot does not match current state")
    runner.save_progress(output, server, protocol)
    selected = selector.selected
    return selected["step"] if selected else None


def assert_isolation(server, arm):
    target = ARMS[arm]
    found = isolated_names(server)
    if target is None:
        if found:
            raise ValueError(f"Control arm is isolated: {found}")
        return
    if found != [target]:
        raise ValueError(f"Arm {arm} must isolate only {target}; found {found}")


def advance(args, output, config, end_step, measure_steps, allow_isolation, complete_status):
    control = SessionControl(getattr(args, "stop_at", None), getattr(args, "survey_cutoff", None),
                             getattr(args, "stop_file", None))
    control.check("simulation")
    evidence = output / "preflight.json"
    if not evidence.exists():
        raise ValueError("Branch directories reuse the source preflight; do not start a new one")
    if json.loads(evidence.read_text())["config"] != config:
        raise ValueError("Preflight configuration changed; use a new output directory")
    bind_storage(output)
    payload = load_snapshot(output / "latest.pkl.gz")
    server, protocol = payload["server"], payload["protocol"]
    if protocol["config"] != config:
        raise ValueError("Resume requires the original source, dependencies, seed and model configuration")
    if allow_isolation:
        assert_isolation(server, protocol["arm"])
    elif isolated_names(server):
        raise ValueError("Replay only runs an unperturbed trajectory")
    materialize_environment(server, output / "worlds" / server.sim_code)
    protocol["status"] = "running"
    protocol.pop("pause", None)
    context = {"server": server, "protocol": protocol, "output": output}
    if allow_isolation and isolated_names(server):
        install_block_recorder(context)
    last_save = time.monotonic()
    try:
        while True:
            selected = protocol["selector"]["selected"]
            selected_step = selected["step"] if selected else None
            if server.step % CHECK_INTERVAL == 0:
                process_step(output, server, protocol, measure_steps)
                last_save = time.monotonic()
                print(f"Branch: {server.curr_time.isoformat()} step={server.step}/{end_step} "
                      f"arm={protocol.get('arm')}", flush=True)
            control.check("simulation")
            stop_step = getattr(args, "stop_at_step", None)
            if stop_step is not None and server.step >= stop_step and server.step != end_step:
                raise SessionPaused("stop_at_step")
            if server.step >= end_step:
                break
            target = min(end_step, next_boundary(server.step, selected_step))
            while server.step < target:
                control.check("simulation")
                expected = server.step + 1
                context["server"] = server
                server.start_server(1)
                if server.step != expected:
                    raise RuntimeError("Engine did not complete its requested tick")
                atomic_json(output / "live_progress.json", {
                    "step": server.step, "sim_time": server.curr_time.isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat()})
                if time.monotonic() - last_save >= 60 and server.step < target:
                    runner.save_progress(output, server, protocol)
                    last_save = time.monotonic()
                if server.step < target:
                    control.check("simulation")
        protocol["status"] = complete_status
        protocol.pop("pause", None)
        runner.save_progress(output, server, protocol)
    except SessionPaused as pause:
        runner.pause_progress(output, server, protocol, str(pause))
    from pilot_report import write_report
    write_report(output)
    return server, protocol


def destination_scaffolding(destination):
    destination = Path(destination)
    if (destination / "latest.pkl.gz").exists() or (destination / "branch.pkl.gz").exists():
        return False
    scaffolding = {".pilot.lock", ".session.lock", "frontend_temp", "worlds", "preflight.json"}
    leftover = [path for path in destination.iterdir() if path.name not in scaffolding] if destination.exists() else []
    return not leftover


def fork_arm(source, destination, arm, config, step=CHECKPOINT_STEP):
    source, destination = Path(source), Path(destination)
    if arm not in ARMS:
        raise ValueError(f"Arm must be one of {sorted(ARMS)}")
    if (destination / "latest.pkl.gz").exists() or (destination / "branch.pkl.gz").exists():
        raise FileExistsError(f"{destination} already has a forked snapshot")
    if destination.exists() and not destination_scaffolding(destination):
        raise FileExistsError(f"{destination} already exists; choose a new directory or resume it")
    snapshot = source / "snapshots" / f"step_{step:06d}.pkl.gz"
    payload = load_snapshot(snapshot)
    server, protocol = payload["server"], payload["protocol"]
    if protocol["config"] != config:
        raise ValueError("Checkpoint configuration does not match this runner")
    if server.step != step:
        raise ValueError(f"Checkpoint step is {server.step}, not {step}")
    if isolated_names(server):
        raise ValueError("Checkpoint must be unperturbed before a named fork")
    destination.mkdir(parents=True, exist_ok=True)
    copy_preflight(source, destination, config)
    bind_storage(destination)
    target = ARMS[arm]
    for name, persona in server.personas.items():
        persona.isolated = name == target
    protocol = dict(protocol)
    protocol["arm"] = arm
    protocol["status"] = "running"
    protocol.pop("pause", None)
    protocol["snapshots"] = {}
    protocol["intervention"] = {
        "arm": arm, "target": target, "step": server.step,
        "sim_time": server.curr_time.isoformat(),
        "source_run": str(source), "source_state_digest": payload["state_digest"],
        "ongoing_chats": {name: persona.scratch.chatting_with
                          for name, persona in server.personas.items() if persona.scratch.chatting_with},
        "definition": ("Prevent new interactions in both directions; existing conversations are retained. "
                       "The isolated person still moves, perceives, plans, and forms memories."),
        "first_blocked": None, "post_hoc_checkpoint": True,
        "note": "Named targets; not the preregistered hub/broker lock",
    }
    origin = destination / "snapshots" / f"step_{step:06d}.pkl.gz"
    protocol["snapshots"][str(step)] = {
        "path": str(origin.relative_to(destination)), "labels": BRANCH_ORIGIN_LABELS}
    materialize_environment(server, destination / "worlds" / server.sim_code)
    save_snapshot(origin, server, protocol)
    save_snapshot(destination / "branch.pkl.gz", server, protocol)
    runner.save_progress(destination, server, protocol)
    atomic_json(destination / "intervention.json", protocol["intervention"])
    return protocol["intervention"]


def inspect_output(output):
    payload = load_snapshot(Path(output) / "latest.pkl.gz", restore_random=False)
    report = inspect_server(payload["server"])
    report["state_digest"] = payload["state_digest"]
    report["arm"] = payload["protocol"].get("arm")
    return report


def verify_output(output, config):
    checked = runner.verify_progress(Path(output), config)
    payload = load_snapshot(Path(output) / "latest.pkl.gz", restore_random=False)
    checked["arm"] = payload["protocol"].get("arm")
    checked["isolated"] = isolated_names(payload["server"])
    checked["intervention"] = payload["protocol"].get("intervention")
    return checked


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("replay", "inspect", "fork", "run", "verify"))
    parser.add_argument("--source-run", type=Path, default=PROJECT / "pilot_runs/smoke_mobility_v5")
    parser.add_argument("--output", type=Path, default=PROJECT / "pilot_runs/smoke_mobility_v5_checkpoint_1600")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--arm", choices=tuple(ARMS), default="JM")
    parser.add_argument("--env-file", type=Path, default=PROJECT / ".env")
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--repeats", type=int, default=2,
                        help="Frozen trajectory repeats; v2 surveys use measure_css_survey.py --repeats 3")
    parser.add_argument("--stop-at", help="Timezone-aware wall time: save and pause after a complete tick")
    parser.add_argument("--survey-cutoff", help="Unused; surveys run through measure_css_survey.py")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--stop-at-step", type=int)
    parser.add_argument("--template", type=Path,
                        default=PROJECT / "environment/frontend_server/storage/base_the_ville_n25")
    args = parser.parse_args(argv)
    for field in ("source_run", "output", "destination", "env_file", "stop_file", "template"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.resolve())
    if args.stop_at_step is not None and (args.stop_at_step <= 0 or args.stop_at_step % CHECK_INTERVAL):
        parser.error(f"--stop-at-step must be a positive multiple of {CHECK_INTERVAL}")
    return parser, args


def main(argv=None):
    parser, args = parse_args(argv)
    if args.command == "fork":
        if args.destination is None:
            parser.error("fork requires --destination")
        args.output = args.destination
    output = runner.configure(args)
    config = runner.configuration(args)
    require_locked_config(config)
    with runner.run_lock(output):
        if args.command == "replay":
            seed_from_snapshot(args.source_run, output, config)
            advance(args, output, config, CHECKPOINT_STEP, REPLAY_MEASURE_STEPS,
                    allow_isolation=False, complete_status="checkpoint_ready")
            report = inspect_output(output)
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
            if not report["ok"]:
                raise TargetsChanged(
                    "Replay 16:00 no longer has Jennifer Moore as unique max betweenness "
                    "or Jennifer Moore and Carlos Gomez as the unique degree top two")
        elif args.command == "inspect":
            report = inspect_output(output)
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
            if not report["ok"]:
                sys.exit(2)
        elif args.command == "fork":
            intervention = fork_arm(args.source_run, output, args.arm, config)
            print(json.dumps(intervention, ensure_ascii=False, indent=2, default=str))
        elif args.command == "run":
            measures = {CHECKPOINT_STEP: BRANCH_ORIGIN_LABELS, **BRANCH_MEASURE_STEPS}
            advance(args, output, config, BRANCH_END_STEP, measures,
                    allow_isolation=True, complete_status="simulation_complete_surveys_pending")
            print(json.dumps(verify_output(output, config), ensure_ascii=False, indent=2, default=str))
        elif args.command == "verify":
            print(json.dumps(verify_output(output, config), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Resume from the last completed recovery checkpoint.", file=sys.stderr)
        sys.exit(130)
    except TargetsChanged as error:
        print(f"Stop for user decision: {error}", file=sys.stderr)
        sys.exit(2)
    except BaseException as error:
        if isinstance(error, SystemExit):
            raise
        from pilot_schedule import PilotScheduleError
        detail = str(error) if type(error).__module__ == "builtins" or isinstance(error, PilotScheduleError) else "see local diagnostic logs"
        print(f"Pilot stopped: {type(error).__name__}: {detail}", file=sys.stderr)
        sys.exit(1)
