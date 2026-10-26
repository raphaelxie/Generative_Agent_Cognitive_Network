#!/usr/bin/env python3
"""API-free acceptance checks. Read only trusted, locally generated snapshots."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from pilot_protocol import graph_from_chats, graph_summary
from pilot_state import load_snapshot
from pilot_tasks import REACTION_REVISION_VERSION, state_for

PUBLIC_SECTORS = {"Hobbs Cafe", "The Rose and Crown Pub", "Johnson Park", "Oak Hill College",
                  "The Willows Market and Pharmacy", "Harvey Oak Supply Store"}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_audit(contracts, audits, debug):
    """Independently check allocations; do not reuse the implementation under test."""
    attempts = defaultdict(list)
    for record in contracts:
        if record["contract"] == "Schedule revision":
            attempts[record["revision_id"]].append(record)
    require(bool(attempts), "No model schedule revisions were exercised")
    audit_by_id = {r["revision_id"]: r for r in audits}
    require(len(audit_by_id) == len(audits), "Duplicate revision audit IDs")
    for revision_id, records in attempts.items():
        require([r["attempt"] for r in records] == list(range(1, len(records)+1)),
                "Nonconsecutive revision attempts")
        require(len(records) <= 3 and records[-1]["status"] == "accepted"
                and all(r["status"] == "rejected" for r in records[:-1]),
                "Exhausted or unresolved schedule revision")
        require(revision_id in audit_by_id, "Accepted priority has no allocation audit")
        require(all(r["persona"] == audit_by_id[revision_id]["persona"] for r in records),
                "Revision persona attribution mismatch")
    for record in audits:
        source, order, allocated = record["source"], record["priority"], record["allocated_minutes"]
        budget = record["remaining_minutes"]
        require(record["reaction_revision_version"] == REACTION_REVISION_VERSION,
                "Unexpected revision version")
        require(type(budget) is int and budget == max(0, record["window_end"]-record["end"]),
                "Audit budget does not match its interval")
        require(all(type(i) is int for i in order) and sorted(order) == list(range(len(source))),
                "Audit priority is not a complete permutation")
        require(len(allocated) == len(source) and sum(allocated) == budget, "Incorrect allocated total")
        require(all(type(m) is int and 0 <= m <= row["minutes"] for m, row in zip(allocated, source)),
                "Invalid or lengthened source duration")
        remaining = budget
        for i in order:
            expected = min(source[i]["minutes"], remaining)
            require(allocated[i] == expected, "Allocation violates priority policy")
            remaining -= expected
        require(record["suffix"] == [{**row, "minutes": allocated[i]}
                                     for i, row in enumerate(source) if allocated[i]],
                "Suffix changed text, provenance or execution order")
        if record["decision"] == "model_priority":
            require(record["revision_id"] in attempts, "Model priority has no accepted attempt")
        elif record["decision"] == "only_feasible_choice":
            require(len(source) == 1 and budget > 0, "Invalid single-choice shortcut")
        else:
            require(record["decision"] == "no_remaining_interval" and not source and budget == 0,
                    "Invalid empty-suffix shortcut")
    traced = {(r["revision_id"], r["attempt"]): r for r in debug}
    expected_keys = {(key, r["attempt"]) for key, records in attempts.items() for r in records}
    require(set(traced) == expected_keys and len(traced) == len(debug), "Incomplete debug trace")
    for revision_id, records in attempts.items():
        for record in records:
            trace = traced[revision_id, record["attempt"]]
            require(trace["error"] == record["error"] and trace["status"] == record["status"],
                    "Debug validation outcome mismatch")
            for field, text in (("prompt_sha256", trace["prompt"]), ("response_sha256", trace["raw"])):
                require(hashlib.sha256(text.encode()).hexdigest() == record[field], "Debug hash mismatch")
        require(json.loads(traced[revision_id, records[-1]["attempt"]]["raw"])["priority"]
                == audit_by_id[revision_id]["priority"], "Allocated priority differs from model response")
    return {"model_revisions": len(attempts), "allocations": len(audits),
            "rejected_attempts": sum(r["status"] == "rejected" for rs in attempts.values() for r in rs),
            "exhausted_revisions": 0, "audit_valid": True, "debug_trace_complete": True}


def mobility_metrics(output, step, server=None):
    import networkx as nx
    if server is None:
        server = load_snapshot(output/"snapshots"/f"step_{step:06d}.pkl.gz", restore_random=False)["server"]
    require(server.step == step, "Metrics snapshot has the wrong step")
    actual, destinations = defaultdict(set), defaultdict(set)
    # Use every completed movement frame; the action address is only a destination.
    movement = output/"worlds"/server.sim_code/"movement"
    for tick in range(step):
        frame = json.loads((movement/f"{tick}.json").read_text())
        for name, info in frame["persona"].items():
            sector = server.maze.access_tile(tuple(info["movement"]))["sector"]
            if sector in PUBLIC_SECTORS:
                actual[name].add(sector)
            description = info.get("description", "")
            if "@" in description:
                address = description.split("@", 1)[1].strip().split(":")
                if len(address) > 1 and address[1] in PUBLIC_SECTORS:
                    destinations[name].add(address[1])
    graph = graph_from_chats(server.personas, server.start_time, server.curr_time)
    summary = graph_summary(graph)
    return {"step": step, "actual_public_visitors": len(actual),
            "public_destination_agents": len(destinations),
            "public_sectors_reached": sorted(set().union(*actual.values())),
            "destination_without_arrival": sorted(set(destinations)-set(actual)),
            "edges": summary["edges"], "largest_component": max(summary["components"]),
            "nonzero_betweenness": sum(v > 0 for v in nx.betweenness_centrality(
                graph, normalized=True, weight=None).values())}


def validate_stage(output, baseline, step):
    from run_midnight_pilot import source_digest, verify_progress
    manifest = json.loads((output/"manifest.json").read_text())
    config = manifest["config"]
    require(config["source_digest"] == source_digest(), "Runtime source changed during smoke")
    require(config.get("reaction_revision_version") == REACTION_REVISION_VERSION, "Wrong revision config")
    verified = verify_progress(output, config)
    require(verified["step"] == step and verified["snapshot_valid"] and verified["manifest_matches"],
            "Stage snapshot or manifest verification failed")
    require(verified["status"] == "paused" and verified["pause"]["reason"] == "stop_at_step",
            "Stage did not pause at its declared boundary")
    server = load_snapshot(output/"latest.pkl.gz", restore_random=False)["server"]
    for persona in server.personas.values():
        state_for(persona.scratch)
    audit = validate_audit(read_jsonl(output/"schedule_diagnostics.jsonl"),
                           read_jsonl(output/"schedule_revision_audit.jsonl"),
                           read_jsonl(output/"schedule_revision_debug.jsonl"))
    metrics = mobility_metrics(output, step, server)
    if step == 5040:
        require(metrics["actual_public_visitors"] > 0 and metrics["edges"] > 3
                and metrics["nonzero_betweenness"] > 0, "Mobility/network functional check failed")
    return {"passed": True, "step": step, "verification": verified, "revision_checks": audit,
            "metrics": metrics, "baseline_metrics": mobility_metrics(baseline, step)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--step", required=True, type=int, choices=(3600, 5040))
    args = parser.parse_args()
    print(json.dumps(validate_stage(args.output.resolve(), args.baseline.resolve(), args.step), indent=2))


if __name__ == "__main__":
    main()
