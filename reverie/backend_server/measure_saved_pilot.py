#!/usr/bin/env python3
"""Measure trusted pilot snapshots independently, without changing trajectory configuration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from pilot_state import atomic_json
import run_midnight_pilot as runner

SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[2]
MEASUREMENT_VERSION = "separate-snapshot-measurement-v1"


def file_digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def environment_configuration(args, source_config):
    # configure() otherwise re-execs the simulation entrypoint, losing this CLI.
    import utils
    from dotenv import load_dotenv
    load_dotenv(args.env_file, override=True)
    args.seed = source_config["seed"]
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.execv(sys.executable, [sys.executable, str(SCRIPT), *sys.argv[1:]])
    runner.configure(args)
    return runner.configuration(args)


def measurement_plan(source_dir, manifest, config, repeats):
    expected = {**manifest["config"], "survey_repeats": repeats}
    if config != expected:
        raise ValueError("Measurement must retain the trajectory's source, Python, dependencies, seed and model")
    snapshots = {}
    for step, entry in manifest["snapshots"].items():
        if not set(entry["labels"]) & {"initial", "scheduled", "checkpoint", "final"}:
            continue
        path = (source_dir / entry["path"]).resolve()
        if not path.is_relative_to(source_dir):
            raise ValueError("Snapshot path escapes the source run")
        snapshots[step] = {"path": str(path.relative_to(source_dir)), "sha256": file_digest(path)}
    if not snapshots:
        raise ValueError("No saved measurement snapshots")
    return {"measurement_version": MEASUREMENT_VERSION, "runner_sha256": file_digest(SCRIPT),
            "source_run_name": manifest["run_name"], "source_runtime_config": manifest["config"],
            "measurement_config": config, "repeats": repeats, "snapshot_files": snapshots}


def validate_plan_extension(old, new):
    if {k: v for k, v in old.items() if k != "snapshot_files"} != {
            k: v for k, v in new.items() if k != "snapshot_files"}:
        raise ValueError("Existing measurement settings changed; use a new measurement output directory")
    for step, entry in old["snapshot_files"].items():
        if new["snapshot_files"].get(step) != entry:
            raise ValueError("A previously measured source snapshot changed or disappeared")


def run_measurement(args):
    source_dir, output = args.source_run.resolve(), args.output.resolve()
    if output == source_dir or output.is_relative_to(source_dir) or source_dir.is_relative_to(output):
        raise ValueError("Use a separate sibling measurement output; never write inside the source run")
    if args.repeats < 2:
        raise ValueError("At least two independent repeats are required")
    original = json.loads((source_dir / "manifest.json").read_text())
    config = environment_configuration(args, original["config"])
    # Serialize on the small-memory host; the snapshot workers also run one at a time.
    with runner.run_lock(source_dir), runner.run_lock(output):
        original = json.loads((source_dir / "manifest.json").read_text())
        plan = measurement_plan(source_dir, original, config, args.repeats)
        plan_path = output / "measurement_plan.json"
        if plan_path.exists():
            validate_plan_extension(json.loads(plan_path.read_text()), plan)
        elif (output / "manifest.json").exists() or (output / "surveys").exists():
            raise ValueError("Output already contains data without an independent measurement plan")
        protected = {source_dir / "manifest.json", source_dir / "latest.pkl.gz"}
        protected.update(source_dir / row["path"] for row in plan["snapshot_files"].values())
        before = {path: file_digest(path) for path in protected}
        atomic_json(plan_path, plan)
        view = {"kind": "measurement_only", "run_name": output.name,
                "status": "measuring_saved_snapshots", "config": config,
                "source_runtime_config": original["config"], "source_run_name": original["run_name"],
                "source_step": original["step"], "checkpoint_status": original["checkpoint_status"],
                "selected_checkpoint": original.get("selected_checkpoint")}
        atomic_json(output / "manifest.json", view)
        try:
            for step, row in sorted(plan["snapshot_files"].items(), key=lambda item: int(item[0])):
                runner.measure_snapshot(output, source_dir / row["path"], int(step), args.repeats)
            view["status"] = "saved_snapshots_measured"
        except BaseException:
            view["status"] = "measurement_failed"
            raise
        finally:
            unchanged = all(file_digest(path) == digest for path, digest in before.items())
            if not unchanged:
                view["status"] = "source_integrity_failed"
            atomic_json(output / "manifest.json", view)
            atomic_json(output / "source_integrity.json", {"source_files_unchanged": unchanged,
                        "files": {str(path.relative_to(source_dir)): digest for path, digest in before.items()}})
            if not unchanged:
                raise RuntimeError("Measurement changed a protected source file")
        from pilot_report import write_report
        return write_report(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--env-file", type=Path, default=PROJECT / ".env")
    args = parser.parse_args()
    args.source_run, args.output, args.env_file = (p.resolve() for p in
                                                (args.source_run, args.output, args.env_file))
    print(json.dumps(run_measurement(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
