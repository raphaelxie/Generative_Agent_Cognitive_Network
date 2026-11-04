#!/usr/bin/env python3
"""Measure trusted pilot snapshots with the cumulative CSS questionnaire (no recent window).

Writes a separate sibling output; the source trajectory and its earlier
recent-window measurements are never rewritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from pilot_state import atomic_json
import css_cumulative_survey as css
import css_report
from measure_saved_pilot import file_digest, validate_plan_extension
import run_midnight_pilot as runner

SCRIPT = Path(__file__).resolve()
BACKEND = SCRIPT.parent
PROJECT = SCRIPT.parents[2]
INSTRUMENT = BACKEND / "css_cumulative_survey.py"
MEASUREMENT_VERSION = "css-cumulative-measurement-v2"
MEASURED_LABELS = {"initial", "scheduled", "checkpoint", "final"}


def environment_configuration(args, source_config):
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
        if not set(entry["labels"]) & MEASURED_LABELS:
            continue
        path = (source_dir / entry["path"]).resolve()
        if not path.is_relative_to(source_dir):
            raise ValueError("Snapshot path escapes the source run")
        snapshots[step] = {"path": str(path.relative_to(source_dir)), "sha256": file_digest(path)}
    if not snapshots:
        raise ValueError("No saved measurement snapshots")
    return {"measurement_version": MEASUREMENT_VERSION, "survey_version": css.SURVEY_VERSION,
            "truth_definition": css.TRUTH_DEFINITION,
            "runner_sha256": file_digest(SCRIPT), "instrument_sha256": file_digest(INSTRUMENT),
            "scoring_version": css_report.SCORING_VERSION,
            "scorer_sha256": file_digest(Path(css_report.__file__)),
            "source_run_name": manifest["run_name"], "source_runtime_config": manifest["config"],
            "measurement_config": {**config, "survey_version": css.SURVEY_VERSION,
                                   "recent_minutes": None},
            "repeats": repeats, "snapshot_files": snapshots}


def measure_snapshot(output, path, step, repeats, seed, env_file):
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    for repeat in range(1, repeats + 1):
        destination = output / "surveys" / f"step_{step:06d}" / f"repeat_{repeat:02d}"
        if (destination / "complete.json").exists():
            css.read_survey_results(destination, require_complete=True)
            metadata = json.loads((destination / "metadata.json").read_text())
            if any(metadata.get(key) != value for key, value in {
                    "step": step, "repeat": repeat, "snapshot_sha256": source_hash,
                    "instrument_sha256": file_digest(INSTRUMENT)}.items()):
                raise ValueError("Completed survey does not match the current snapshot or instrument")
            continue
        command = [sys.executable, str(SCRIPT), "_worker", "--output", str(output),
                   "--snapshot", str(path), "--survey-output", str(destination),
                   "--repeat", str(repeat), "--seed", str(seed), "--env-file", str(env_file)]
        subprocess.run(command, check=True, cwd=BACKEND, env=os.environ.copy())
        if hashlib.sha256(path.read_bytes()).hexdigest() != source_hash:
            raise RuntimeError("Survey changed its source snapshot")
    atomic_json(output / "surveys" / f"step_{step:06d}" / "isolation_check.json",
                {"source_unchanged": True, "live_state_unchanged": None,
                 "snapshot_sha256": source_hash, "survey_version": css.SURVEY_VERSION})


def run_measurement(args):
    source_dir, output = args.source_run.resolve(), args.output.resolve()
    if output == source_dir or output.is_relative_to(source_dir) or source_dir.is_relative_to(output):
        raise ValueError("Use a separate sibling measurement output; never write inside the source run")
    if args.repeats < 2:
        raise ValueError("At least two independent repeats are required")
    original = json.loads((source_dir / "manifest.json").read_text())
    config = environment_configuration(args, original["config"])
    output.mkdir(parents=True, exist_ok=True)
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
                "status": "measuring_saved_snapshots", "config": plan["measurement_config"],
                "survey_version": css.SURVEY_VERSION,
                "source_runtime_config": original["config"], "source_run_name": original["run_name"],
                "source_step": original["step"], "checkpoint_status": original["checkpoint_status"],
                "selected_checkpoint": original.get("selected_checkpoint")}
        atomic_json(output / "manifest.json", view)
        try:
            for step, row in sorted(plan["snapshot_files"].items(), key=lambda item: int(item[0])):
                measure_snapshot(output, source_dir / row["path"], int(step), args.repeats,
                                 args.seed, args.env_file)
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
        return css_report.write_report(output)


def worker(argv):
    parser = argparse.ArgumentParser(description="One disposable cumulative CSS survey worker")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--survey-output", required=True, type=Path)
    parser.add_argument("--repeat", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--env-file", required=True, type=Path)
    args = parser.parse_args(argv)
    # runner.configure would re-exec run_midnight_pilot.py and lose this worker CLI.
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        raise RuntimeError("PYTHONHASHSEED must be set by the parent measurement process")
    runner.configure(SimpleNamespace(output=args.output.resolve(), env_file=args.env_file.resolve(),
                                     seed=args.seed))
    css.survey_worker(args.snapshot.resolve(), args.survey_output.resolve(), args.repeat)


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
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        worker(sys.argv[2:])
    else:
        main()
