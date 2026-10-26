"""Changing measurement repeats must not rewrite or migrate the simulation."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import measure_saved_pilot as measurement


def fixture(tmp_path, monkeypatch):
    source, output = tmp_path / "trajectory", tmp_path / "measurements"
    source.mkdir(); output.mkdir()
    (source / "snapshots").mkdir()
    manifest = {"run_name": "trajectory", "config": {"survey_repeats": 2, "seed": 104729,
                "source_digest": "frozen"}, "step": 5040, "checkpoint_status": "searching",
                "snapshots": {"3600": {"path": "snapshots/3600.gz", "labels": ["scheduled"]}}}
    (source / "manifest.json").write_text(json.dumps(manifest))
    (source / "latest.pkl.gz").write_bytes(b"unchanged recovery bytes")
    (source / "snapshots/3600.gz").write_bytes(b"unchanged snapshot bytes")
    monkeypatch.setattr(measurement, "environment_configuration",
                        lambda args, config: {**config, "survey_repeats": args.repeats})
    import pilot_report
    monkeypatch.setattr(pilot_report, "write_report", lambda *a: {"reported": True})
    args = SimpleNamespace(source_run=source, output=output, repeats=3)
    return args, manifest


def test_three_repeats_keep_source_two_repeat_configuration_and_bytes(tmp_path, monkeypatch):
    args, manifest = fixture(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in args.source_run.rglob("*") if p.is_file()}
    calls = []
    def measure(output, snapshot, step, repeats):
        calls.append((output, snapshot, step, repeats))
        for repeat in range(1, repeats+1):
            directory = output / "surveys" / f"step_{step:06d}" / f"repeat_{repeat:02d}"
            directory.mkdir(parents=True)
            (directory / "complete.json").write_text("{}")
    monkeypatch.setattr(measurement.runner, "measure_snapshot", measure)
    assert measurement.run_measurement(args) == {"reported": True}
    assert len(calls) == 1 and calls[0][2:] == (3600, 3)
    assert len(list((args.output / "surveys").glob("step_*/repeat_*/complete.json"))) == 3
    assert all(p.read_bytes() == data for p, data in before.items())
    plan = json.loads((args.output / "measurement_plan.json").read_text())
    assert plan["source_runtime_config"]["survey_repeats"] == 2
    assert plan["measurement_config"]["survey_repeats"] == plan["repeats"] == 3
    assert json.loads((args.output / "source_integrity.json").read_text())["source_files_unchanged"]


def test_new_snapshots_can_be_added_but_old_settings_and_hashes_cannot_change(tmp_path, monkeypatch):
    args, manifest = fixture(tmp_path, monkeypatch)
    config = {**manifest["config"], "survey_repeats": 3}
    old = measurement.measurement_plan(args.source_run, manifest, config, 3)
    new = copy.deepcopy(old)
    new["snapshot_files"]["5040"] = {"path": "snapshots/5040.gz", "sha256": "new"}
    measurement.validate_plan_extension(old, new)
    for mutate in [lambda p: p.update(repeats=2),
                   lambda p: p["snapshot_files"]["3600"].update(sha256="changed"),
                   lambda p: p["snapshot_files"].pop("3600")]:
        changed = copy.deepcopy(new); mutate(changed)
        with pytest.raises(ValueError):
            measurement.validate_plan_extension(old, changed)


def test_failure_preserves_source_and_records_measurement_failure(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path, monkeypatch)
    before = (args.source_run / "manifest.json").read_bytes()
    def fail(*a):
        raise RuntimeError("injected worker failure")
    monkeypatch.setattr(measurement.runner, "measure_snapshot", fail)
    with pytest.raises(RuntimeError, match="injected"):
        measurement.run_measurement(args)
    assert (args.source_run / "manifest.json").read_bytes() == before
    assert json.loads((args.output / "manifest.json").read_text())["status"] == "measurement_failed"


def test_environment_mismatch_and_source_output_overlap_are_rejected(tmp_path, monkeypatch):
    args, manifest = fixture(tmp_path, monkeypatch)
    config = {**manifest["config"], "survey_repeats": 3, "source_digest": "changed"}
    with pytest.raises(ValueError, match="source, Python"):
        measurement.measurement_plan(args.source_run, manifest, config, 3)
    args.output = args.source_run / "inside"
    with pytest.raises(ValueError, match="separate sibling"):
        measurement.run_measurement(args)
