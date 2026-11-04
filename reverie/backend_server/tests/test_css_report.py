"""Known small networks and versioned persistence; no paid API calls."""
import copy
import csv
import itertools
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import css_cumulative_survey as css
import css_report
import measure_css_survey as measurement


@pytest.fixture
def response():
    roster = ["Ada", "Bo", "Cy", "Di"]
    pairs = list(itertools.combinations(roster, 2))
    metadata = {"roster": roster, "survey_version": css.SURVEY_VERSION, "snapshot_digest": "state",
                "step": 3600, "repeat": 1, "initial": False,
                "truth_edges": [["Ada", "Bo", 2], ["Bo", "Cy", 1], ["Cy", "Di", 3]]}
    answer = {"respondent": "Ada", "survey_version": css.SURVEY_VERSION, "snapshot_digest": "state",
              "repeat": 1, "degree": roster, "broker": roster, "community": [roster],
              "interaction": [[a, b, t] for (a, b), t in zip(pairs, [1, 1, 0, 0, 0, 1])],
              "interaction_uncertainty": [[a, b, u] for (a, b), u in zip(
                  pairs, [False, True, False, True, False, False])], "invalid_attempts": {}}
    return metadata, answer


def test_known_binary_errors_and_uncertainty_groups(response):
    metadata, answer = response
    row = css_report.score_respondent(metadata, answer)
    assert [row[k] for k in ("tp", "fp", "fn", "tn")] == [2, 1, 1, 2]
    assert row["classification_error"] == pytest.approx(1 / 3)
    assert row["balanced_accuracy"] == row["f1"] == pytest.approx(2 / 3)
    assert row["jaccard"] == .5 and row["pair_coverage"] == 1
    assert row["uncertain_fraction"] == pytest.approx(1 / 3)
    assert row["uncertain_error_rate"] == 1 and row["confident_error_rate"] == 0
    assert row["perceived_edges"] == 3 and row["density_error"] == 0
    assert "brier" not in row


def test_perfect_binary_network_and_separate_elicited_rankings(response):
    metadata, answer = response
    for row, tie in zip(answer["interaction"], [1, 0, 0, 1, 0, 1]):
        row[2] = tie
    scores = css_report.score_respondent(metadata, answer)
    assert scores["classification_error"] == 0
    for key in ("network_degree_tau_b", "network_broker_tau_b", "network_community_ari", "network_community_nmi"):
        assert scores[key] == pytest.approx(1)
    assert scores["degree_tau_b"] != scores["network_degree_tau_b"]


def test_missing_pairs_have_coverage_but_no_imputed_network(response):
    metadata, answer = response
    answer["interaction"][0][2] = answer["interaction_uncertainty"][0][2] = None
    row = css_report.score_respondent(metadata, answer)
    assert row["pairs_evaluated"] == 5 and row["pair_coverage"] == pytest.approx(5 / 6)
    assert row["classification_error"] == pytest.approx(2 / 5)
    assert row["uncertain_pairs"] == 2 and row["interaction_missing"] == 1
    assert row["perceived_edges"] is row["network_degree_tau_b"] is row["network_community_ari"] is None
    for ties, flags in zip(answer["interaction"], answer["interaction_uncertainty"]):
        ties[2] = flags[2] = None
    row = css_report.score_respondent(metadata, answer)
    assert row["pair_coverage"] == 0 and row["classification_error"] is row["uncertain_fraction"] is None


def test_initial_and_empty_graphs_preserve_undefined_scores(response):
    metadata, answer = response
    metadata["initial"] = True
    row = css_report.score_respondent(metadata, answer)
    assert row["classification_error"] is row["tp"] is row["network_community_ari"] is None
    assert row["uncertain_pairs"] == 2
    metadata.update(initial=False, truth_edges=[])
    for ties in answer["interaction"]:
        ties[2] = 0
    row = css_report.score_respondent(metadata, answer)
    assert row["accuracy"] == 1
    assert row["precision"] is row["recall"] is row["balanced_accuracy"] is row["jaccard"] is None
    assert row["network_degree_tau_b"] is row["network_broker_tau_b"] is None
    assert row["network_community_ari"] == 1


@pytest.mark.parametrize("version", ["recent-network-v1", "cumulative-connection-v1"])
def test_legacy_probabilities_keep_their_original_score(response, version):
    metadata, answer = response
    metadata["survey_version"] = version
    answer.pop("survey_version")
    answer.pop("interaction_uncertainty")
    for item in answer["interaction"]:
        item[2] = .25
    before = copy.deepcopy(answer)
    row = css_report.score_respondent(metadata, answer)
    assert row["brier"] == pytest.approx((.75 ** 2 + .25 ** 2) / 2)
    assert "uncertain_fraction" not in row and answer == before


def save_wave(root, response):
    metadata, answer = response
    directory = root / "surveys/step_003600/repeat_01"
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(json.dumps(metadata))
    for i, name in enumerate(metadata["roster"]):
        (directory / f"respondent_{i:02d}.json").write_text(json.dumps({**answer, "respondent": name}))
    (directory / "complete.json").write_text(json.dumps({"respondents": 4, "repeat": 1,
        "snapshot_digest": "state", "survey_version": css.SURVEY_VERSION}))
    return directory


@pytest.mark.parametrize("change", ["version", "digest", "repeat", "pairs", "flag", "missing", "marker", "metadata"])
def test_resume_rejects_mixed_versions_corruption_and_incomplete_markers(tmp_path, response, change):
    directory = save_wave(tmp_path, response)
    assert len(css.read_survey_results(directory, require_complete=True)) == 4
    path = directory / "respondent_00.json"
    answer = json.loads(path.read_text())
    if change == "version":
        answer["survey_version"] = "cumulative-connection-v1"
    elif change == "digest":
        answer["snapshot_digest"] = "wrong"
    elif change == "repeat":
        answer["repeat"] = 2
    elif change == "pairs":
        answer["interaction_uncertainty"].reverse()
    elif change == "flag":
        answer["interaction_uncertainty"][0][2] = None
    elif change == "missing":
        (directory / "respondent_03.json").unlink()
    elif change == "marker":
        (directory / "complete.json").write_text('{}')
    else:
        (directory / "metadata.json").unlink()
    path.write_text(json.dumps(answer))
    with pytest.raises(ValueError):
        css.read_survey_results(directory, require_complete=True)


def test_parent_does_not_skip_a_completed_wave_from_another_snapshot(tmp_path, response, monkeypatch):
    save_wave(tmp_path, response)
    snapshot = tmp_path / "snapshot.gz"
    snapshot.write_bytes(b"test snapshot")
    monkeypatch.setattr(measurement.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected worker"))
    with pytest.raises(ValueError, match="current snapshot"):
        measurement.measure_snapshot(tmp_path, snapshot, 3600, 1, 42, tmp_path / ".env")


def test_report_exports_binary_and_uncertainty_scores(tmp_path, response):
    directory = save_wave(tmp_path, response)
    before = {p: p.read_bytes() for p in directory.iterdir()}
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "saved_snapshots_measured",
        "checkpoint_status": "no_eligible_checkpoint", "survey_version": css.SURVEY_VERSION}))
    report = css_report.write_report(tmp_path)
    assert report["scoring_version"] == css_report.SCORING_VERSION
    assert "brier" not in {r["metric"] for r in report["measurement_noise"]}
    with (tmp_path / "perception_scores.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 4 and rows[0]["uncertain_error_rate"] == "1.0"
    assert "brier" not in rows[0] and rows[0]["network_degree_tau_b"] != ""
    text = (tmp_path / "REPORT.md").read_text()
    assert css.SURVEY_VERSION in text and "preceding two hours" not in text
    assert all(p.read_bytes() == data for p, data in before.items())
