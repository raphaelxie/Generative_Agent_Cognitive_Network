"""Binary perceived connections and separate uncertainty, with no recent-time window.

Runs only inside a disposable snapshot worker. The recent-window instrument in
pilot_survey.py stays unchanged because it is part of the simulation source digest.
"""
import hashlib
import itertools
import json
from pathlib import Path

from pilot_protocol import graph_from_chats
from pilot_state import atomic_json, load_snapshot
from pilot_survey import ask, parse_answer

SURVEY_VERSION = "cumulative-tie-uncertainty-v2"
TRUTH_DEFINITION = "distinct conversation starts in [simulation_start, snapshot_time), all 25 nodes"
RETRIEVAL_QUERY = "Direct conversations and community interactions with other people"
PAIR_BATCH = 50

QUESTIONS = {
    "degree": 'Rank everyone by the number of distinct people they are connected to, highest first. '
              'Include everyone once. Format: {"ranking": ["name", ...]}.',
    "broker": 'Rank everyone by their bridge role on shortest paths between other people in the '
              'undirected, unweighted network of these connections (betweenness), highest first. '
              'Include everyone once. Format: {"ranking": ["name", ...]}.',
    "community": 'Partition everyone into communities of people who are more densely connected with '
                 'one another. Include isolates as singleton groups where appropriate. '
                 'Format: {"groups": [["name", ...], ...]}.',
}


def truth_graph(server):
    return graph_from_chats(server.personas, server.start_time, server.curr_time)


def survey_context(name, biography, statements, roster):
    return (f"You are {name}. Your biography: {json.dumps(biography, ensure_ascii=False)}\n"
            f"Your available memories:\n{statements}\n"
            f"Roster: {json.dumps(roster)}\n"
            "For this questionnaire, answer about the network of connections among the people on "
            "the roster. Infer from your own knowledge; the actual network is not provided. "
            "Return only the requested JSON.\n")


def pair_prompt(context, batch):
    return context + ('For each listed pair, give your best judgment of whether the two people '
                      'have a connection with each other. Set tie to 1 if you judge that they '
                      'have a connection, or 0 if you judge that they do not. Even if uncertain, '
                      'choose your best judgment of 0 or 1. Separately, set uncertain to true '
                      'if you are unsure of that judgment, or false if you are confident in it. '
                      'An uncertain judgment can be either 0 or 1. These are judgments, not '
                      'probabilities or relationship strengths. Return two arrays in the same '
                      'order as the listed pairs, with exactly one entry per pair in each array. '
                      'Use only integers 0 or 1 for tie and JSON booleans true or false for '
                      'uncertain. Format: {"tie": [1, 0], "uncertain": [true, false]}. '
                      'Pairs: ' + json.dumps(batch))


def parse_pair_answer(text, count):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    answer = json.loads(text)
    ties, uncertain = answer["tie"], answer["uncertain"]
    if not isinstance(ties, list) or len(ties) != count or any(
            type(t) is not int or t not in (0, 1) for t in ties):
        raise ValueError("Expected one integer tie (0 or 1) per pair")
    if not isinstance(uncertain, list) or len(uncertain) != count or any(
            type(u) is not bool for u in uncertain):
        raise ValueError("Expected one boolean uncertainty flag per pair")
    return ties, uncertain


def ask_pairs(prompt, count):
    from persona.prompt_template.gpt_structure import _chat
    attempts = []
    for _ in range(3):
        raw = _chat(prompt, max_tokens=4096)
        try:
            return parse_pair_answer(raw, count), attempts
        except (ValueError, TypeError, KeyError, IndexError) as error:
            attempts.append({"error": type(error).__name__, "response": raw})
    # Failure is missing data in both fields, never a negative or an uncertainty flag.
    return None, attempts


def validate_pair_rows(answer, roster):
    pairs = list(itertools.combinations(roster, 2))
    ties, uncertain = answer["interaction"], answer["interaction_uncertainty"]
    for rows in (ties, uncertain):
        if not isinstance(rows, list) or len(rows) != len(pairs) or any(
                not isinstance(row, list) or len(row) != 3 or tuple(row[:2]) != pair
                for row, pair in zip(rows, pairs)):
            raise ValueError("Stored pairs must match the complete ordered roster combinations")
    for (_, _, tie), (_, _, flag) in zip(ties, uncertain):
        if tie is None and flag is None:
            continue
        if type(tie) is not int or tie not in (0, 1) or type(flag) is not bool:
            raise ValueError("Expected a binary tie and boolean flag, or two missing values")


def read_survey_results(output, metadata=None, require_complete=False):
    """Validate a v2 wave before resuming or scoring; never reinterpret old answers."""
    output = Path(output)
    meta_path = output / "metadata.json"
    files = sorted(output.glob("respondent_*.json"))
    marker = output / "complete.json"
    if not meta_path.exists():
        if files or marker.exists() or require_complete:
            raise ValueError("Survey output is missing its metadata")
        return {}
    saved = json.loads(meta_path.read_text())
    if saved.get("survey_version") != SURVEY_VERSION or (metadata is not None and saved != metadata):
        raise ValueError("Survey version or snapshot settings changed; use a new output directory")
    expected = {f"respondent_{i:02d}.json": name for i, name in enumerate(saved["roster"])}
    answers = {}
    for file in files:
        answer = json.loads(file.read_text())
        if file.name not in expected or answer.get("respondent") != expected[file.name] or any(
                answer.get(key) != saved[key] for key in ("survey_version", "snapshot_digest", "repeat")):
            raise ValueError("Stored respondent belongs to a different survey, snapshot or repeat")
        validate_pair_rows(answer, saved["roster"])
        for kind, key in (("degree", "ranking"), ("broker", "ranking"), ("community", "groups")):
            if answer[kind] is not None:
                parse_answer(json.dumps({key: answer[kind]}), kind, saved["roster"])
        answers[file.name] = answer
    if marker.exists():
        complete = json.loads(marker.read_text())
        if any(complete.get(key) != saved[key] for key in ("survey_version", "snapshot_digest", "repeat")):
            raise ValueError("Completion marker belongs to a different survey")
        if complete.get("respondents") != len(expected) or set(answers) != set(expected):
            raise ValueError("Completion marker is missing respondent files")
    elif require_complete:
        raise ValueError("Survey has no completion marker")
    return answers


def survey_worker(snapshot, output, repeat):
    from perception_survey import _retrieve_with_diagnostics
    from pilot_api import set_context

    payload = load_snapshot(snapshot)
    server, protocol = payload["server"], payload["protocol"]
    roster = sorted(server.personas)
    now, start = server.curr_time, server.start_time
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    graph = truth_graph(server)
    metadata = {"survey_version": SURVEY_VERSION, "model": protocol["config"]["model"],
                "instrument_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "snapshot_sha256": hashlib.sha256(Path(snapshot).read_bytes()).hexdigest(),
                "reasoning_effort": "none", "repeat": repeat,
                "step": server.step, "sim_time": now.isoformat(),
                "window_start": start.isoformat(), "window_end_exclusive": now.isoformat(),
                "recent_minutes": None, "truth_definition": TRUTH_DEFINITION,
                "initial": server.step == 0, "snapshot_digest": payload["state_digest"],
                "roster": roster,
                "truth_edges": [[a, b, d["weight"]] for a, b, d in graph.edges(data=True)]}
    existing = read_survey_results(output, metadata)
    atomic_json(output / "metadata.json", metadata)
    pairs = list(itertools.combinations(roster, 2))
    for index, name in enumerate(roster):
        result_path = output / f"respondent_{index:02d}.json"
        if result_path.name in existing:
            continue
        set_context("survey", now, server.step, repeat=repeat, respondent=name)
        persona = server.personas[name]
        statements, diagnostics = _retrieve_with_diagnostics(
            persona, RETRIEVAL_QUERY, None, include_timestamps=True)
        biography = {key: getattr(persona.scratch, key, None)
                     for key in ("name", "age", "innate", "learned", "currently", "lifestyle")}
        context = survey_context(name, biography, statements, roster)
        ties, uncertainty, failures = [], [], {}
        for offset in range(0, len(pairs), PAIR_BATCH):
            batch = pairs[offset:offset + PAIR_BATCH]
            answer, errors = ask_pairs(pair_prompt(context, batch), len(batch))
            batch_ties, batch_uncertainty = answer if answer is not None else (
                [None] * len(batch), [None] * len(batch))
            ties.extend(batch_ties)
            uncertainty.extend(batch_uncertainty)
            if errors:
                failures[f"interaction_{offset}"] = errors
        answers = {}
        for kind, question in QUESTIONS.items():
            answers[kind], errors = ask(context + question, kind, roster)
            if errors:
                failures[kind] = errors
        atomic_json(result_path, {"respondent": name, "snapshot_digest": payload["state_digest"],
                    "survey_version": SURVEY_VERSION, "repeat": repeat,
                    "interaction": [[a, b, tie] for (a, b), tie in zip(pairs, ties)],
                    "interaction_uncertainty": [[a, b, flag] for (a, b), flag in zip(pairs, uncertainty)],
                    **answers, "invalid_attempts": failures, "memory_context": statements,
                    "retrieval_diagnostics": diagnostics})
    # Only this marker means all respondents finished. Source snapshot is never saved by worker.
    atomic_json(output / "complete.json", {"snapshot_digest": payload["state_digest"],
                                          "respondents": len(roster), "repeat": repeat,
                                          "survey_version": SURVEY_VERSION})
