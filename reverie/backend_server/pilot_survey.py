"""Four timestamped instruments run only inside a disposable snapshot worker."""
import itertools
import json
import math
from datetime import timedelta
from pathlib import Path

from pilot_protocol import RECENT_MINUTES, graph_from_chats
from pilot_state import atomic_json, load_snapshot

SURVEY_VERSION = "recent-network-v1"


def parse_answer(text, instrument, roster, count=None):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    answer = json.loads(text)
    if instrument == "interaction":
        values = answer["p"]
        if not isinstance(values, list) or len(values) != count or any(
                isinstance(p, bool) or not isinstance(p, (int, float))
                or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
            raise ValueError("Expected one probability per unordered pair")
        return values
    if instrument in ("degree", "broker"):
        ranking = answer["ranking"]
        if not isinstance(ranking, list) or len(ranking) != len(roster) or set(ranking) != set(roster):
            raise ValueError("Ranking must contain every roster member exactly once")
        return ranking
    groups = answer["groups"]
    if not isinstance(groups, list) or not groups or any(not isinstance(g, list) or not g for g in groups):
        raise ValueError("Communities must be a nonempty partition")
    flat = [n for g in groups for n in g]
    if len(flat) != len(roster) or set(flat) != set(roster):
        raise ValueError("Communities must partition the complete roster")
    return groups


def ask(prompt, instrument, roster, count=None):
    from persona.prompt_template.gpt_structure import _chat
    attempts = []
    for _ in range(3):
        raw = _chat(prompt, max_tokens=4096)
        try:
            return parse_answer(raw, instrument, roster, count), attempts
        except (ValueError, TypeError, KeyError) as error:
            attempts.append({"error": type(error).__name__, "response": raw})
    # Missing data is explicit. Never invent a ranking or a zero probability.
    return None, attempts


def survey_worker(snapshot, output, repeat):
    from perception_survey import _retrieve_with_diagnostics
    from pilot_api import set_context

    payload = load_snapshot(snapshot)
    server, protocol = payload["server"], payload["protocol"]
    roster = sorted(server.personas)
    now, start = server.curr_time, server.start_time
    cutoff = max(start, now - timedelta(minutes=RECENT_MINUTES))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = protocol["config"]
    graph = graph_from_chats(server.personas, start, now, RECENT_MINUTES)
    metadata = {"survey_version": SURVEY_VERSION, "model": config["model"],
                "reasoning_effort": "none", "repeat": repeat,
                "step": server.step, "sim_time": now.isoformat(),
                "window_start": cutoff.isoformat(), "window_end_exclusive": now.isoformat(),
                "recent_minutes": RECENT_MINUTES, "initial": server.step == 0,
                "snapshot_digest": payload["state_digest"], "roster": roster,
                "truth_edges": [[a, b, d["weight"]] for a, b, d in graph.edges(data=True)]}
    atomic_json(output / "metadata.json", metadata)
    pairs = list(itertools.combinations(roster, 2))
    for index, name in enumerate(roster):
        result_path = output / f"respondent_{index:02d}.json"
        if result_path.exists():
            old = json.loads(result_path.read_text())
            if old.get("snapshot_digest") != payload["state_digest"]:
                raise ValueError("Existing survey belongs to another snapshot")
            continue
        set_context("survey", now, server.step, repeat=repeat, respondent=name)
        persona = server.personas[name]
        statements, diagnostics = _retrieve_with_diagnostics(
            persona, f"Direct conversations and community interactions between {cutoff} and {now}",
            None, include_timestamps=True)
        biography = {key: getattr(persona.scratch, key, None)
                     for key in ("name", "age", "innate", "learned", "currently", "lifestyle")}
        context = (f"You are {name}. Your biography: {json.dumps(biography, ensure_ascii=False)}\n"
                   f"Your available memories:\n{statements}\n"
                   f"Roster: {json.dumps(roster)}\n"
                   f"Assess direct conversations STARTED in [{cutoff.isoformat()}, {now.isoformat()}). "
                   "For this questionnaire, an undirected edge exists if two people had at least one "
                   "direct conversation in that interval. Earlier relationships are not such edges. "
                   "A thought's recorded timestamp is when it was formed, not necessarily when "
                   "the event it describes occurred. Infer from your own knowledge; the actual network "
                   "is not provided. Return only the requested JSON.\n")
        probabilities, failures = [], {}
        for offset in range(0, len(pairs), 50):
            batch = pairs[offset:offset+50]
            prompt = context + ('For each listed pair, give your probability (0 to 1) of an edge, '
                                'in the same order. Format: {"p": [0.5, ...]}. Pairs: ' + json.dumps(batch))
            answer, errors = ask(prompt, "interaction", roster, len(batch))
            probabilities.extend(answer if answer is not None else [None]*len(batch))
            if errors:
                failures[f"interaction_{offset}"] = errors
        answers = {}
        tasks = {
            "degree": 'Rank everyone by number of distinct direct conversation partners in the interval, '
                      'highest first. Include everyone once. Format: {"ranking": ["name", ...]}.',
            "broker": 'Rank everyone by their bridge role on shortest paths between other people '
                      'in this interval\'s undirected, unweighted conversation network (betweenness), '
                      'highest first. Include everyone once. Format: {"ranking": ["name", ...]}.',
            "community": 'Partition everyone into communities of people who conversed more frequently '
                         'with one another in the interval. Include isolates as singleton groups where '
                         'appropriate. Format: {"groups": [["name", ...], ...]}.'}
        for kind, question in tasks.items():
            answers[kind], errors = ask(context + question, kind, roster)
            if errors:
                failures[kind] = errors
        atomic_json(result_path, {"respondent": name, "snapshot_digest": payload["state_digest"],
                    "interaction": [[a, b, p] for (a, b), p in zip(pairs, probabilities)],
                    **answers, "invalid_attempts": failures, "memory_context": statements,
                    "retrieval_diagnostics": diagnostics})
    # Only this marker means all respondents finished. Source snapshot is never saved by worker.
    atomic_json(output / "complete.json", {"snapshot_digest": payload["state_digest"],
                                          "respondents": len(roster), "repeat": repeat})
