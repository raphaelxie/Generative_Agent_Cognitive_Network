"""
Perception survey runner for network cognition research.

Runs three instruments on a live ReverieServer without disturbing agent state:
  1. Micro-tie survey  (CSN batch: does j know k?)
  2. Centrality ranking (NCN: rank everyone by connectedness)
  3. Self-position      (NCN: rate own connectedness 1-5)

Snapshots and restores last_accessed on all memory nodes so retrieval
side-effects are fully reversed after the survey completes.

Also writes matching ground-truth files via ground_truth_log and a
per-wave retrieval-diagnostics JSONL (see `_retrieve_with_diagnostics`).
"""
import csv
import json
import os
import sys

_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from persona.cognitive_modules.retrieve import (
    extract_recency,
    extract_importance,
    extract_relevance,
    normalize_dict_floats,
    top_highest_x_values,
)
from persona.prompt_template.run_gpt_prompt import (
    run_gpt_prompt_csn_connection_batch,
    run_gpt_prompt_ncn_centrality_rank,
    run_gpt_prompt_ncn_self_position,
)
from ground_truth_log import write_ground_truth_csv

DT_FMT = "%Y-%m-%d %H:%M:%S"

SURVEY_COLUMNS = [
    "wave_id", "step", "sim_time",
    "respondent", "question_type", "target_j", "target", "value", "is_fail_safe",
]


# ── snapshot / restore ──────────────────────────────────────────────────

def _snapshot_last_accessed(personas):
    snap = {}
    for name, p in personas.items():
        snap[name] = {nid: n.last_accessed
                      for nid, n in p.a_mem.id_to_node.items()}
    return snap


def _restore_last_accessed(personas, snap):
    for name, p in personas.items():
        for nid, ts in snap[name].items():
            if nid in p.a_mem.id_to_node:
                p.a_mem.id_to_node[nid].last_accessed = ts


# ── retrieval helper + diagnostics wrapper (Option B) ───────────────────
#
# SYNC POINT: This wrapper replicates the scoring logic from
#   reverie/backend_server/persona/cognitive_modules/retrieve.py
#   function new_retrieve (see lines around `gw = [0.5, 3, 2]`).
# It imports the same extract_* helpers so the math cannot drift, but
# intentionally does NOT mutate node.last_accessed (belt-and-suspenders;
# the caller already snapshots/restores last_accessed around the survey).
# If retrieve.py's weights or formulas change, update this wrapper in
# lockstep.

def _retrieve_with_diagnostics(persona, focal, query_target_person,
                                n_count=50):
    """Retrieve top-k memory nodes and return (statements_str, diagnostics).

    diagnostics is a dict with keys:
        focal_query, query_target_person, n_candidates, n_returned, nodes
    where each node is a dict with scores and mention flags.
    """
    respondent_name = getattr(persona.scratch, "name", "") or ""
    id_to_node = persona.a_mem.id_to_node

    candidate_nodes = [n for n in (persona.a_mem.seq_event
                                   + persona.a_mem.seq_thought)
                       if "idle" not in n.embedding_key]
    candidate_nodes = sorted(candidate_nodes, key=lambda n: n.last_accessed)
    n_candidates = len(candidate_nodes)

    empty_diag = {
        "focal_query":          focal,
        "query_target_person":  query_target_person,
        "n_candidates":         n_candidates,
        "n_returned":           0,
        "nodes":                [],
    }

    if n_candidates == 0:
        return "(No specific memories found.)", empty_diag

    recency_raw    = extract_recency(persona, candidate_nodes)
    importance_raw = extract_importance(persona, candidate_nodes)
    relevance_raw  = extract_relevance(persona, candidate_nodes, focal)

    recency_out    = normalize_dict_floats(dict(recency_raw), 0, 1)
    importance_out = normalize_dict_floats(dict(importance_raw), 0, 1)
    relevance_out  = normalize_dict_floats(dict(relevance_raw), 0, 1)

    gw = [0.5, 3, 2]  # sync point with retrieve.new_retrieve
    master_out = {}
    for key in recency_out.keys():
        master_out[key] = (
            persona.scratch.recency_w    * recency_out[key]    * gw[0]
            + persona.scratch.relevance_w  * relevance_out[key]  * gw[1]
            + persona.scratch.importance_w * importance_out[key] * gw[2]
        )

    top = top_highest_x_values(master_out, n_count)
    top_ids = list(top.keys())

    qt_lower = query_target_person.lower() if query_target_person else None
    self_lower = respondent_name.lower() if respondent_name else None

    statements_parts = []
    diag_nodes = []
    for nid in top_ids:
        node = id_to_node[nid]
        ek = node.embedding_key or ""
        subj = getattr(node, "subject", "") or ""
        pred = getattr(node, "predicate", "") or ""
        obj = getattr(node, "object", "") or ""

        statements_parts.append(ek)

        text_pool = f"{ek} {subj} {obj}".lower()
        mentions_qt = (qt_lower in text_pool) if qt_lower else None
        is_self = (self_lower in text_pool) if self_lower else False

        diag_nodes.append({
            "node_id":               nid,
            "node_type":             getattr(node, "type", None),
            "subject":               subj,
            "predicate":             pred,
            "object":                obj,
            "embedding_key_trunc":   ek[:200],
            "recency_score":         recency_out.get(nid),
            "relevance_score":       relevance_out.get(nid),
            "importance_score":      importance_out.get(nid),
            "final_score":           master_out.get(nid),
            "mentions_query_target": mentions_qt,
            "is_self_mention":       is_self,
        })

    statements = "\n".join(statements_parts) + "\n"
    if not statements.strip():
        statements = "(No specific memories found.)"

    return statements, {
        "focal_query":          focal,
        "query_target_person":  query_target_person,
        "n_candidates":         n_candidates,
        "n_returned":           len(top_ids),
        "nodes":                diag_nodes,
    }


# ── main survey runner ──────────────────────────────────────────────────

def run_perception_survey(personas, sim_code, step, curr_time,
                          output_dir, wave_id="",
                          recent_window_minutes=480):
    """
    Run all three perception instruments on a live ReverieServer.

    Snapshots and restores last_accessed so the survey is non-invasive.
    Writes:
      - perception_survey_{wave_id}.csv
      - perception_survey_{wave_id}_meta.json
      - retrieval_diagnostics_{wave_id}.jsonl
      - ground_truth_chats_{step}.csv
      - ground_truth_edges_{step}.csv
    Returns path to the survey CSV.
    """
    sim_time_str = curr_time.strftime(DT_FMT)
    persona_names = sorted(personas.keys())
    n = len(persona_names)

    # snapshot before any retrieval calls
    snap = _snapshot_last_accessed(personas)

    # retrieval diagnostics sink
    os.makedirs(output_dir, exist_ok=True)
    diag_path = os.path.join(
        output_dir, f"retrieval_diagnostics_{wave_id}.jsonl")
    n_retrieval_calls = 0

    def _write_diag(diag_file, respondent, diag):
        entry = {
            "wave_id":    wave_id,
            "step":       step,
            "sim_time":   sim_time_str,
            "respondent": respondent,
        }
        entry.update(diag)
        diag_file.write(json.dumps(entry, ensure_ascii=False,
                                   default=str) + "\n")

    rows = []
    try:
        with open(diag_path, "w", encoding="utf-8") as diag_file:
            for idx, respondent in enumerate(persona_names):
                persona = personas[respondent]
                other_names = [nm for nm in persona_names if nm != respondent]
                print(f"  [{wave_id}] Respondent {idx+1}/{n}: {respondent}")

                # ── 1. micro-tie (reuses CSN batch pattern exactly) ─────
                for j in persona_names:
                    list_k = [k for k in persona_names if k != j]
                    focal = f"connection or relationship involving {j}"
                    statements, diag = _retrieve_with_diagnostics(
                        persona, focal, query_target_person=j)
                    _write_diag(diag_file, respondent, diag)
                    n_retrieval_calls += 1

                    values, meta = run_gpt_prompt_csn_connection_batch(
                        persona, statements, j, list_k
                    )
                    is_fs = "1" if meta[0] is None else "0"
                    if len(values) != len(list_k):
                        values = [None] * len(list_k)
                        is_fs = "1"
                    for k, val in zip(list_k, values):
                        rows.append({
                            "wave_id":       wave_id,
                            "step":          step,
                            "sim_time":      sim_time_str,
                            "respondent":    respondent,
                            "question_type": "micro_tie",
                            "target_j":      j,
                            "target":        k,
                            "value":         "" if val is None else str(val),
                            "is_fail_safe":  is_fs,
                        })

                # ── 2. centrality ranking + 3. self-position ────────────
                focal_social = ("social connections and interactions "
                                "in the community")
                social_statements, social_diag = _retrieve_with_diagnostics(
                    persona, focal_social, query_target_person=None)
                _write_diag(diag_file, respondent, social_diag)
                n_retrieval_calls += 1

                rank_list, rank_meta = run_gpt_prompt_ncn_centrality_rank(
                    persona, social_statements, persona_names
                )
                rank_fs = "1" if rank_meta[0] is None else "0"
                for rank_pos, target_name in enumerate(rank_list, start=1):
                    rows.append({
                        "wave_id":       wave_id,
                        "step":          step,
                        "sim_time":      sim_time_str,
                        "respondent":    respondent,
                        "question_type": "centrality_rank",
                        "target_j":      "",
                        "target":        target_name,
                        "value":         str(rank_pos),
                        "is_fail_safe":  rank_fs,
                    })

                pos_val, pos_meta = run_gpt_prompt_ncn_self_position(
                    persona, social_statements, other_names
                )
                pos_fs = "1" if pos_meta[0] is None else "0"
                rows.append({
                    "wave_id":       wave_id,
                    "step":          step,
                    "sim_time":      sim_time_str,
                    "respondent":    respondent,
                    "question_type": "self_position",
                    "target_j":      "",
                    "target":        "",
                    "value":         str(pos_val),
                    "is_fail_safe":  pos_fs,
                })

    finally:
        _restore_last_accessed(personas, snap)

    # ── write survey CSV ────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    survey_path = os.path.join(output_dir, f"perception_survey_{wave_id}.csv")
    with open(survey_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SURVEY_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    # ── write matching ground-truth files ───────────────────────────────
    gt_dir = os.path.join(output_dir, "ground_truth")
    write_ground_truth_csv(personas, sim_code, step, curr_time,
                           gt_dir, recent_window_minutes=recent_window_minutes,
                           wave_id=wave_id)

    n_micro = sum(1 for r in rows if r["question_type"] == "micro_tie")
    n_rank  = sum(1 for r in rows if r["question_type"] == "centrality_rank")
    n_self  = sum(1 for r in rows if r["question_type"] == "self_position")
    n_fs    = sum(1 for r in rows if r["is_fail_safe"] == "1")
    print(f"  Survey complete: {len(rows)} rows "
          f"(micro_tie={n_micro}, centrality_rank={n_rank}, "
          f"self_position={n_self}, fail_safe={n_fs})")
    print(f"  Wrote {survey_path}")
    print(f"  Wrote {diag_path} ({n_retrieval_calls} retrieval calls)")

    # ── write per-wave metadata ──────────────────────────────────────────
    meta_path = os.path.join(output_dir, f"perception_survey_{wave_id}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "wave_id":                   wave_id,
            "step":                      step,
            "sim_time":                  sim_time_str,
            "sim_code":                  sim_code,
            "n_personas":                n,
            "persona_names":             persona_names,
            "n_rows":                    len(rows),
            "n_micro_tie":               n_micro,
            "n_centrality_rank":         n_rank,
            "n_self_position":           n_self,
            "n_fail_safe":               n_fs,
            "n_retrieval_calls":         n_retrieval_calls,
            "retrieval_diagnostics_path": os.path.basename(diag_path),
        }, f, indent=2)

    return survey_path
