"""
Cognitive Social Network (CSN) Survey for generative agents.
After running a simulation (e.g. base_the_ville_n15 or a fork), run this script
to ask each persona about every ordered pair (j, k): "Does j have a connection with k?"
Output: CSV with columns respondent, person_j, person_k, connection (0, 1, or empty for null).

Usage (from project root):
  python reverie/backend_server/run_csn_survey.py <sim_code>
  python reverie/backend_server/run_csn_survey.py base_the_ville_n15
  python reverie/backend_server/run_csn_survey.py Mar3_1try_15_agent_run

Or from reverie/backend_server:
  python run_csn_survey.py <sim_code>
"""
import os
import sys
import csv

# Ensure we can import reverie, utils, persona from backend_server
_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
os.chdir(_backend_dir)

from reverie import ReverieServer
from utils import fs_storage
from persona.cognitive_modules.retrieve import new_retrieve
from persona.prompt_template.run_gpt_prompt import run_gpt_prompt_csn_connection_batch


def run_csn_survey(sim_code, output_path=None):
    """
    Load simulation, ask each persona about all (j,k) pairs, write CSV.
    Uses batch mode: one API call per (respondent, j) for all k != j.
    """
    if output_path is None:
        output_path = os.path.join(fs_storage, sim_code, "cognitive_social_network.csv")

    print(f"Loading simulation: {sim_code}")
    rs = ReverieServer(sim_code, sim_code)
    persona_names = list(rs.personas.keys())
    n = len(persona_names)

    rows = []
    for idx_respondent, respondent in enumerate(persona_names):
        persona = rs.personas[respondent]
        print(f"Respondent {idx_respondent + 1}/{n}: {respondent}")

        for j in persona_names:
            list_k = [k for k in persona_names if k != j]
            focal = f"connection or relationship involving {j}"
            retrieved = new_retrieve(persona, [focal], 50)
            nodes = retrieved.get(focal, [])
            statements = ""
            for node in nodes:
                statements += f"{node.embedding_key}\n"
            if not statements.strip():
                statements = "(No specific memories found.)"

            values, _ = run_gpt_prompt_csn_connection_batch(
                persona, statements, j, list_k
            )
            if len(values) != len(list_k):
                values = [None] * len(list_k)
            for k, val in zip(list_k, values):
                connection = "" if val is None else str(val)
                rows.append((respondent, j, k, connection))

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["respondent", "person_j", "person_k", "connection"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_csn_survey.py <sim_code> [output_csv_path]")
        print("Example: python run_csn_survey.py base_the_ville_n15")
        sys.exit(1)
    sim_code = sys.argv[1].strip()
    output_path = sys.argv[2].strip() if len(sys.argv) > 2 and sys.argv[2].strip() else None
    run_csn_survey(sim_code, output_path)
