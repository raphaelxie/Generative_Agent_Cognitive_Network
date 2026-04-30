"""
Per-persona planner freeze audit for a completed simulation snapshot.

Usage:
  python planner_freeze_audit.py <survey_dir>
"""
import csv
import json
import os
import re
import sys
from collections import Counter


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_chats(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _is_prompt_leak(desc):
    if not isinstance(desc, str):
        return False
    s = desc.strip()
    return s.startswith("Here's the completed hourly schedule") or s.startswith("Here’s the completed hourly schedule")


def _discover_latest_chats_file(gt_dir):
    pat = re.compile(r"^ground_truth_chats_(\d+)\.csv$")
    best = None
    for name in os.listdir(gt_dir):
        m = pat.match(name)
        if not m:
            continue
        step = int(m.group(1))
        if best is None or step > best[0]:
            best = (step, os.path.join(gt_dir, name))
    if best is None:
        raise FileNotFoundError(f"No ground_truth_chats_*.csv found in {gt_dir}")
    return best


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python planner_freeze_audit.py <survey_dir>")

    survey_dir = os.path.abspath(sys.argv[1])
    sim_dir = os.path.dirname(survey_dir)
    personas_dir = os.path.join(sim_dir, "personas")
    gt_dir = os.path.join(survey_dir, "ground_truth")
    latest_step, chats_path = _discover_latest_chats_file(gt_dir)
    chat_rows = _load_chats(chats_path)

    chat_counter = Counter()
    for row in chat_rows:
        chat_counter[row["node_a"]] += 1
        chat_counter[row["node_b"]] += 1

    out_rows = []
    for persona in sorted(os.listdir(personas_dir)):
        scratch_path = os.path.join(personas_dir, persona, "bootstrap_memory", "scratch.json")
        if not os.path.isfile(scratch_path):
            continue
        s = _load_json(scratch_path)

        daily_req = s.get("daily_req", []) or []
        schedule = s.get("f_daily_schedule", []) or []
        n_schedule_slots = len(schedule)
        n_prompt_leak_slots = 0
        n_zero_duration_sleep_slots = 0
        n_normal_slots = 0
        last_legit_slot_index = -1

        for idx, slot in enumerate(schedule):
            if not isinstance(slot, list) or len(slot) < 2:
                continue
            desc = slot[0] if isinstance(slot[0], str) else ""
            dur = slot[1] if isinstance(slot[1], (int, float)) else 0
            prompt_leak = _is_prompt_leak(desc)
            zero_sleep = (desc.strip() == "sleeping" and int(dur) == 0)
            if prompt_leak:
                n_prompt_leak_slots += 1
            if zero_sleep:
                n_zero_duration_sleep_slots += 1
            if (not prompt_leak) and int(dur) > 0:
                n_normal_slots += 1
                last_legit_slot_index = idx

        act_event = s.get("act_event", [])
        act_event_at_save = act_event[1] if isinstance(act_event, list) and len(act_event) >= 2 else ""
        chatting_at_save = bool(s.get("chatting_with"))

        out_rows.append({
            "persona": persona,
            "latest_chats_step": latest_step,
            "n_daily_req": len(daily_req),
            "n_schedule_slots": n_schedule_slots,
            "n_normal_slots": n_normal_slots,
            "n_zero_duration_sleep_slots": n_zero_duration_sleep_slots,
            "n_prompt_leak_slots": n_prompt_leak_slots,
            "last_legit_slot_index": last_legit_slot_index,
            "chats_initiated_cumulative": chat_counter.get(persona, 0),
            "act_event_at_save": act_event_at_save,
            "chatting_at_save": int(chatting_at_save),
        })

    out_path = os.path.join(survey_dir, "planner_freeze_audit.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
