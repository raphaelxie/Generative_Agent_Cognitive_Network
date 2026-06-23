#!/usr/bin/env python3
"""
Quick verification: fork base_the_ville_n25 and run 200 steps (~33 min game time).
Confirms agents wake up and are not all sleeping at 08:00+ start time.

Usage (from reverie/backend_server/):
  python verify_awake_n25.py
"""
import json
import os
import sys
import threading
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

FORK_FROM = "base_the_ville_n25"
SIM_CODE = "verify_awake_n25"
STEPS = 200
FS_STORAGE = "../../environment/frontend_server/storage"


class HeadlessFrontend:
    def __init__(self, sim_folder):
        self.sim_folder = sim_folder
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self):
        last_step_processed = -1
        while not self._stop.is_set():
            move_dir = os.path.join(self.sim_folder, "movement")
            env_dir = os.path.join(self.sim_folder, "environment")
            if not os.path.isdir(move_dir):
                time.sleep(0.05)
                continue
            move_files = []
            for fn in os.listdir(move_dir):
                if fn.endswith(".json"):
                    try:
                        step = int(fn.replace(".json", ""))
                        move_files.append(step)
                    except ValueError:
                        pass
            for step in sorted(move_files):
                if step <= last_step_processed:
                    continue
                next_env = os.path.join(env_dir, f"{step + 1}.json")
                if os.path.exists(next_env):
                    last_step_processed = step
                    continue
                move_file = os.path.join(move_dir, f"{step}.json")
                try:
                    with open(move_file) as f:
                        data = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    continue
                persona_data = data.get("persona", {})
                env_out = {}
                for name, info in persona_data.items():
                    mv = info.get("movement", [0, 0])
                    env_out[name] = {"x": mv[0], "y": mv[1]}
                if not env_out:
                    continue
                os.makedirs(env_dir, exist_ok=True)
                with open(next_env, "w") as f:
                    json.dump(env_out, f, indent=2)
                last_step_processed = step
            time.sleep(0.02)


def main():
    import shutil
    from reverie import ReverieServer

    sim_folder = os.path.abspath(f"{FS_STORAGE}/{SIM_CODE}")

    if os.path.exists(sim_folder):
        print(f"Removing old verification sim: {sim_folder}")
        shutil.rmtree(sim_folder)

    print(f"Forking: {FORK_FROM} -> {SIM_CODE}")
    rs = ReverieServer(FORK_FROM, SIM_CODE)
    print(f"  Start time: {rs.curr_time}")
    print(f"  Step: {rs.step}, Personas: {len(rs.personas)}")

    frontend = HeadlessFrontend(sim_folder)
    frontend.start()

    try:
        print(f"\nRunning {STEPS} steps...")
        rs.start_server(STEPS)
        print(f"  -> Now at step {rs.step}, time {rs.curr_time}")
    finally:
        frontend.stop()

    move_dir = os.path.join(sim_folder, "movement")
    sleeping_steps = 0
    awake_steps = 0
    chat_steps = 0
    for step in range(STEPS):
        try:
            with open(os.path.join(move_dir, f"{step}.json")) as f:
                mov = json.load(f)
            sl = sum(1 for info in mov["persona"].values()
                     if "sleeping" in info.get("description", "").lower())
            ch = sum(1 for info in mov["persona"].values()
                     if info.get("chat") is not None)
            if sl == len(mov["persona"]):
                sleeping_steps += 1
            else:
                awake_steps += 1
            if ch > 0:
                chat_steps += 1
        except Exception:
            pass

    print(f"\n{'=' * 50}")
    print(f"  All-sleeping steps: {sleeping_steps}")
    print(f"  Steps with awake agents: {awake_steps}")
    print(f"  Steps with chats: {chat_steps}")
    print(f"{'=' * 50}")

    if awake_steps > 0:
        print("  SUCCESS: Agents are waking up!")
    else:
        print("  FAIL: All agents still sleeping after 200 steps")

    print(f"\nCleaning up verification sim...")
    shutil.rmtree(sim_folder)
    print("Done.")


if __name__ == "__main__":
    main()
