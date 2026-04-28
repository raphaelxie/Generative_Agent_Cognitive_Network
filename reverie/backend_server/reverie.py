"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: reverie.py
Description: This is the main program for running generative agent simulations
that defines the ReverieServer class. This class maintains and records all  
states related to the simulation. The primary mode of interaction for those  
running the simulation should be through the open_server function, which  
enables the simulator to input command-line prompts for running and saving  
the simulation, among other tasks.

Release note (June 14, 2023) -- Reverie implements the core simulation 
mechanism described in my paper entitled "Generative Agents: Interactive 
Simulacra of Human Behavior." If you are reading through these lines after 
having read the paper, you might notice that I use older terms to describe 
generative agents and their cognitive modules here. Most notably, I use the 
term "personas" to refer to generative agents, "associative memory" to refer 
to the memory stream, and "reverie" to refer to the overarching simulation 
framework.
"""
import json
import numpy
import datetime
import pickle
import time
import math
import os
import shutil
import traceback

from selenium import webdriver

from global_methods import *
from utils import *
from maze import *
from persona.persona import *

##############################################################################
#                                  REVERIE                                   #
##############################################################################
import os, random
try:
    import numpy as np
except Exception:
    np = None

def set_global_seed():
    seed_str = os.environ.get("GA_SEED", "")
    if not seed_str:
        return
    seed = int(seed_str)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)

set_global_seed()

class ReverieServer: 
  def __init__(self, 
               fork_sim_code,
               sim_code):
    # FORKING FROM A PRIOR SIMULATION:
    # <fork_sim_code> indicates the simulation we are forking from. 
    # Interestingly, all simulations must be forked from some initial 
    # simulation, where the first simulation is "hand-crafted".
    self.fork_sim_code = fork_sim_code
    fork_folder = f"{fs_storage}/{self.fork_sim_code}"

    # <sim_code> indicates our current simulation. The first step here is to 
    # copy everything that's in <fork_sim_code>, but edit its 
    # reverie/meta/json's fork variable. When fork_sim_code == sim_code we
    # load the existing simulation without copying (for post-simulation analysis).
    self.sim_code = sim_code
    sim_folder = f"{fs_storage}/{self.sim_code}"
    if fork_sim_code != sim_code:
      copyanything(fork_folder, sim_folder)

    with open(f"{sim_folder}/reverie/meta.json") as json_file:  
      reverie_meta = json.load(json_file)

    if fork_sim_code != sim_code:
      with open(f"{sim_folder}/reverie/meta.json", "w") as outfile: 
        reverie_meta["fork_sim_code"] = fork_sim_code
        outfile.write(json.dumps(reverie_meta, indent=2))

    # LOADING REVERIE'S GLOBAL VARIABLES
    # The start datetime of the Reverie: 
    # <start_datetime> is the datetime instance for the start datetime of 
    # the Reverie instance. Once it is set, this is not really meant to 
    # change. It takes a string date in the following example form: 
    # "June 25, 2022"
    # e.g., ...strptime(June 25, 2022, "%B %d, %Y")
    self.start_time = datetime.datetime.strptime(
                        f"{reverie_meta['start_date']}, 00:00:00",  
                        "%B %d, %Y, %H:%M:%S")
    # <curr_time> is the datetime instance that indicates the game's current
    # time. This gets incremented by <sec_per_step> amount everytime the world
    # progresses (that is, everytime curr_env_file is recieved). 
    self.curr_time = datetime.datetime.strptime(reverie_meta['curr_time'], 
                                                "%B %d, %Y, %H:%M:%S")
    # <sec_per_step> denotes the number of seconds in game time that each 
    # step moves foward. 
    self.sec_per_step = reverie_meta['sec_per_step']
    
    # <maze> is the main Maze instance. Note that we pass in the maze_name
    # (e.g., "double_studio") to instantiate Maze. 
    # e.g., Maze("double_studio")
    self.maze = Maze(reverie_meta['maze_name'])
    
    # <step> denotes the number of steps that our game has taken. A step here
    # literally translates to the number of moves our personas made in terms
    # of the number of tiles. 
    self.step = reverie_meta['step']

    # SETTING UP PERSONAS IN REVERIE
    # <personas> is a dictionary that takes the persona's full name as its 
    # keys, and the actual persona instance as its values.
    # This dictionary is meant to keep track of all personas who are part of
    # the Reverie instance. 
    # e.g., ["Isabella Rodriguez"] = Persona("Isabella Rodriguezs")
    self.personas = dict()
    # <personas_tile> is a dictionary that contains the tile location of
    # the personas (!-> NOT px tile, but the actual tile coordinate).
    # The tile take the form of a set, (row, col). 
    # e.g., ["Isabella Rodriguez"] = (58, 39)
    self.personas_tile = dict()
    
    # # <persona_convo_match> is a dictionary that describes which of the two
    # # personas are talking to each other. It takes a key of a persona's full
    # # name, and value of another persona's full name who is talking to the 
    # # original persona. 
    # # e.g., dict["Isabella Rodriguez"] = ["Maria Lopez"]
    # self.persona_convo_match = dict()
    # # <persona_convo> contains the actual content of the conversations. It
    # # takes as keys, a pair of persona names, and val of a string convo. 
    # # Note that the key pairs are *ordered alphabetically*. 
    # # e.g., dict[("Adam Abraham", "Zane Xu")] = "Adam: baba \n Zane:..."
    # self.persona_convo = dict()

    # Loading in all personas. 
    init_env_file = f"{sim_folder}/environment/{str(self.step)}.json"
    init_env = json.load(open(init_env_file))
    for persona_name in reverie_meta['persona_names']: 
      persona_folder = f"{sim_folder}/personas/{persona_name}"
      p_x = init_env[persona_name]["x"]
      p_y = init_env[persona_name]["y"]
      curr_persona = Persona(persona_name, persona_folder)

      self.personas[persona_name] = curr_persona
      self.personas_tile[persona_name] = (p_x, p_y)
      self.maze.tiles[p_y][p_x]["events"].add(curr_persona.scratch
                                              .get_curr_event_and_desc())

    # REVERIE SETTINGS PARAMETERS:  
    # <server_sleep> denotes the amount of time that our while loop rests each
    # cycle; this is to not kill our machine. 
    self.server_sleep = 0.1

    # SIGNALING THE FRONTEND SERVER: 
    # curr_sim_code.json contains the current simulation code, and
    # curr_step.json contains the current step of the simulation. These are 
    # used to communicate the code and step information to the frontend. 
    # Note that step file is removed as soon as the frontend opens up the 
    # simulation. 
    curr_sim_code = dict()
    curr_sim_code["sim_code"] = self.sim_code
    with open(f"{fs_temp_storage}/curr_sim_code.json", "w") as outfile: 
      outfile.write(json.dumps(curr_sim_code, indent=2))
    
    curr_step = dict()
    curr_step["step"] = self.step
    with open(f"{fs_temp_storage}/curr_step.json", "w") as outfile: 
      outfile.write(json.dumps(curr_step, indent=2))


  def save(self): 
    """
    Save all Reverie progress -- this includes Reverie's global state as well
    as all the personas.  

    INPUT
      None
    OUTPUT 
      None
      * Saves all relevant data to the designated memory directory
    """
    # <sim_folder> points to the current simulation folder.
    sim_folder = f"{fs_storage}/{self.sim_code}"

    # Save Reverie meta information.
    reverie_meta = dict() 
    reverie_meta["fork_sim_code"] = self.fork_sim_code
    reverie_meta["start_date"] = self.start_time.strftime("%B %d, %Y")
    reverie_meta["curr_time"] = self.curr_time.strftime("%B %d, %Y, %H:%M:%S")
    reverie_meta["sec_per_step"] = self.sec_per_step
    reverie_meta["maze_name"] = self.maze.maze_name
    reverie_meta["persona_names"] = list(self.personas.keys())
    reverie_meta["step"] = self.step
    reverie_meta_f = f"{sim_folder}/reverie/meta.json"
    with open(reverie_meta_f, "w") as outfile: 
      outfile.write(json.dumps(reverie_meta, indent=2))

    # Save the personas.
    for persona_name, persona in self.personas.items(): 
      save_folder = f"{sim_folder}/personas/{persona_name}/bootstrap_memory"
      persona.save(save_folder)


  def start_path_tester_server(self): 
    """
    Starts the path tester server. This is for generating the spatial memory
    that we need for bootstrapping a persona's state. 

    To use this, you need to open server and enter the path tester mode, and
    open the front-end side of the browser. 

    INPUT 
      None
    OUTPUT 
      None
      * Saves the spatial memory of the test agent to the path_tester_env.json
        of the temp storage. 
    """
    def print_tree(tree): 
      def _print_tree(tree, depth):
        dash = " >" * depth

        if type(tree) == type(list()): 
          if tree:
            print (dash, tree)
          return 

        for key, val in tree.items(): 
          if key: 
            print (dash, key)
          _print_tree(val, depth+1)
      
      _print_tree(tree, 0)

    # <curr_vision> is the vision radius of the test agent. Recommend 8 as 
    # our default. 
    curr_vision = 8
    # <s_mem> is our test spatial memory. 
    s_mem = dict()

    # The main while loop for the test agent. 
    while (True): 
      try: 
        curr_dict = {}
        tester_file = fs_temp_storage + "/path_tester_env.json"
        if check_if_file_exists(tester_file): 
          with open(tester_file) as json_file: 
            curr_dict = json.load(json_file)
            os.remove(tester_file)
          
          # Current camera location
          curr_sts = self.maze.sq_tile_size
          curr_camera = (int(math.ceil(curr_dict["x"]/curr_sts)), 
                         int(math.ceil(curr_dict["y"]/curr_sts))+1)
          curr_tile_det = self.maze.access_tile(curr_camera)

          # Initiating the s_mem
          world = curr_tile_det["world"]
          if curr_tile_det["world"] not in s_mem: 
            s_mem[world] = dict()

          # Iterating throughn the nearby tiles.
          nearby_tiles = self.maze.get_nearby_tiles(curr_camera, curr_vision)
          for i in nearby_tiles: 
            i_det = self.maze.access_tile(i)
            if (curr_tile_det["sector"] == i_det["sector"] 
                and curr_tile_det["arena"] == i_det["arena"]): 
              if i_det["sector"] != "": 
                if i_det["sector"] not in s_mem[world]: 
                  s_mem[world][i_det["sector"]] = dict()
              if i_det["arena"] != "": 
                if i_det["arena"] not in s_mem[world][i_det["sector"]]: 
                  s_mem[world][i_det["sector"]][i_det["arena"]] = list()
              if i_det["game_object"] != "": 
                if (i_det["game_object"] 
                    not in s_mem[world][i_det["sector"]][i_det["arena"]]):
                  s_mem[world][i_det["sector"]][i_det["arena"]] += [
                                                         i_det["game_object"]]

        # Incrementally outputting the s_mem and saving the json file. 
        print ("= " * 15)
        out_file = fs_temp_storage + "/path_tester_out.json"
        with open(out_file, "w") as outfile: 
          outfile.write(json.dumps(s_mem, indent=2))
        print_tree(s_mem)

      except:
        pass

      time.sleep(self.server_sleep * 10)


  def start_server(self, int_counter): 
    """
    The main backend server of Reverie. 
    This function retrieves the environment file from the frontend to 
    understand the state of the world, calls on each personas to make 
    decisions based on the world state, and saves their moves at certain step
    intervals. 
    INPUT
      int_counter: Integer value for the number of steps left for us to take
                   in this iteration. 
    OUTPUT 
      None
    """
    # <sim_folder> points to the current simulation folder.
    sim_folder = f"{fs_storage}/{self.sim_code}"
    # #region agent log
    _dbg_logged_waiting_env = set()
    _dbg_log_path = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", ".cursor", "debug-2483ef.log"))

    def _dbg2483ef(message, data, hypothesis_id):
      try:
        os.makedirs(os.path.dirname(_dbg_log_path), exist_ok=True)
        payload = {
          "sessionId": "2483ef",
          "runId": "frontend-handoff",
          "hypothesisId": hypothesis_id,
          "location": "reverie.py:start_server",
          "message": message,
          "data": data,
          "timestamp": int(time.time() * 1000),
        }
        with open(_dbg_log_path, "a", encoding="utf-8") as _df:
          _df.write(json.dumps(payload, default=str) + "\n")
      except Exception:
        pass
    # #endregion

    # When a persona arrives at a game object, we give a unique event
    # to that object. 
    # e.g., ('double studio[...]:bed', 'is', 'unmade', 'unmade')
    # Later on, before this cycle ends, we need to return that to its 
    # initial state, like this: 
    # e.g., ('double studio[...]:bed', None, None, None)
    # So we need to keep track of which event we added. 
    # <game_obj_cleanup> is used for that. 
    game_obj_cleanup = dict()

    # The main while loop of Reverie. 
    while (True): 
      # Done with this iteration if <int_counter> reaches 0. 
      if int_counter == 0: 
        break
      # Reset per outer iteration so a failed read cannot leave a stale True.
      env_retrieved = False
      new_env = None

      # <curr_env_file> file is the file that our frontend outputs. When the
      # frontend has done its job and moved the personas, then it will put a 
      # new environment file that matches our step count. That's when we run 
      # the content of this for loop. Otherwise, we just wait. 
      curr_env_file = f"{sim_folder}/environment/{self.step}.json"
      if check_if_file_exists(curr_env_file):
        # If we have an environment file, it means we have a new perception
        # input to our personas. So we first retrieve it.
        try: 
          # Try and save block for robustness of the while loop.
          with open(curr_env_file) as json_file:
            new_env = json.load(json_file)
            env_retrieved = True
        except: 
          pass
        
        if env_retrieved: 
          # This is where we go through <game_obj_cleanup> to clean up all 
          # object actions that were used in this cylce. 
          for key, val in game_obj_cleanup.items(): 
            # We turn all object actions to their blank form (with None). 
            self.maze.turn_event_from_tile_idle(key, val)
          # Then we initialize game_obj_cleanup for this cycle. 
          game_obj_cleanup = dict()

          # We first move our personas in the backend environment to match 
          # the frontend environment. 
          for persona_name, persona in self.personas.items(): 
            # <curr_tile> is the tile that the persona was at previously. 
            curr_tile = self.personas_tile[persona_name]
            # <new_tile> is the tile that the persona will move to right now,
            # during this cycle. 
            new_tile = (new_env[persona_name]["x"], 
                        new_env[persona_name]["y"])

            # We actually move the persona on the backend tile map here. 
            self.personas_tile[persona_name] = new_tile
            self.maze.remove_subject_events_from_tile(persona.name, curr_tile)
            self.maze.add_event_from_tile(persona.scratch
                                         .get_curr_event_and_desc(), new_tile)

            # Now, the persona will travel to get to their destination. *Once*
            # the persona gets there, we activate the object action.
            if not persona.scratch.planned_path: 
              # We add that new object action event to the backend tile map. 
              # At its creation, it is stored in the persona's backend. 
              game_obj_cleanup[persona.scratch
                               .get_curr_obj_event_and_desc()] = new_tile
              self.maze.add_event_from_tile(persona.scratch
                                     .get_curr_obj_event_and_desc(), new_tile)
              # We also need to remove the temporary blank action for the 
              # object that is currently taking the action. 
              blank = (persona.scratch.get_curr_obj_event_and_desc()[0], 
                       None, None, None)
              self.maze.remove_event_from_tile(blank, new_tile)

          # Then we need to actually have each of the personas perceive and
          # move. The movement for each of the personas comes in the form of
          # x y coordinates where the persona will move towards. e.g., (50, 34)
          # This is where the core brains of the personas are invoked. 
          movements = {"persona": dict(), 
                       "meta": dict()}
          for persona_name, persona in self.personas.items(): 
            # <next_tile> is a x,y coordinate. e.g., (58, 9)
            # <pronunciatio> is an emoji. e.g., "\ud83d\udca4"
            # <description> is a string description of the movement. e.g., 
            #   writing her next novel (editing her novel) 
            #   @ double studio:double studio:common room:sofa
            next_tile, pronunciatio, description = persona.move(
              self.maze, self.personas, self.personas_tile[persona_name], 
              self.curr_time)
            movements["persona"][persona_name] = {}
            movements["persona"][persona_name]["movement"] = next_tile
            movements["persona"][persona_name]["pronunciatio"] = pronunciatio
            movements["persona"][persona_name]["description"] = description
            movements["persona"][persona_name]["chat"] = (persona
                                                          .scratch.chat)

          # Include the meta information about the current stage in the 
          # movements dictionary. 
          movements["meta"]["curr_time"] = (self.curr_time 
                                             .strftime("%B %d, %Y, %H:%M:%S"))

          # We then write the personas' movements to a file that will be sent 
          # to the frontend server. 
          # Example json output: 
          # {"persona": {"Maria Lopez": {"movement": [58, 9]}},
          #  "persona": {"Klaus Mueller": {"movement": [38, 12]}}, 
          #  "meta": {curr_time: <datetime>}}
          curr_move_file = f"{sim_folder}/movement/{self.step}.json"
          _dbg_move_parent = os.path.dirname(curr_move_file)
          os.makedirs(_dbg_move_parent, exist_ok=True)
          with open(curr_move_file, "w") as outfile: 
            outfile.write(json.dumps(movements, indent=2))
          # #region agent log
          _dbg2483ef("movement_written", {
            "step": self.step,
            "curr_move_file": curr_move_file,
            "next_env_expected": f"{sim_folder}/environment/{self.step + 1}.json",
          }, "HANDOFF-A")
          # #endregion

          # After this cycle, the world takes one step forward, and the 
          # current time moves by <sec_per_step> amount. 
          self.step += 1
          self.curr_time += datetime.timedelta(seconds=self.sec_per_step)

          int_counter -= 1
      else:
        # #region agent log
        if self.step not in _dbg_logged_waiting_env:
          _dbg_logged_waiting_env.add(self.step)
          _dbg2483ef("waiting_for_environment_file", {
            "step": self.step,
            "curr_env_file": curr_env_file,
            "movement_prev_exists": check_if_file_exists(
                f"{sim_folder}/movement/{self.step - 1}.json"),
          }, "HANDOFF-B")
        # #endregion
          
      # Sleep so we don't burn our machines. 
      time.sleep(self.server_sleep)

  def _try_shock_isolate_command(self, sim_command, sim_folder):
    """
    Handle shock isolate / shock isolate-hub / shock isolate-broker.
    Returns message string. Prints and returns early message on hard errors
    (unknown agent, no broker, no hub) matching open_server behavior.
    """
    from collections import defaultdict as _dd
    from ground_truth_log import (
        build_ground_truth,
        highest_degree_agent,
        agent_betweenness,
        highest_betweenness_agent,
    )

    ret_str = ""
    cmd_lower = sim_command.lower()
    if cmd_lower.startswith("shock isolate-broker"):
      variant = "broker"
      prefix = "shock isolate-broker"
      command_field = "shock_isolate_broker"
      treatment_type = "broker_removal"
    elif cmd_lower.startswith("shock isolate-hub"):
      variant = "hub"
      prefix = "shock isolate-hub"
      command_field = "shock_isolate_hub"
      treatment_type = "hub_removal"
    else:
      variant = "legacy"
      prefix = "shock isolate"
      command_field = "shock_isolate"
      treatment_type = "hub_removal"

    agent_name = sim_command[len(prefix):].strip() or None

    prev = [n for n, p in self.personas.items() if p.isolated]
    for n in prev:
      self.personas[n].isolated = False
    if prev:
      ret_str += f"Cleared previous isolation: {', '.join(prev)}\n"

    if agent_name and agent_name not in self.personas:
      ret_str += (f"Error: '{agent_name}' not found. "
                  f"Valid names: {sorted(self.personas.keys())}")
      print(ret_str)
      return ""

    _, edge_rows = build_ground_truth(
        self.personas, self.sim_code, self.step, self.curr_time)

    best_hub_name, best_hub_deg = highest_degree_agent(edge_rows)
    best_broker_name, best_broker_bc = highest_betweenness_agent(edge_rows)

    all_degrees = _dd(int)
    for row in edge_rows:
      if row["count_cumulative"] > 0:
        all_degrees[row["node_a"]] += 1
        all_degrees[row["node_b"]] += 1
    all_degrees = dict(all_degrees)
    all_betweenness = agent_betweenness(edge_rows)

    if agent_name:
      target = agent_name
      selection_criterion = "manual"
    elif variant == "broker":
      if best_broker_name is None:
        ret_str += ("No broker structure in current graph -- "
                    "cannot auto-select broker (all betweenness=0).")
        print(ret_str)
        return ""
      target = best_broker_name
      selection_criterion = "betweenness"
    else:
      if best_hub_name is None:
        ret_str += "No interactions yet -- cannot auto-select target."
        print(ret_str)
        return ""
      target = best_hub_name
      selection_criterion = "degree"

    target_deg = all_degrees.get(target, 0)
    target_bc = all_betweenness.get(target, 0.0)

    self.personas[target].isolated = True
    ret_str += (f"Isolated: {target}  "
                f"(treatment={treatment_type}, "
                f"criterion={selection_criterion}, "
                f"degree={target_deg}, "
                f"betweenness={target_bc:.4f}, "
                f"step={self.step})")
    if (variant in ("broker", "hub")
        and best_hub_name and best_broker_name):
      if best_hub_name == best_broker_name:
        ret_str += (f"\n  (note: highest-degree and highest-"
                    f"betweenness agent coincide: {best_hub_name})")
      else:
        ret_str += (f"\n  (highest-degree={best_hub_name} deg={best_hub_deg}; "
                    f"highest-betweenness={best_broker_name} "
                    f"bc={best_broker_bc:.4f})")
    elif agent_name and best_hub_name:
      ret_str += (f"\n  (highest-degree agent is {best_hub_name} "
                  f"with degree={best_hub_deg})")

    shock_dir = f"{sim_folder}/survey"
    os.makedirs(shock_dir, exist_ok=True)
    with open(f"{shock_dir}/shock_log.jsonl", "a") as _sf:
      _sf.write(json.dumps({
          "command":              command_field,
          "step":                 self.step,
          "sim_time":             self.curr_time.strftime("%B %d, %Y, %H:%M:%S"),
          "target_agent":         target,
          "target_degree":        target_deg,
          "auto_selected":        agent_name is None,
          "highest_degree_agent": best_hub_name,
          "highest_degree":       best_hub_deg,
          "treatment_type":            treatment_type,
          "selection_criterion":       selection_criterion,
          "target_betweenness":        target_bc,
          "highest_betweenness_agent": best_broker_name,
          "highest_betweenness":       best_broker_bc,
          "all_degrees":               all_degrees,
          "all_betweenness":           all_betweenness,
      }) + "\n")

    return ret_str

  def _try_unshock_command(self, sim_folder):
    prev = [n for n, p in self.personas.items() if p.isolated]
    for n in prev:
      self.personas[n].isolated = False
    if prev:
      ret_str = f"Un-isolated: {', '.join(prev)}"
      shock_dir = f"{sim_folder}/survey"
      os.makedirs(shock_dir, exist_ok=True)
      with open(f"{shock_dir}/shock_log.jsonl", "a") as _sf:
        _sf.write(json.dumps({
            "command": "unshock",
            "step": self.step,
            "sim_time": self.curr_time.strftime("%B %d, %Y, %H:%M:%S"),
            "agents_unshocked": prev,
            "treatment_type": "unshock",
            "selection_criterion": None,
        }) + "\n")
      return ret_str
    return "No agent was isolated."

  def run_preflight_measurement_batch(self, wave_id="pre"):
    """
    After burn-in: run perception survey + hub/broker shock probe (no post wave).
    Use with sim_code preflight_the_ville_n15-1 (or any loaded Reverie state).
    """
    sim_folder = f"{fs_storage}/{self.sim_code}"
    from perception_survey import run_perception_survey
    out_dir = f"{sim_folder}/survey"
    survey_path = run_perception_survey(
        self.personas, self.sim_code, self.step, self.curr_time,
        out_dir, wave_id=wave_id)
    print(f"Survey complete: {survey_path}")
    for cmd in ("shock isolate-hub", "unshock",
                "shock isolate-broker", "unshock"):
      msg = (self._try_shock_isolate_command(cmd, sim_folder)
             if cmd.startswith("shock")
             else self._try_unshock_command(sim_folder))
      if msg:
        print(msg)

  def open_server(self): 
    """
    Open up an interactive terminal prompt that lets you run the simulation 
    step by step and probe agent state. 

    INPUT 
      None
    OUTPUT
      None
    """
    print ("Note: The agents in this simulation package are computational")
    print ("constructs powered by generative agents architecture and LLM. We")
    print ("clarify that these agents lack human-like agency, consciousness,")
    print ("and independent decision-making.\n---")

    # <sim_folder> points to the current simulation folder.
    sim_folder = f"{fs_storage}/{self.sim_code}"

    while True: 
      sim_command = input("Enter option: ")
      sim_command = sim_command.strip()
      ret_str = ""

      try: 
        if sim_command.lower() in ["f", "fin", "finish", "save and finish"]: 
          # Finishes the simulation environment and saves the progress. 
          # Example: fin
          self.save()
          break

        elif sim_command.lower() == "start path tester mode": 
          # Starts the path tester and removes the currently forked sim files.
          # Note that once you start this mode, you need to exit out of the
          # session and restart in case you want to run something else. 
          shutil.rmtree(sim_folder) 
          self.start_path_tester_server()

        elif sim_command.lower() == "exit": 
          # Finishes the simulation environment but does not save the progress
          # and erases all saved data from current simulation. 
          # Example: exit 
          shutil.rmtree(sim_folder) 
          break 

        elif sim_command.lower() == "save": 
          # Saves the current simulation progress. 
          # Example: save
          self.save()

        elif sim_command[:3].lower() == "run": 
          # Runs the number of steps specified in the prompt.
          # Example: run 1000
          int_count = int(sim_command.split()[-1])
          self.start_server(int_count)

        elif ("print persona schedule" 
              in sim_command[:22].lower()): 
          # Print the decomposed schedule of the persona specified in the 
          # prompt.
          # Example: print persona schedule Isabella Rodriguez
          ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                      .scratch.get_str_daily_schedule_summary())

        elif ("print all persona schedule" 
              in sim_command[:26].lower()): 
          # Print the decomposed schedule of all personas in the world. 
          # Example: print all persona schedule
          for persona_name, persona in self.personas.items(): 
            ret_str += f"{persona_name}\n"
            ret_str += f"{persona.scratch.get_str_daily_schedule_summary()}\n"
            ret_str += f"---\n"

        elif ("print hourly org persona schedule" 
              in sim_command.lower()): 
          # Print the hourly schedule of the persona specified in the prompt.
          # This one shows the original, non-decomposed version of the 
          # schedule.
          # Ex: print persona schedule Isabella Rodriguez
          ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                      .scratch.get_str_daily_schedule_hourly_org_summary())

        elif ("print persona current tile" 
              in sim_command[:26].lower()): 
          # Print the x y tile coordinate of the persona specified in the 
          # prompt. 
          # Ex: print persona current tile Isabella Rodriguez
          ret_str += str(self.personas[" ".join(sim_command.split()[-2:])]
                      .scratch.curr_tile)

        elif ("print persona chatting with buffer" 
              in sim_command.lower()): 
          # Print the chatting with buffer of the persona specified in the 
          # prompt.
          # Ex: print persona chatting with buffer Isabella Rodriguez
          curr_persona = self.personas[" ".join(sim_command.split()[-2:])]
          for p_n, count in curr_persona.scratch.chatting_with_buffer.items(): 
            ret_str += f"{p_n}: {count}"

        elif ("print persona associative memory (event)" 
              in sim_command.lower()):
          # Print the associative memory (event) of the persona specified in
          # the prompt
          # Ex: print persona associative memory (event) Isabella Rodriguez
          ret_str += f'{self.personas[" ".join(sim_command.split()[-2:])]}\n'
          ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                                       .a_mem.get_str_seq_events())

        elif ("print persona associative memory (thought)" 
              in sim_command.lower()): 
          # Print the associative memory (thought) of the persona specified in
          # the prompt
          # Ex: print persona associative memory (thought) Isabella Rodriguez
          ret_str += f'{self.personas[" ".join(sim_command.split()[-2:])]}\n'
          ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                                       .a_mem.get_str_seq_thoughts())

        elif ("print persona associative memory (chat)" 
              in sim_command.lower()): 
          # Print the associative memory (chat) of the persona specified in
          # the prompt
          # Ex: print persona associative memory (chat) Isabella Rodriguez
          ret_str += f'{self.personas[" ".join(sim_command.split()[-2:])]}\n'
          ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                                       .a_mem.get_str_seq_chats())

        elif ("print persona spatial memory" 
              in sim_command.lower()): 
          # Print the spatial memory of the persona specified in the prompt
          # Ex: print persona spatial memory Isabella Rodriguez
          self.personas[" ".join(sim_command.split()[-2:])].s_mem.print_tree()

        elif ("print current time" 
              in sim_command[:18].lower()): 
          # Print the current time of the world. 
          # Ex: print current time
          ret_str += f'{self.curr_time.strftime("%B %d, %Y, %H:%M:%S")}\n'
          ret_str += f'steps: {self.step}'

        elif ("print tile event" 
              in sim_command[:16].lower()): 
          # Print the tile events in the tile specified in the prompt 
          # Ex: print tile event 50, 30
          cooordinate = [int(i.strip()) for i in sim_command[16:].split(",")]
          for i in self.maze.access_tile(cooordinate)["events"]: 
            ret_str += f"{i}\n"

        elif ("print tile details" 
              in sim_command.lower()): 
          # Print the tile details of the tile specified in the prompt 
          # Ex: print tile event 50, 30
          cooordinate = [int(i.strip()) for i in sim_command[18:].split(",")]
          for key, val in self.maze.access_tile(cooordinate).items(): 
            ret_str += f"{key}: {val}\n"

        elif ("call -- analysis" 
              in sim_command.lower()): 
          # Starts a stateless chat session with the agent. It does not save 
          # anything to the agent's memory. 
          # Ex: call -- analysis Isabella Rodriguez
          persona_name = sim_command[len("call -- analysis"):].strip() 
          self.personas[persona_name].open_convo_session("analysis")

        elif ("call -- load history" 
              in sim_command.lower()): 
          curr_file = maze_assets_loc + "/" + sim_command[len("call -- load history"):].strip() 
          # call -- load history the_ville/agent_history_init_n3.csv

          rows = read_file_to_list(curr_file, header=True, strip_trail=True)[1]
          clean_whispers = []
          for row in rows: 
            agent_name = row[0].strip() 
            whispers = row[1].split(";")
            whispers = [whisper.strip() for whisper in whispers]
            for whisper in whispers: 
              clean_whispers += [[agent_name, whisper]]

          load_history_via_whisper(self.personas, clean_whispers)

        elif sim_command.lower().startswith("survey"):
          parts = sim_command.split()
          wave_id = parts[1] if len(parts) > 1 else f"step_{self.step}"
          from perception_survey import run_perception_survey
          out_dir = f"{sim_folder}/survey"
          survey_path = run_perception_survey(
              self.personas, self.sim_code, self.step, self.curr_time,
              out_dir, wave_id=wave_id)
          ret_str += f"Survey complete: {survey_path}"

        elif (sim_command.lower().startswith("shock isolate-broker")
              or sim_command.lower().startswith("shock isolate-hub")
              or sim_command.lower().startswith("shock isolate")):
          ret_str += self._try_shock_isolate_command(sim_command, sim_folder)

        elif sim_command.lower().strip() == "unshock":
          ret_str += self._try_unshock_command(sim_folder)

        print (ret_str)

      except:
        traceback.print_exc()
        print ("Error.")
        pass


if __name__ == '__main__':
  # rs = ReverieServer("base_the_ville_isabella_maria_klaus", 
  #                    "July1_the_ville_isabella_maria_klaus-step-3-1")
  # rs = ReverieServer("July1_the_ville_isabella_maria_klaus-step-3-20", 
  #                    "July1_the_ville_isabella_maria_klaus-step-3-21")
  # rs.open_server()

  origin = input("Enter the name of the forked simulation: ").strip()
  target = input("Enter the name of the new simulation: ").strip()

  rs = ReverieServer(origin, target)
  rs.open_server()




















































