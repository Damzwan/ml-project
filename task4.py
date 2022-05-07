
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import numpy as np
import random

from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import pyspiel

def LoadAgent(agent_type, game, player_id, rng):
  """Return a bot based on the agent type."""
  if agent_type == "random":
    return uniform_random.UniformRandomBot(player_id, rng)
  elif agent_type == "human":
    return human.HumanBot()
  elif agent_type == "check_call":
    policy = pyspiel.PreferredActionPolicy([1, 0])
    return pyspiel.make_policy_bot(game, player_id, rng, policy)
  elif agent_type == "fold":
    policy = pyspiel.PreferredActionPolicy([0, 1])
    return pyspiel.make_policy_bot(game, player_id, rng, policy)
  else:
    raise RuntimeError("Unrecognized agent type: {}".format(agent_type))

def runPokerGame(agents):
  game = pyspiel.load_game(pyspiel.hunl_game_string("fcpa"))
  state = game.new_initial_state()

  print("INITIAL STATE")
  print(str(state))

  while not state.is_terminal():
    current_player = state.current_player()
    if state.is_chance_node():
      # Chance node: sample an outcome
      outcomes = state.chance_outcomes()
      num_actions = len(outcomes)
      print("Chance node with " + str(num_actions) + " outcomes")
      action_list, prob_list = zip(*outcomes)
      action = random.choices(action_list, weights=prob_list)[0]
      print("Sampled outcome: ",
            state.action_to_string(state.current_player(), action))
      state.apply_action(action)
    else:
      legal_actions = state.legal_actions()
      for action in legal_actions:
        print("Legal action: {} ({})".format(
            state.action_to_string(current_player, action), action))
        
      #action = agents[current_player].step(state)
      action = random.choice(list(range(len(legal_actions))))
      action_string = state.action_to_string(current_player, action)
      print("Player ", current_player, ", chose action: ",
            action_string)
      state.apply_action(action)

    print("")
    print("NEXT STATE:")
    print(str(state))

  return state.returns()
#   for pid in range(game.num_players()):
#     print("Utility for player {} is {}".format(pid, returns[pid]))


if __name__ == "__main__":
  agents = [LoadAgent("random", 'fcpa', i, 1234) for i in range(2)]
  runPokerGame(agents)