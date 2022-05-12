from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from calendar import SATURDAY
from random import random

from absl import app

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
import pyspiel
import logging
import tensorflow.compat.v1 as tf
import os
import numpy as np
from open_spiel.python.algorithms.evaluate_bots import evaluate_bots
from tournament import load_agent_from_dir
from fcpa_agent_temp import createAgentFromDQN

NUM_TRAIN_EPISODES = 2000000
EVAL_EVERY = 20000
HIDDEN_LAYERS_SIZES = [128]
REPLAY_BUFFER_CAPACITY = int(10e3)  # 1e3 ~= 650MB  -> don't overdo!          
RESERVOIR_BUFFER_CAPACITY = int(2e6)

import os, sys


def getAverageScore(game, agents, random_agents, num_rounds):
    results = [[], []]
    agents = [createAgentFromDQN(0, agents[0]), createAgentFromDQN(1, agents[1])]
    random_agents = [random_agents['agent_p1'], random_agents['agent_p2']]

    for (p1, p2) in [(agents[0], random_agents[1]), (random_agents[0], agents[1])]:
        for _ in range(num_rounds):
            returns = evaluate_bots(game.new_initial_state(), [p1, p2], np.random)
            results[p1 == random_agents[0]].append(returns[p1 == random_agents[0]])

    return [sum(x) / len(x) for x in results]


def main(unused_argv):
  fcpa_game_string = (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=150 100,"
        "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
        "stack=20000 20000,bettingAbstraction=fcpa)")
  random_agent = load_agent_from_dir('random', './bots/random/')
  game = pyspiel.load_game(fcpa_game_string)
  num_players = 2

  env_configs = {"players": num_players}
  env = rl_environment.Environment(game, **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  hidden_layers_sizes = [int(l) for l in HIDDEN_LAYERS_SIZES]

  dqnout = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bots', 'custom', 'models')

  graph = tf.Graph()
  sess = tf.Session(graph=graph)
  sess.__enter__()
  with graph.as_default():
    agents = [
        dqn.DQN(sess, 
                idx, 
                info_state_size, 
                num_actions, 
                hidden_layers_sizes, 
                REPLAY_BUFFER_CAPACITY,
                epsilon_start=0.8, 
                epsilon_end=0.001,
                learn_every=1000,
                optimizer_str='adam',
                loss_str='mse',
                min_buffer_size_to_learn=500) for idx
        in range(num_players)
    ]
    sess.run(tf.global_variables_initializer())

    bestAverage = [-20000, -20000] # initialize best average on negative of betting stack

    for ep in range(NUM_TRAIN_EPISODES):
      if (ep + 1) % EVAL_EVERY == 0 and ep > 500:
        avg = getAverageScore(game, agents, random_agent, 10000)
        logging.info("[%s] average reward: %s, loss: %s", ep+1, avg, [agent.loss for agent in agents])

        for playerIndex in range(2):
          if avg[playerIndex] > bestAverage[playerIndex]:
            bestAverage[playerIndex] = avg[playerIndex]
            agents[playerIndex].save(dqnout)

      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        time_step = env.step([agents[player_id].step(time_step).action])

      for agent in agents:
        agent.step(time_step)


if __name__ == "__main__":
    app.run(main)
    # agents = [LoadAgent("random", 'fcpa', i, 1234) for i in range(2)]
    # runPokerGame(agents)
