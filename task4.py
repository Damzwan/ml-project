
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from calendar import SATURDAY

from absl import app

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
import pyspiel
import logging
import tensorflow.compat.v1 as tf
import os
from numpy import format_float_scientific


NUM_TRAIN_EPISODES = 2000000
EVAL_EVERY = 1000
SAVE_EVERY = 5000
HIDDEN_LAYERS_SIZES = [128]
REPLAY_BUFFER_CAPACITY = int(10e3)  # 1e3 ~= 650MB  -> don't overdo!          
RESERVOIR_BUFFER_CAPACITY = int(2e6)                                    
ANTICITORY_PARAM = 0.1
                  

def main(unused_argv):
  fcpa_game_string = (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=150 100,"
        "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
        "stack=20000 20000,bettingAbstraction=fcpa)")
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
                learn_every=100,
                optimizer_str='adam',
                loss_str='mse',
                min_buffer_size_to_learn=800) for idx
        in range(num_players)
    ]
    sess.run(tf.global_variables_initializer())

    bestLoss = float('inf')

    for ep in range(NUM_TRAIN_EPISODES):
      if (ep + 1) % EVAL_EVERY == 0:
        logging.info("[%s] Losses: %s (%s)", ep+1, [agent.loss for agent in agents], [format_float_scientific(agent.loss, precision=1) for agent in agents])

        if agents[0].loss < bestLoss: #(ep + 1) % SAVE_EVERY == 0:
          bestLoss = agents[0].loss
          agents[0].save(dqnout)
          agents[1].save(dqnout)

      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        time_step = env.step([agents[player_id].step(time_step).action])

      for agent in agents:
        agent.step(time_step)

  # print("first")
  # player_id = 0
  # graph = tf.Graph()
  # sess = tf.Session(graph=graph)
  # sess.__enter__()
  # with graph.as_default():
  #     dqnAgent = dqn.DQN(sess, player_id, info_state_size, num_actions, 128, int(2e5))
  #     dqnAgent.restore(dqnout)

  # print('second')
  # player_id = 1
  # graph = tf.Graph()
  # sess = tf.Session(graph=graph)
  # sess.__enter__()
  # with graph.as_default():
  #     dqnAgent = dqn.DQN(sess, player_id, info_state_size, num_actions, 128, int(2e5))
  #     dqnAgent.restore(dqnout)

if __name__ == "__main__":
  app.run(main)
  # agents = [LoadAgent("random", 'fcpa', i, 1234) for i in range(2)]
  # runPokerGame(agents)