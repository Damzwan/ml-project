#!/usr/bin/env python3
# encoding: utf-8
"""
fcpa_agent.py

Extend this class to provide an agent that can participate in a tournament.

Created by Pieter Robberechts, Wannes Meert.
Copyright (c) 2021 KU Leuven. All rights reserved.
"""

import sys
import argparse
import logging
import numpy as np
import pyspiel
from open_spiel.python.algorithms import evaluate_bots, dqn
from open_spiel.python import rl_environment
import tensorflow.compat.v1 as tf
import os, copy


logger = logging.getLogger('be.kuleuven.cs.dtai.fcpa')


def get_agent_for_tournament(player_id):
    """Change this function to initialize your agent.
    This function is called by the tournament code at the beginning of the
    tournament.

    :param player_id: The integer id of the player for this bot, e.g. `0` if
        acting as the first player.
    """
    my_player = Agent(player_id)
    return my_player


def createAgentFromDQN(player_id, dqnObject):
    my_player = Agent(player_id, dqnObject)
    return my_player

class Agent(pyspiel.Bot):
    """Agent template"""

    def __init__(self, player_id, dqnObject=None):
        """Initialize an agent to play FCPA poker.

        Note: This agent should make use of a pre-trained policy to enter
        the tournament. Initializing the agent should thus take no more than
        a few seconds.
        """
        pyspiel.Bot.__init__(self)
        self.player_id = player_id
        if player_id == 1: # todo make player 2 loadable
            return

        self.verbose = False

        self.setNewEnv()
        info_state_size = self.simulatedEnv.observation_spec()["info_state"][0]
        num_actions = self.simulatedEnv.action_spec()["num_actions"]

        if dqnObject:
            self.dqnAgent = dqnObject
            return


        self.graph = tf.Graph()
        sess = tf.Session(graph=self.graph)
        sess.__enter__()
        with self.graph.as_default():
            agents = [
                dqn.DQN(sess,
                        idx,
                        info_state_size,
                        num_actions) for idx # todo maybe needs all parameters
                in range(2)
            ]  # First define all agents, this way the session is aware of all the agent variables
            dqnout = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            agents = [agent.restore(dqnout) for agent in agents]
            self.dqnAgent = agents[player_id]


        # self.graph = tf.Graph()
        # sess = tf.Session(graph=self.graph)
        # sess.__enter__()
        # with self.graph.as_default():
        #     dqnout = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        #     self.dqnAgent = dqn.DQN(sess, player_id, info_state_size, num_actions, 128, int(2e5))
        #     self.dqnAgent.restore(dqnout)

    def setNewEnv(self):
        fcpa_game_string = (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=150 100,"
        "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
        "stack=20000 20000,bettingAbstraction=fcpa)")
        game = pyspiel.load_game(fcpa_game_string)
        num_players = 2

        env_configs = {"players": num_players}
        env = rl_environment.Environment(game, **env_configs)
        
        self.simulatedEnv = env
          

    def restart_at(self, state):
        """Starting a new game in the given state.

        :param state: The initial state of the game.
        """
        if self.verbose:
            print("restarting....")
        if not self.simulatedEnv:
            self.setNewEnv()


    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions.

        :param state: The current state of the game.
        :param player_id: The ID of the player that executed an action.
        :param action: The action which the player executed.
        """
        if self.verbose:
            print("informed of action", state.action_to_string(action))
        

    def step(self, state):
        """Returns the selected action in the given state.

        :param state: The current state of the game.
        :returns: The selected action from the legal actions, or
            `pyspiel.INVALID_ACTION` if there are no legal actions available.
        """
        # a = random.choice(state.legal_actions())
        # print("custom chooses", state.action_to_string(a))
        # return a
        
        stato = copy.deepcopy(state)
        self.updateSimulatedEnv(stato)
        time_step = self.simulatedEnv.get_time_step()
        action = self.dqnAgent.step(time_step, is_evaluation=True).action
        if self.verbose:
            print("custom chose action ", stato.action_to_string(action))
        
        # uncomment to train
        # time_step = self.simulatedEnv.step([action])
        # if time_step.last():
        #     self.dqnAgent.step(time_step) # update with last timestep, as usual
        return action

    def updateSimulatedEnv(self, state):
        self.simulatedEnv.set_state(state)

def test_api_calls():
    """This method calls a number of API calls that are required for the
    tournament. It should not trigger any Exceptions.
    """
    fcpa_game_string = (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=150 100,"
        "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
        "stack=20000 20000,bettingAbstraction=fcpa)")
    game = pyspiel.load_game(fcpa_game_string)
    bots = [get_agent_for_tournament(player_id) for player_id in [0,1]]
    returns = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
    assert len(returns) == 2
    assert isinstance(returns[0], float)
    assert isinstance(returns[1], float)
    print("SUCCESS!")


def main(argv=None):
    test_api_calls()


if __name__ == "__main__":
    sys.exit(main())

