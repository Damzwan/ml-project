import logging

import pyspiel
from open_spiel.python.utils import file_utils
from absl import app
import numpy as np
import random
import matplotlib.pyplot as plt

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner, random_agent


def _manually_create_game(utilities1, utilities2):
    """Creates the game manually from the spiel building blocks."""
    game_type = pyspiel.GameType(
        "matrix_rps",
        "Battle of sexes",
        pyspiel.GameType.Dynamics.SIMULTANEOUS,
        pyspiel.GameType.ChanceMode.DETERMINISTIC,
        pyspiel.GameType.Information.ONE_SHOT,
        pyspiel.GameType.Utility.GENERAL_SUM,
        pyspiel.GameType.RewardModel.TERMINAL,
        2,  # max num players
        2,  # min_num_players
        True,  # provides_information_state
        True,  # provides_information_state_tensor
        False,  # provides_observation
        False,  # provides_observation_tensor
        dict()  # parameter_specification
    )
    game = pyspiel.MatrixGame(
        game_type,
        {},  # game_parameters
        ["Rock", "Paper", "sc"],  # row_action_names
        ["Rock", "Paper", "sc"],  # col_action_names
        utilities1,  # row player utilities
        utilities2  # col player utilities
    )
    return game


def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    wins = 0
    for _ in range(num_episodes):
        time_step = env.reset()
        while not time_step.last():
            trained_output = random.choice(trained_agents).step(time_step, is_evaluation=True)
            random_output = random.choice(random_agents).step(time_step, is_evaluation=True)
            time_step = env.step([trained_output.action, random_output.action])
        if time_step.rewards[0] > 0:
            wins += 1
    return wins/num_episodes

def eval_average(env, trained_agents):
    time_step = env.reset()
    while not time_step.last():
        trained_output0 = trained_agents[0].step(time_step, is_evaluation=True)
        trained_output1 = trained_agents[1].step(time_step, is_evaluation=True)
        time_step = env.step([trained_output0.action, trained_output1.action])
    return time_step.rewards[0], time_step.rewards[1]

def main(_):
    rps_game = _manually_create_game([[0, -0.25, 0.5], [0.25, 0, -0.05], [-0.5, 0.05, 0]],
                                     [[0, 0.25, -0.5], [-0.25, 0, 0.05], [0.5, -0.05, 0]])

    sexes_game = _manually_create_game([[3, 0], [0, 2]], [[2, 0], [0, 3]]) 
    
    dispersion_game = _manually_create_game([[-1, 1], [1, -1]], [[1, -1], [1, -1]])

    subsidy_game = _manually_create_game([[10, 0], [11, 12]], [[10, 0], [11, 12]])


    num_players = 2
    training_episodes = int(1e5) + 1

    env = rl_environment.Environment(subsidy_game)
    num_actions = env.action_spec()["num_actions"]

    agents = [
        tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    # random agents for evaluation
    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    q_val = {0: [], 1: [], 2:[]}
    # 1. Train the agents
    for cur_episode in range(training_episodes):
        if cur_episode % int(1e3) == 0:
            # win_rates = eval_against_random_bots(env, agents, random_agents, 1000)
            avg0, avg1 = eval_average(env, agents)
            logging.info("Starting episode %s, average rewards: %s vs %s", cur_episode, avg0, avg1)
        time_step = env.reset()
        while not time_step.last():
            agent1_output = agents[0].step(time_step, is_evaluation=False)
            agent2_output = agents[1].step(time_step, is_evaluation=False)
            time_step = env.step([agent1_output.action, agent2_output.action])

        # Episode is over, step all agents with final info state.
        #agents[0].step(time_step)
        for agent in agents:
            agent.step(time_step)
        
        for key, val in list(agents[0]._q_values.values())[0].items():
            q_val[key].append(val)
    
    plt.plot(range(len(q_val[0])), q_val[0], label='q0')
    plt.plot(range(len(q_val[1])), q_val[1], label='q1')
    plt.plot(range(len(q_val[2])), q_val[2], label='q2')
    plt.legend()
    plt.savefig('qval.png')


if __name__ == "__main__":
    app.run(main)