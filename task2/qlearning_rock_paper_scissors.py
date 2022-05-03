import logging

import pyspiel
from open_spiel.python.utils import file_utils
from absl import app
import numpy as np
import random

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner, random_agent


def _manually_create_game(utilities1, utilities2):
    """Creates the game manually from the spiel building blocks."""
    game_type = pyspiel.GameType(
        "matrix_rps",
        "Rock Paper Scissors",
        pyspiel.GameType.Dynamics.SIMULTANEOUS,
        pyspiel.GameType.ChanceMode.DETERMINISTIC,
        pyspiel.GameType.Information.ONE_SHOT,
        pyspiel.GameType.Utility.ZERO_SUM,
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
        ["Rock", "Paper", "Scissors"],  # row_action_names
        ["Rock", "Paper", "Scissors"],  # col_action_names
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


def main(_):
    rps_game = _manually_create_game([[0, -0.25, 0.5], [0.25, 0, -0.05], [-0.5, 0.05, 0]],
                                     [[0, 0.25, -0.5], [-0.25, 0, 0.05], [0.5, -0.05, 0]])

    dispersion_game = _manually_create_game([[3, 0], [0, 2]], [[2, 0], [0, 3]])  # TODO is not ZERO SUM
    # TODO define other games

    num_players = 2
    training_episodes = int(5e5) + 1

    env = rl_environment.Environment(rps_game)
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

    # 1. Train the agents
    for cur_episode in range(training_episodes):
        if cur_episode % int(1e4) == 0:
            win_rates = eval_against_random_bots(env, agents, random_agents, 1000)
            logging.info("Starting episode %s, win_rates %s", cur_episode, win_rates)
        time_step = env.reset()
        while not time_step.last():
            agent1_output = agents[0].step(time_step, is_evaluation=True)
            agent2_output = agents[1].step(time_step, is_evaluation=True)
            time_step = env.step([agent1_output.action, agent2_output.action])

        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)


if __name__ == "__main__":
    app.run(main)