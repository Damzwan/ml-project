import logging

import pyspiel

from open_spiel.python.egt import dynamics
from open_spiel.python.egt.visualization import Dynamics2x2Axes, _eval_dynamics_2x2_grid, Dynamics3x3Axes
from open_spiel.python.utils import file_utils
from absl import app
import numpy as np
import random
import matplotlib.pyplot as plt

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner, random_agent
from open_spiel.python.egt.utils import game_payoffs_array


def _manually_create_game(utilities1, utilities2, sum_type, row_action_names):
    """Creates the game manually from the spiel building blocks."""
    game_type = pyspiel.GameType(
        "matrix_rps",
        "Battle of sexes",
        pyspiel.GameType.Dynamics.SIMULTANEOUS,
        pyspiel.GameType.ChanceMode.DETERMINISTIC,
        pyspiel.GameType.Information.ONE_SHOT,
        sum_type,
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
        row_action_names,  # row_action_names
        row_action_names,  # col_action_names
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
    return wins / num_episodes


def eval_average(env, trained_agents):
    time_step = env.reset()
    while not time_step.last():
        trained_output0 = trained_agents[0].step(time_step, is_evaluation=True)
        trained_output1 = trained_agents[1].step(time_step, is_evaluation=True)
        time_step = env.step([trained_output0.action, trained_output1.action])
    return time_step.rewards[0], time_step.rewards[1]


rps_game = _manually_create_game([[0, -0.25, 0.5], [0.25, 0, -0.05], [-0.5, 0.05, 0]],
                                 [[0, 0.25, -0.5], [-0.25, 0, 0.05], [0.5, -0.05, 0]],
                                 pyspiel.GameType.Utility.ZERO_SUM, ["Rock", "Paper", "Scissors"])

dispersion_game = _manually_create_game([[-1, 1], [1, -1]], [[1, -1], [1, -1]],
                                        pyspiel.GameType.Utility.GENERAL_SUM, ["D1", "D2"])

battle_of_sexes_game = _manually_create_game([[3, 0], [0, 2]], [[2, 0], [0, 3]],
                                             pyspiel.GameType.Utility.GENERAL_SUM,
                                             ["O", "M"])

subsidy_game = _manually_create_game([[10, 0], [11, 12]], [[10, 11], [0, 12]],
                                     pyspiel.GameType.Utility.GENERAL_SUM, ["S1", "S2"])


def subtask1():
    env = rl_environment.Environment(battle_of_sexes_game)

    num_players = 2
    training_episodes = int(1e5) + 1
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

    q_val = {0: [], 1: [], 2: []}
    # 1. Train the agents
    for cur_episode in range(training_episodes):
        if cur_episode % int(1e3) == 0:
            # win_rates = eval_against_random_bots(env, agents, random_agents, 1000)
            avg0, avg1 = eval_average(env, agents)
            logging.info("Starting episode %s, average rewards: %s vs %s", cur_episode, avg0, avg1)
        time_step = env.reset()
        while not time_step.last():
            agent1_output = agents[0].step(time_step)
            agent2_output = agents[1].step(time_step)
            time_step = env.step([agent1_output.action, agent2_output.action])

        # Episode is over, step all agents with final info state.
        # agents[0].step(time_step)
        for agent in agents:
            agent.step(time_step)

        for key, val in list(agents[0]._q_values.values())[0].items():
            q_val[key].append(val)

    plt.plot(range(len(q_val[0])), q_val[0], label='q0')
    plt.plot(range(len(q_val[1])), q_val[1], label='q1')
    plt.plot(range(len(q_val[2])), q_val[2], label='q2')
    plt.legend()
    plt.savefig('qval.png')

    for agent in agents:
        print(agent._q_values)


def subtask2():
    game = battle_of_sexes_game
    rd = dynamics.boltzmannq
    game_size = game.num_rows()
    payoff_matrix = game_payoffs_array(game)

    if game_size == 2:
        dyn = dynamics.MultiPopulationDynamics(payoff_matrix, rd)
        visualiser = Dynamics2x2Axes(plt.figure(), [0, 0, 1, 1])
        visualiser.streamplot(dyn)

        x, y, u, v = _eval_dynamics_2x2_grid(dyn, 50)
        plt.streamplot(x, y, u, v)
        plt.show()
    elif game_size == 3:
        dyn = dynamics.SinglePopulationDynamics(payoff_matrix, rd)
        visualiser = Dynamics3x3Axes(plt.figure(), [0, 0, 1, 1])
        visualiser.streamplot(dyn)



def main(_):
    subtask2()


if __name__ == "__main__":
    app.run(main)
