import logging

import pyspiel

from open_spiel.python.egt import dynamics
from open_spiel.python.egt.dynamics import boltzmannq
from open_spiel.python.egt.visualization import Dynamics2x2Axes, _eval_dynamics_2x2_grid, Dynamics3x3Axes
from open_spiel.python.utils import file_utils
from absl import app
import numpy as np
import random
import matplotlib.pyplot as plt

from open_spiel.python import rl_environment, policy
import custom_tabular_qlearner
from LFAQ_custom_tabular_qlearner import LFAQQLEARNING
from custom_tabular_qlearner import Custom_QLearner
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

dispersion_game = _manually_create_game([[-1, 1], [1, -1]], [[-1, 1], [1, -1]],
                                        pyspiel.GameType.Utility.GENERAL_SUM, ["D1", "D2"])

battle_of_sexes_game = _manually_create_game([[3, 0], [0, 2]], [[2, 0], [0, 3]],
                                             pyspiel.GameType.Utility.GENERAL_SUM,
                                             ["O", "M"])

subsidy_game = _manually_create_game([[10, 0], [11, 12]], [[10, 11], [0, 12]],
                                     pyspiel.GameType.Utility.GENERAL_SUM, ["S1", "S2"])


def subtask1():
    env = rl_environment.Environment(subsidy_game)

    num_players = 2
    training_episodes = 1001
    num_actions = env.action_spec()["num_actions"]

    agents = [
        custom_tabular_qlearner.Custom_QLearner(player_id=idx, num_actions=num_actions)
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

    plt.plot(range(len(q_val[0])), q_val[0], label='S1')
    plt.plot(range(len(q_val[1])), q_val[1], label='S2')
    # plt.plot(range(len(q_val[2])), q_val[2], label='scissors')
    plt.xlabel('rounds')
    plt.ylabel('Q-value')
    plt.legend()
    plt.savefig('task2.1-subsidy.png')

    for agent in agents:
        print(agent._q_values)


class MultiPopulationDynamicsLenientBoltzman(object):
    def __init__(self, payoff_tensor, dynamics, k=2):
        """Initializes the multi-population dynamics."""
        if isinstance(dynamics, list) or isinstance(dynamics, tuple):
            assert payoff_tensor.shape[0] == len(dynamics)
        else:
            dynamics = [dynamics] * payoff_tensor.shape[0]
        self.payoff_tensor = payoff_tensor
        self.dynamics = dynamics
        self.k = k

    def lenient_boltzmannq(self, state, payoff):
        res = []

        for i in range(len(state)):
            u = 0
            for j in range(len(state)):
                first_term = payoff[i][j] * state[j]
                candidate_values = payoff[i][payoff[i][:] <= payoff[i][j]]

                indices_smaller_or_eq = np.argwhere(candidate_values[:] <= payoff[i][j]).flatten()
                indices_smaller = np.argwhere(candidate_values[:] < payoff[i][j]).flatten()
                indices_eq = np.argwhere(candidate_values[:] == payoff[i][j]).flatten()

                second_term = (np.sum(state[indices_smaller_or_eq]) ** self.k - np.sum(
                    state[indices_smaller] ** self.k)) / np.sum(
                    state[indices_eq])

                u += first_term * second_term

            res.append(u)

        return res

    def __call__(self, state, time=None):
        state = np.array(state)
        n = self.payoff_tensor.shape[0]  # number of players
        ks = self.payoff_tensor.shape[1:]  # number of strategies for each player
        assert state.shape[0] == sum(ks)

        states = np.split(state, np.cumsum(ks)[:-1])
        dstates = [None] * n
        for i in range(n):
            payoff = np.moveaxis(self.payoff_tensor[i], i, 0)
            fitness = self.lenient_boltzmannq(states[1 - i], payoff)
            dstates[i] = self.dynamics[i](states[i], fitness)

        return np.concatenate(dstates)


def subtask2():
    game = subsidy_game
    env = rl_environment.Environment(game)

    use_boltz = True
    k = 5 if use_boltz else ''

    num_players = 2
    training_episodes = int(1e4)
    num_actions = env.action_spec()["num_actions"]
    prob_histories = []
    probs = [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [.65, 0.5]], [[0.65, 0.5], [0.5, 0.5]]]

    for prob in probs:
        history = []

        if use_boltz:
            agents = [
                LFAQQLEARNING(player_id=idx, num_actions=num_actions, init_q_values=prob[idx], step_size=0.01, k=k,
                              discount_factor=1)
                for idx in range(num_players)
            ]

        else:
            agents = [
                Custom_QLearner(player_id=idx, num_actions=num_actions, init_q_values=prob[idx], step_size=0.001)
                for idx in range(num_players)
            ]

        for cur_episode in range(training_episodes):
            time_step = env.reset()
            while not time_step.last():
                agent1_output = agents[0].step(time_step)
                agent2_output = agents[1].step(time_step)
                if not use_boltz:
                    history.append([agent1_output.probs[0], agent2_output.probs[0]])
                time_step = env.step([agent1_output.action, agent2_output.action])

            # Episode is over, step all agents with final info state.
            # agents[0].step(time_step)

            if use_boltz:
                agent1_output = agents[0].step(time_step)
                agent2_output = agents[1].step(time_step)

                if agent1_output is not None:
                    history.append([agent1_output.probs[0], agent2_output.probs[0]])
            else:
                agents[0].step(time_step)
                agents[1].step(time_step)

        prob_histories.append(history)
    plot_trajectory(prob_histories, game, use_boltz, k)


def plot_trajectory(prob_histories, game, use_boltz, k):
    rd = dynamics.boltzmannq if use_boltz else dynamics.replicator
    game_size = game.num_rows()
    payoff_matrix = game_payoffs_array(game)

    dyn = dynamics.MultiPopulationDynamics(payoff_matrix, rd) if game_size == 2 else dynamics.SinglePopulationDynamics(
        payoff_matrix, rd)
    dyn = MultiPopulationDynamicsLenientBoltzman(payoff_matrix, rd, k=k) if use_boltz else dyn

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='2x2') if game_size == 2 else fig.add_subplot(111, projection='3x3')
    ax.quiver(dyn)

    for history in prob_histories:
        x = [lst[0] for lst in history]
        y = [lst[1] for lst in history]
        plt.plot(x, y)

    if game_size == 2:
        plt.xlabel('Player 0: Probability of playing S1')
        plt.ylabel('Player 1: Probability of playing S2')
    else:
        ax.set_labels(['R', 'P', 'S'])

    name = 'boltz' if use_boltz else 'replicator'
    plt.savefig('task2.2-' + name + '-subs' + str(k) + '.jpg')


def main(_):
    subtask2()


if __name__ == "__main__":
    app.run(main)
