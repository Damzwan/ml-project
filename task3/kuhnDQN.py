# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NFSP agents trained on Kuhn Poker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability, dqn, random_agent
import numpy as np
import matplotlib.pyplot as plt

NUM_TRAIN_EPISODES = int(300)
EVAL_EVERY = 5
HIDDEN_LAYERS_SIZES = [128]
REPLAY_BUFFER_CAPACITY = int(2e5)  # 1e3 ~= 650MB  -> don't overdo!          
RESERVOIR_BUFFER_CAPACITY = int(2e6)


class DqnPolicies(policy.Policy):
    """Joint policy to be evaluated."""


def __init__(self, env, dqn_policies):
    game = env.game
    player_ids = [0, 1]
    super(DqnPolicies, self).__init__(game, player_ids)
    self._policies = dqn_policies
    self._obs = {"info_state": [None, None], "legal_actions": [None, None]}


def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    print(info_state)
    p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict


def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    num_players = len(trained_agents)
    sum_episode_rewards = np.zeros(num_players)
    for player_pos in range(num_players):
        cur_agents = random_agents[:]
        cur_agents[player_pos] = trained_agents[player_pos]
        for _ in range(num_episodes):
            time_step = env.reset()
            episode_rewards = 0
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
                action_list = [agent_output.action]
                time_step = env.step(action_list)
                episode_rewards += time_step.rewards[player_pos]
            sum_episode_rewards[player_pos] += episode_rewards
    return sum_episode_rewards / num_episodes


def main(unused_argv):
    game = "kuhn_poker"
    num_players = 2

    env_configs = {"players": num_players}
    env = rl_environment.Environment(game, **env_configs)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    hidden_layers_sizes = [int(l) for l in HIDDEN_LAYERS_SIZES]

    expls = []
    nashs = []
    iterations = []

    with tf.Session() as sess:
        # pylint: disable=g-complex-comprehension
        agents = [
            dqn.DQN(
                session=sess,
                player_id=idx,
                state_representation_size=info_state_size,
                num_actions=num_actions,
                hidden_layers_sizes=hidden_layers_sizes,
                replay_buffer_capacity=REPLAY_BUFFER_CAPACITY,
                batch_size=5,
                min_buffer_size_to_learn=10) for idx in range(num_players)
        ]
        expl_policies_avg = DqnPolicies(env, agents)

        random_agents = [
            random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
            for idx in range(num_players)
        ]

        results = [[], [], [], []]  # iterations, average 1, average 2, expl
        expl_policies_avg = DqnPolicies(env, agents)

        for ep in range(NUM_TRAIN_EPISODES):
            if (ep + 1) % EVAL_EVERY == 0 and ep >= 10:
                # losses = [agent.loss for agent in agents]
                # logging.info("Losses: %s", losses)
                expl = exploitability.exploitability(env.game, expl_policies_avg)
                nash = exploitability.nash_conv(env.game, expl_policies_avg)
                avg = eval_against_random_bots(env, agents, random_agents, 10000)
                logging.info("[%s] average reward %s, exploit and nash: %s %s", ep + 1, avg, expl, nash)
                results[0].append(ep + 1)
                results[1].append(avg[0])
                results[2].append(avg[1])
                results[3].append(expl)

                expl = exploitability.exploitability(env.game, expl_policies_avg)
                nash = exploitability.nash_conv(env.game, expl_policies_avg)
                expls.append(expl)
                nashs.append(nash)
                iterations.append(ep + 1)
                logging.info("[%s] Exploitability AVG %s, %s", ep + 1, expl, nash)
                logging.info("_____________________________________________")

            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = agents[player_id].step(time_step)
                action_list = [agent_output.action]
                time_step = env.step(action_list)

            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)

    plt.plot(results[0], results[1], label='Avg agent 1')
    plt.plot(results[0], results[2], label='Avg agent 2')
    plt.legend()
    print(results)
    plt.savefig('dqnexpl.png')


if __name__ == "__main__":
    app.run(main)
