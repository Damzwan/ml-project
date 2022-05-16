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

"""Tabular Q-learning agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from custom_tabular_qlearner import Custom_QLearner

import collections
import random

import numpy as np

from open_spiel.python import rl_agent
from open_spiel.python import rl_tools


def valuedict():
    return collections.defaultdict(float)


class FAQQLearner(Custom_QLearner):
    def __init__(self,
                 player_id,
                 num_actions,
                 step_size=0.001,
                 epsilon_schedule=rl_tools.ConstantSchedule(0.2),
                 discount_factor=1.0,
                 centralized=False,
                 init_q_values=None):
        super(FAQQLearner, self).__init__(player_id, num_actions, step_size, epsilon_schedule, discount_factor,
                                       centralized, init_q_values)
        self._prev_probs = None
        self.beta = 0.01

    def step(self, time_step, is_evaluation=False):
        """Returns the action to be taken and updates the Q-values if needed.

        Args:
          time_step: an instance of rl_environment.TimeStep.
          is_evaluation: bool, whether this is a training or evaluation call.

        Returns:
          A `rl_agent.StepOutput` containing the action probs and chosen action.
        """
        if self._centralized:
            info_state = str(time_step.observations["info_state"])
        else:
            info_state = str(time_step.observations["info_state"][self._player_id])
        legal_actions = time_step.observations["legal_actions"][self._player_id]

        # Prevent undefined errors if this agent never plays until terminal step
        action, probs = None, None

        # Act step: don't act at terminal states.
        if not time_step.last():
            epsilon = 0.0 if is_evaluation else self._epsilon
            action, probs = self._boltzman(info_state, legal_actions, epsilon)

        # Learn step: don't learn during evaluation owith their rewards respectiver at first agent steps.
        if self._prev_info_state and not is_evaluation:
            target = time_step.rewards[self._player_id]
            if not time_step.last():  # Q values are zero for terminal.
                target += self._discount_factor * max(
                    [self._q_values[info_state][a] for a in legal_actions])

            prev_q_value = self._q_values[self._prev_info_state][self._prev_action]
            self._last_loss_value = target - prev_q_value
            self._q_values[self._prev_info_state][self._prev_action] += min(
                self.beta / self._prev_probs[self._prev_action], 1) * self._step_size * self._last_loss_value

            # Decay epsilon, if necessary.
            self._epsilon = self._epsilon_schedule.step()

            if time_step.last():  # prepare for the next episode.
                self._prev_info_state = None
                return

        # Don't mess up with the state during evaluation.
        if not is_evaluation:
            self._prev_info_state = info_state
            self._prev_action = action
            self._prev_probs = probs
        return rl_agent.StepOutput(action=action, probs=probs)
