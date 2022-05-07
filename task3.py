import logging
from cProfile import run
import random
import pyspiel
import numpy as np
from open_spiel.python import rl_environment, policy
from open_spiel.python.algorithms import tabular_qlearner, random_agent, exploitability


class NFSPPolicies(policy.Policy):
    """Joint policy to be evaluated."""

    def __init__(self, env, nfsp_policies):
        game = env.game
        player_ids = [0, 1]
        super(NFSPPolicies, self).__init__(game, player_ids)
        self._policies = nfsp_policies
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

        p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
        prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict


def eval_against_random_bots(env, trained_agent, num_episodes):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    wins = 0
    for _ in range(num_episodes):
        time_step = env.reset()
        who_random = random.choice([0, 1])
        while not time_step.step_type.last():
            pid = time_step.observations["current_player"]

            if pid == who_random:
                legal_actions = time_step.observations['legal_actions'][pid]
                action = random.choice(legal_actions)
            else:
                # legal_actions = time_step.observations['legal_actions'][pid]
                # action = random.choice(legal_actions)
                action = trained_agent.step(time_step, is_evaluation=True).action

            time_step = env.step([action])

        if time_step.rewards[1 - who_random] > 0:
            wins += 1
    return wins / num_episodes


env = rl_environment.Environment("kuhn_poker")
num_actions = env.action_spec()["num_actions"]

agents = [
    tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
    for idx in range(2)
]
expl_policies_avg = NFSPPolicies(env, agents)

random_agent = random_agent.RandomAgent(player_id=0, num_actions=num_actions)

for cur_episode in range(int(1e6)):
    if cur_episode % int(5e3) == 0:
        losses = [agent.loss for agent in agents]
        logging.info("Losses: %s", losses)
        expl = exploitability.exploitability(env.game, expl_policies_avg)
        logging.info("[%s] Exploitability AVG %s", cur_episode + 1, expl)
        logging.info("_____________________________________________")

    time_step = env.reset()
    agents_order = random.sample(agents, 2)
    while not time_step.step_type.last():
        pid = time_step.observations["current_player"]
        action = agents_order[pid].step(time_step).action

        time_step = env.step([action])

    for agent in agents_order:
        agent.step(time_step)

    # for stateKey, stateVal in agents[0]._q_values.items():
    #     for key, val in stateVal.items():
    #         print(stateKey, "||", key, val)
    # print("------------------------------------------------")
