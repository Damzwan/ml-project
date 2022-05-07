from cProfile import run
import random
import pyspiel
import numpy as np
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner, random_agent

# def runOneGame(agents):
#     game = pyspiel.load_game("kuhn_poker")
#     state = game.new_initial_state()
#     while not state.is_terminal():
#         legal_actions = state.legal_actions()
#         if state.is_chance_node():
#             outcomes_with_probs = state.chance_outcomes()
#             action_list, prob_list = zip(*outcomes_with_probs)
#             print(action_list, prob_list)
#             action = np.random.choice(action_list, p=prob_list)
#             state.apply_action(action)
#         else:
#             state.apply_action(legal_actions[getAction(state, agents)])
#         print("state:", state)
#     print(state.rewards)

# def getAction(state, agents):
#     # returns index of legal action to perform
#     return 0


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

        if time_step.rewards[1-who_random] > 0:
            wins += 1
    return wins / num_episodes


env = rl_environment.Environment("kuhn_poker")
num_actions = env.action_spec()["num_actions"]

agents = [
    tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
    for idx in range(2)
]

random_agent = random_agent.RandomAgent(player_id=0, num_actions=num_actions)

for cur_episode in range(int(1e6)):
    if cur_episode % int(5e3) == 0:
        win_rates = eval_against_random_bots(env, agents[0], 10000)
        print("episode " + str(cur_episode) + ": winrate " + str(win_rates))
    
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
