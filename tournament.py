import importlib.util
import itertools
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
import random

import pyspiel
from open_spiel.python.algorithms.evaluate_bots import evaluate_bots

logger = logging.getLogger('be.kuleuven.cs.dtai.fcpa.tournament')


def load_agent(path, player_id):
    """Inintialize an agent from a 'fcpa_agent.py' file.
    """
    module_dir = os.path.dirname(os.path.abspath(path))
    sys.path.insert(1, module_dir)
    spec = importlib.util.spec_from_file_location("fcpa_agent", path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo.get_agent_for_tournament(player_id)


def load_agent_from_dir(agent_id, path):
    """Scrapes a directory for an fcpa agent.
    This function searches all subdirectories for an 'fcpa_agent.py' file and
    calls the ``get_agent_for_tournament`` method to create an instance of
    a player 1 and player 2 agent. If multiple matching files are found,
    a random one will be used.
    """
    agent_path = next(Path(path).glob('**/fcpa_agent.py'))
    try:
        return {
            'id':  agent_id,
            'agent_p1': load_agent(agent_path, 0),
            'agent_p2': load_agent(agent_path, 1),
        }
    except Exception as e:
        logger.exception("Failed to load %s" % agent_id)


def play_match(game, agent1, agent2, seed=1234, rounds=100):
    """Play a set of games between two agents.
    """
    rng = np.random.RandomState(seed)
    results = []
    for _ in tqdm(range(rounds)):
        # Alternate between the two agents as p1 and p2
        for (p1, p2) in [(agent1, agent2), (agent2, agent1)]:
            try:
                returns = evaluate_bots(
                        game.new_initial_state(),
                        [p1['agent_p1'], p2['agent_p2']],
                        rng)
                error = None
            except Exception as ex:
                logger.exception("Failed to play between %s and %s" % (agent1['id'], agent2['id']))
                template = "An exception of type {0} occurred. Message: {1}"
                error = template.format(type(ex).__name__, ex)
                returns = [None, None]
            finally:
                results.append({
                    "agent_p1": p1['id'],
                    "agent_p2": p2['id'],
                    "return_p1": returns[p2!=agent2],
                    "return_p2": returns[p2==agent2],
                    "error": error
                })
    return results


def play_tournament(game, agents, seed=1234, rounds=100):
    """Play a round robin tournament between multiple agents.
    """
    rng = np.random.RandomState(seed)
    # Load each team's agent
    results = []
    for _ in tqdm(range(rounds)):
        for (agent1, agent2) in list(itertools.permutations(agents.keys(), 2)):
            returns = evaluate_bots(
                    game.new_initial_state(), 
                    [agents[agent1]['agent_p1'], agents[agent2]['agent_p2']], 
                    rng)
            results.append({
                "agent_p1": agent1,
                "agent_p2": agent2,
                "return_p1": returns[0],
                "return_p2": returns[1]
            })
    return results


def cli(agent1_id, agent1_dir, agent2_id, agent2_dir, output, rounds, seed):
    """Play a set of games between two agents."""
    logging.basicConfig(level=logging.INFO)

    # Create the game
    fcpa_game_string = (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=150 100,"
        "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
        "stack=20000 20000,bettingAbstraction=fcpa)")
    logger.info("Creating game: {}".format(fcpa_game_string))
    game = pyspiel.load_game(fcpa_game_string)
    # Load the agents
    logger.info("Loading the agents")
    agent1 = load_agent_from_dir(agent1_id, agent1_dir)
    agent2 = load_agent_from_dir(agent2_id, agent2_dir)
    # Play the tournament
    logger.info("Playing {} matches between {} and {}".format(rounds, agent1_id,  agent2_id))
    results = play_match(game, agent1, agent2, seed, rounds)
    printResultsOverview(results)
    # Process the results
    logger.info("Processing the results")
    results = pd.DataFrame(results)
    results.to_csv(output, index=False)
    logger.info("Done. Results saved to {}".format(output))


def printResultsOverview(results):
    # print("\n".join([str(x) for x in results]))
    reward_p1 = [result['return_p1'] for result in results]
    reward_p2 = [result['return_p2'] for result in results]
    logger.info("Player 0 won %s of the games", 100*len([x for x in reward_p1 if x > 0])/len(results))
    logger.info("Average reward for player 0 was %s", sum(reward_p1)/len(results))


if __name__ == '__main__':
    agent1 = 'custom'
    agent2 = 'random'

    agent1_dir = './bots/' + agent1 + '/'
    agent2_dir = './bots/' + agent2 + '/'

    output = 'output.csv'
    rounds = 1
    seed = int(random.random()*100000)
    sys.exit(cli(agent1, agent1_dir, agent2, agent2_dir, output, rounds, seed))