import enum

import numpy as np

import pyspiel

_NUM_PLAYERS = 2
_DEFAULT_PARAMS = {"max_game_length": 1}
_PAYOFF = [[0, -0.25, 0.5], [0.25, 0, -0.05], [-0.05, 0.05, 0]]

_GAME_TYPE = pyspiel.GameType(
    short_name="python_rock_paper_scissors",
    long_name="Python Rock-Paper-Scissors",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=False,
    provides_observation_tensor=False,
    provides_factored_observation_string=False,
    parameter_specification=_DEFAULT_PARAMS)


class Action(enum.IntEnum):
    ROCK = 0
    PAPER = 1
    SCISSOR = 2


class RPSGame(pyspiel.Game):
    """The game, from which states and observers can be made."""

    # pylint:disable=dangerous-default-value
    def __init__(self, params=_DEFAULT_PARAMS):
        max_game_length = params["max_game_length"]
        super().__init__(
            _GAME_TYPE,
            pyspiel.GameInfo(
                num_distinct_actions=3,
                num_players=2,
                min_utility=np.min(_PAYOFF) * max_game_length,
                max_utility=np.max(_PAYOFF) * max_game_length,
                utility_sum=0.0,
                max_game_length=max_game_length), params)

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return RPSSTATE(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        return RPSObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
            params)


class RPSSTATE(pyspiel.State):
    """Current state of the game."""

    def __init__(self, game):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self._game_over = False
        self._rewards = np.zeros(_NUM_PLAYERS)
        self._returns = np.zeros(_NUM_PLAYERS)

    # OpenSpiel (PySpiel) API functions are below. This is the standard set that
    # should be implemented by every simultaneous-move game with chance.

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        else:
            return pyspiel.PlayerId.SIMULTANEOUS

    def _legal_actions(self, player):
        """Returns a list of legal actions, sorted in ascending order."""
        assert player >= 0
        return [Action.ROCK, Action.PAPER, Action.SCISSOR]

    # def chance_outcomes(self):
    #     """Returns the possible chance outcomes and their probabilities."""
    #     assert self._is_chance
    #     return [(Chance.CONTINUE, 1 - self._termination_probability),
    #             (Chance.STOP, self._termination_probability)]

    # def _apply_action(self):
    #     """Applies the specified action to the state."""
    #     # This is not called at simultaneous-move states.
    #     assert not self._game_over
    #     self._game_over = True
    #     if self._current_iteration > self.get_game().max_game_length():
    #         self._game_over = True

    def _apply_actions(self, actions):
        """Applies the specified actions (per player) to the state."""
        assert not self._is_chance and not self._game_over
        self._is_chance = True
        self._rewards[0] = _PAYOFF[actions[0]][actions[1]]
        self._rewards[1] = _PAYOFF[actions[1]][actions[0]]
        self._returns += self._rewards
        self._game_over = True

    def _action_to_string(self, player, action):
        """Action -> string."""
        return Action(action).name

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._game_over

    def rewards(self):
        """Reward at the previous step."""
        return self._rewards

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        return self._returns

    def __str__(self):
        """String for debug purposes. No particular semantics are required."""
        return (f"p0:{self.action_history_string(0)} "
                f"p1:{self.action_history_string(1)}")

    def action_history_string(self, player):
        return "".join(
            self._action_to_string(pa.player, pa.action)[0]
            for pa in self.full_history()
            if pa.player == player)


class RPSObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, iig_obs_type, params):
        """Initializes an empty observation tensor."""
        assert not bool(params)
        self.iig_obs_type = iig_obs_type
        self.tensor = None
        self.dict = {}

    def set_from(self, state, player):
        pass

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        if self.iig_obs_type.public_info:
            return (f"us:{state.action_history_string(player)} "
                    f"op:{state.action_history_string(1 - player)}")
        else:
            return None


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, RPSGame)
