from open_spiel.python import rl_environment, policy

class NFSPPolicies(policy.Policy):
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