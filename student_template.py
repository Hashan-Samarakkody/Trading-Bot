"""
student_template.py
====================
Rename this file to agent.py (or whatever your submission filename is) and
fill in the TODOs. Your file must define a class named exactly `Agent`.

Delete whichever section (Phase 1 or Phase 2) doesn't apply to what you're
currently submitting -- keep only ONE `Agent` class per file.
"""

import numpy as np
from agent_base import BanditAgent, MDPAgent


# ===================================================================== #
# PHASE 1: Bandit agent (strategy selection, n_arms = 5)
# ===================================================================== #
class Agent(BanditAgent):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        # TODO: initialize whatever state your algorithm needs
        #  (e.g. counts, estimated values, UCB confidence terms,
        #   Thompson sampling priors, a sliding window of recent rewards...)
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self) -> int:
        # TODO: implement your action-selection rule here
        # (epsilon-greedy / UCB / Thompson sampling / sliding-window variant...)
        raise NotImplementedError

    def update(self, arm: int, reward: float) -> None:
        # TODO: update your value estimates given the observed reward
        raise NotImplementedError


# ===================================================================== #
# PHASE 2: MDP agent (tabular Q-learning / SARSA, n_states=30, n_actions=3)
# ===================================================================== #
# class Agent(MDPAgent):
#     def __init__(self, n_states, n_actions):
#         super().__init__(n_states, n_actions)
#         # TODO: initialize your Q-table and hyperparameters
#         self.Q = np.zeros((n_states, n_actions))
#
#     def act(self, state: int) -> int:
#         # TODO: epsilon-greedy (or other) action selection over self.Q[state]
#         raise NotImplementedError
#
#     def update(self, state, action, reward, next_state, done) -> None:
#         # TODO: Q-learning update (off-policy, uses max over next actions)
#         #    or SARSA update (on-policy, uses the actually-taken next action --
#         #    for SARSA you'll need to track the next action yourself)
#         raise NotImplementedError
#
#     def set_eval_mode(self, is_eval: bool) -> None:
#         # OPTIONAL: turn off exploration during evaluation, e.g.
#         # self.epsilon = 0.0 if is_eval else self._training_epsilon
#         pass
