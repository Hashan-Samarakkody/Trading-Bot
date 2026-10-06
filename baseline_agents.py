"""
baseline_agents.py
==================
Reference agents. These serve two purposes:
  1. Worked examples of the Agent interface (agent_base.py) students must match.
  2. Permanent entries on the leaderboard as a "floor" -- everyone should be
     able to beat these once they've implemented a real algorithm. Deliberately
     left untuned (simple hyperparameters, no decay schedules) so there's
     headroom for students to improve on them.
"""

import numpy as np
from agent_base import BanditAgent, MDPAgent


# Phase 1 baselines
class RandomBanditAgent(BanditAgent):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self._rng = np.random.default_rng()

    def select_arm(self):
        return int(self._rng.integers(0, self.n_arms))

    def update(self, arm, reward):
        pass


class EpsilonGreedyBanditAgent(BanditAgent):
    """Simple, fixed epsilon (no decay) -- intentionally beatable."""

    def __init__(self, n_arms, epsilon: float = 0.1):
        super().__init__(n_arms)
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self._rng = np.random.default_rng()

    def select_arm(self):
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_arms))
        return int(np.argmax(self.values))

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


 
# Phase 2 baselines
 
class RandomMDPAgent(MDPAgent):
    def __init__(self, n_states, n_actions):
        super().__init__(n_states, n_actions)
        self._rng = np.random.default_rng()

    def act(self, state):
        return int(self._rng.integers(0, self.n_actions))

    def update(self, state, action, reward, next_state, done):
        pass


class BuyAndHoldMDPAgent(MDPAgent):
    """Always tries to be long. A common naive strategy -- should be beatable
    in bear/choppy regimes but strong in bull regimes."""

    def act(self, state):
        return 2  # always "buy/stay long"

    def update(self, state, action, reward, next_state, done):
        pass


class QLearningReferenceAgent(MDPAgent):
    """
    Working but UNTUNED Q-learning agent (fixed epsilon, fixed learning
    rate, no eligibility traces). This is what "implementing the interface
    correctly but not optimizing it" looks like -- a real submission should
    be able to beat this with epsilon decay, tuned alpha/gamma, or SARSA.
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1):
        super().__init__(n_states, n_actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))
        self._rng = np.random.default_rng()
        self._base_epsilon = epsilon

    def set_eval_mode(self, is_eval):
        # example use of the optional hook: go greedy during evaluation
        self.epsilon = 0.0 if is_eval else self._base_epsilon

    def act(self, state):
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target += self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
