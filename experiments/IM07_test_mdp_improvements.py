"""
IM07_test_mdp_improvements.py — test drawdown shaping + regime-informed Q init

Compares four variants:
  A. Current Q-learning (baseline, 0.0389)
  B. + Drawdown reward shaping during training
  C. + Regime-informed Q init (breaks symmetry toward momentum-implied regime)
  D. + Both B and C

Uses same harness as eval_harness.py for fairness.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from eval_harness import run_mdp_eval
from agent_base import MDPAgent
from trading_env import TradingEnv

N_TRAIN = 300
N_EVAL  = 50


class _TabularBase(MDPAgent):
    def __init__(self, n_states, n_actions,
                 alpha=0.05, gamma=0.95,
                 eps_start=0.5, eps_end=0.10, q_init=0.03,
                 regime_init=False, drawdown_shape=False):
        super().__init__(n_states, n_actions)
        self.alpha      = alpha
        self.gamma      = gamma
        self.eps_start  = eps_start
        self.eps_end    = eps_end
        self.epsilon    = eps_start
        self.q_init     = q_init
        self.regime_init    = regime_init
        self.drawdown_shape = drawdown_shape

        self._rng     = np.random.default_rng()
        self._episode = 0
        self._eval    = False
        self._eps_decay = (eps_end / eps_start) ** (1.0 / max(1, N_TRAIN))

        # State encoding: state = m_bucket*6 + v_bucket*2 + position
        # m_bucket in [0..4], v_bucket in [0..2], position in [0,1]
        self.Q = np.full((n_states, n_actions), q_init, dtype=float)
        if regime_init:
            # Break symmetry: high momentum → prefer LONG(2), low → prefer FLAT(0)
            for m in range(5):
                for v in range(3):
                    for pos in range(2):
                        s = m * 6 + v * 2 + pos
                        if m >= 3:   # strong up momentum → lean long
                            self.Q[s, 2] += 0.01   # LONG
                            self.Q[s, 0] -= 0.01   # FLAT
                        elif m <= 1: # strong down momentum → lean flat
                            self.Q[s, 0] += 0.01   # FLAT
                            self.Q[s, 2] -= 0.01   # LONG

        # Drawdown tracking (training only)
        self._portfolio_value = 1.0
        self._running_max     = 1.0

    def start_episode(self):
        self._episode += 1
        if not self._eval:
            self.epsilon = max(self.eps_end,
                               self.eps_start * (self._eps_decay ** self._episode))
        self._portfolio_value = 1.0
        self._running_max     = 1.0

    def set_eval_mode(self, is_eval: bool):
        self._eval   = bool(is_eval)
        self.epsilon = 0.0 if is_eval else self.eps_start

    def _eps_greedy(self, state):
        if self.epsilon > 0.0 and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[state]))

    def act(self, state):
        return self._eps_greedy(state)

    def _shaped_reward(self, reward):
        if not self.drawdown_shape or self._eval:
            return reward
        self._portfolio_value *= max(1 + reward, 1e-9)
        self._running_max      = max(self._running_max, self._portfolio_value)
        dd = 1.0 - self._portfolio_value / self._running_max
        return reward - 0.5 * dd   # match graded score formula weight

    def update(self, state, action, reward, next_state, done):
        r = self._shaped_reward(reward)
        target = r
        if not done and next_state is not None:
            target += self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])


# Four variants as Agent classes for run_mdp_eval
class AgentA(_TabularBase):
    """Baseline Q-learning — no changes."""
    def __init__(self, n, m): super().__init__(n, m, regime_init=False, drawdown_shape=False)

class AgentB(_TabularBase):
    """+ Drawdown reward shaping."""
    def __init__(self, n, m): super().__init__(n, m, regime_init=False, drawdown_shape=True)

class AgentC(_TabularBase):
    """+ Regime-informed Q init."""
    def __init__(self, n, m): super().__init__(n, m, regime_init=True,  drawdown_shape=False)

class AgentD(_TabularBase):
    """+ Both drawdown shaping and regime-informed Q init."""
    def __init__(self, n, m): super().__init__(n, m, regime_init=True,  drawdown_shape=True)


if __name__ == "__main__":
    variants = [
        ("A  baseline Q-learning          ", AgentA),
        ("B  + drawdown shaping           ", AgentB),
        ("C  + regime Q init              ", AgentC),
        ("D  + both                       ", AgentD),
    ]
    for label, cls in variants:
        r = run_mdp_eval(cls)
        print(f"{label}  score={r['score']:.4f}"
              f"  return={r['mean_return']:.4f}"
              f"  drawdown={r['mean_max_drawdown']:.4f}")
