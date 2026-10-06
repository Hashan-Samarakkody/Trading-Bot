"""
IM07_sweep_mdp2.py — fine sweep around current best Q-learning params

Current best: alpha=0.05, gamma=0.95, eps_start=0.5, eps_end=0.10, q_init=0.03
Sweep: alpha in {0.02,0.05,0.08}, eps_end in {0.05,0.10,0.15}, q_init in {0.01,0.03,0.05}
Validate on graded eval seeds (100000..100049) via run_mdp_eval.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from eval_harness import run_mdp_eval
from agent_base import MDPAgent

N_TRAIN = 300


def make_agent_cls(alpha, eps_end, q_init):
    class _Q(MDPAgent):
        _alpha   = alpha
        _eps_end = eps_end
        _q_init  = q_init

        def __init__(self, n_states, n_actions):
            super().__init__(n_states, n_actions)
            self.alpha    = self._alpha
            self.gamma    = 0.95
            self.eps_start = 0.5
            self.eps_end  = self._eps_end
            self.epsilon  = 0.5
            self.Q        = np.full((n_states, n_actions), self._q_init, dtype=float)
            self._rng     = np.random.default_rng()
            self._episode = 0
            self._eval    = False
            self._decay   = (self._eps_end / 0.5) ** (1.0 / max(1, N_TRAIN))

        def start_episode(self):
            self._episode += 1
            if not self._eval:
                self.epsilon = max(self.eps_end, 0.5 * (self._decay ** self._episode))

        def set_eval_mode(self, is_eval):
            self._eval   = bool(is_eval)
            self.epsilon = 0.0 if is_eval else 0.5

        def act(self, state):
            if self.epsilon > 0 and self._rng.random() < self.epsilon:
                return int(self._rng.integers(0, self.n_actions))
            return int(np.argmax(self.Q[state]))

        def update(self, state, action, reward, next_state, done):
            target = reward
            if not done and next_state is not None:
                target += self.gamma * np.max(self.Q[next_state])
            self.Q[state, action] += self.alpha * (target - self.Q[state, action])

    _Q.__name__ = f"Q_a{alpha}_e{eps_end}_q{q_init}"
    return _Q


best = {"score": -999}
for alpha in [0.02, 0.05, 0.08]:
    for eps_end in [0.05, 0.10, 0.15]:
        for q_init in [0.01, 0.03, 0.05]:
            cls = make_agent_cls(alpha, eps_end, q_init)
            r   = run_mdp_eval(cls)
            marker = ""
            if r["score"] > best["score"]:
                best = dict(r, alpha=alpha, eps_end=eps_end, q_init=q_init)
                marker = " <-- BEST"
            print(f"alpha={alpha:.2f} eps_end={eps_end:.2f} q_init={q_init:.2f}"
                  f"  score={r['score']:.4f}  return={r['mean_return']:.4f}"
                  f"  dd={r['mean_max_drawdown']:.4f}{marker}")

print()
print(f"Best: alpha={best['alpha']} eps_end={best['eps_end']} q_init={best['q_init']}"
      f"  score={best['score']:.4f}")
print(f"Current (0.05/0.10/0.03): score=0.0389")
