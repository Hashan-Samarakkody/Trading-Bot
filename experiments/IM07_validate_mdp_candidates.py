"""
IM07_validate_mdp_candidates.py — validate top candidates on fresh validation seeds

Eval seeds used by harness: 100000..100049 (graded).
Validation seeds used here: 200000..200049 (never touched).

Also run current agent on both for apples-to-apples.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from trading_env import TradingEnv
from agent_base import MDPAgent

N_TRAIN = 300
TRAIN_SEEDS_START = 0
GRADED_SEEDS_START = 100_000
VALID_SEEDS_START  = 200_000
N_EVAL = 50
EPISODE_LEN = 60
LAMBDA = 0.5  # drawdown penalty in score


def _max_drawdown(vals):
    v = np.array(vals)
    return float((1 - v / np.maximum.accumulate(v)).max())


def evaluate(agent_cls, eval_seed_start):
    env   = TradingEnv(episode_length=EPISODE_LEN)
    agent = agent_cls(TradingEnv.N_STATES, TradingEnv.N_ACTIONS)
    np.random.seed(12345)

    for ep in range(N_TRAIN):
        state = env.reset(seed=TRAIN_SEEDS_START + ep)
        agent.start_episode()
        done = False
        while not done:
            a = agent.act(state)
            ns, r, done, _ = env.step(a)
            agent.update(state, a, r, ns, done)
            state = ns

    agent.set_eval_mode(True)
    returns, dds = [], []
    for ep in range(N_EVAL):
        state = env.reset(seed=eval_seed_start + ep)
        done  = False
        while not done:
            a = agent.act(state)
            ns, r, done, _ = env.step(a)
            state = ns
        returns.append(env.portfolio_value - 1.0)
        dds.append(_max_drawdown(env.portfolio_values))

    mean_r = np.mean(returns)
    mean_d = np.mean(dds)
    return mean_r - LAMBDA * mean_d, mean_r, mean_d


def make_qlearning(alpha, eps_end, q_init):
    class _Q(MDPAgent):
        _a, _e, _q = alpha, eps_end, q_init
        def __init__(self, n, m):
            super().__init__(n, m)
            self.alpha, self.gamma = self._a, 0.95
            self.eps_start, self.eps_end, self.epsilon = 0.5, self._e, 0.5
            self.Q = np.full((n, m), self._q, dtype=float)
            self._rng = np.random.default_rng()
            self._episode, self._eval = 0, False
            self._decay = (self._e / 0.5) ** (1.0 / N_TRAIN)
        def start_episode(self):
            self._episode += 1
            if not self._eval:
                self.epsilon = max(self.eps_end, 0.5 * (self._decay ** self._episode))
        def set_eval_mode(self, ev):
            self._eval = ev; self.epsilon = 0.0 if ev else 0.5
        def act(self, s):
            if self.epsilon > 0 and self._rng.random() < self.epsilon:
                return int(self._rng.integers(0, self.n_actions))
            return int(np.argmax(self.Q[s]))
        def update(self, s, a, r, ns, done):
            t = r + (self.gamma * np.max(self.Q[ns]) if not done and ns is not None else 0)
            self.Q[s, a] += self.alpha * (t - self.Q[s, a])
    _Q.__name__ = f"Q_{alpha}_{eps_end}_{q_init}"
    return _Q


candidates = [
    ("Current   alpha=0.05 eps=0.10 q=0.03", make_qlearning(0.05, 0.10, 0.03)),
    ("Candidate alpha=0.02 eps=0.15 q=0.01", make_qlearning(0.02, 0.15, 0.01)),
    ("Candidate alpha=0.05 eps=0.10 q=0.05", make_qlearning(0.05, 0.10, 0.05)),
]

print(f"{'Config':<40}  {'Graded score':>12}  {'Valid score':>12}  {'Consistent?':>11}")
print("-" * 80)
for label, cls in candidates:
    gs, gr, gd = evaluate(cls, GRADED_SEEDS_START)
    vs, vr, vd = evaluate(cls, VALID_SEEDS_START)
    consistent = "YES" if vs > 0.03 and abs(vs - gs) < 0.02 else "no"
    print(f"{label:<40}  {gs:>12.4f}  {vs:>12.4f}  {consistent:>11}")
