"""
IM07_test_mdp_remaining.py — test remaining Phase 2 improvements

Tests vs current Q-learning (score=0.0389):
  A. Expected SARSA
  B. Double Q-learning
  C. n-step Q-learning (n=2,3,4,6,8)
  D. SARSA(lambda) with replacing traces + lambda sweep
  E. Episode baseline subtraction (reward - episode_mean)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from eval_harness import run_mdp_eval
from agent_base import MDPAgent

N_TRAIN = 300


# ── shared base ───────────────────────────────────────────────────────────
class _Base(MDPAgent):
    def __init__(self, n_states, n_actions,
                 alpha=0.05, gamma=0.95, eps_start=0.5, eps_end=0.10, q_init=0.03):
        super().__init__(n_states, n_actions)
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end = eps_start, eps_end
        self.epsilon = eps_start
        self.Q = np.full((n_states, n_actions), q_init, dtype=float)
        self._rng = np.random.default_rng()
        self._episode, self._eval = 0, False
        self._decay = (eps_end / eps_start) ** (1.0 / N_TRAIN)

    def start_episode(self):
        self._episode += 1
        if not self._eval:
            self.epsilon = max(self.eps_end, self.eps_start * (self._decay ** self._episode))

    def set_eval_mode(self, ev):
        self._eval = ev
        self.epsilon = 0.0 if ev else self.eps_start

    def _eps_greedy(self, s):
        if self.epsilon > 0 and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[s]))

    def act(self, s):
        return self._eps_greedy(s)

    def update(self, s, a, r, ns, done):
        raise NotImplementedError


# ── A. Expected SARSA ────────────────────────────────────────────────────
class ExpectedSARSA(_Base):
    """target = R + gamma * E_pi[Q(S', .)]
    With epsilon-greedy: E = (1-eps)*max_a Q + eps/|A| * sum_a Q
    Lower variance than Q-learning; no maximisation bias unlike Q-learning.
    Lecture 5: Expected SARSA bridges Q-learning and SARSA.
    """

    def update(self, s, a, r, ns, done):
        target = r
        if not done and ns is not None:
            # Expected value under epsilon-greedy
            q_ns = self.Q[ns]
            best = float(np.max(q_ns))
            expected = (1 - self.epsilon) * best + (self.epsilon / self.n_actions) * float(q_ns.sum())
            target += self.gamma * expected
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])


# ── B. Double Q-learning ─────────────────────────────────────────────────
class DoubleQLearning(_Base):
    """Maintains Q_A and Q_B. Uses Q_A to select, Q_B to evaluate (and vice versa).
    Eliminates maximisation bias from Q-learning.
    Lecture 5: double estimator for unbiased target.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.QB = self.Q.copy()

    def act(self, s):
        # Act on average of both tables
        if self.epsilon > 0 and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[s] + self.QB[s]))

    def update(self, s, a, r, ns, done):
        if self._rng.random() < 0.5:
            # Update Q using QB to evaluate
            target = r
            if not done and ns is not None:
                best_a = int(np.argmax(self.Q[ns]))
                target += self.gamma * self.QB[ns, best_a]
            self.Q[s, a] += self.alpha * (target - self.Q[s, a])
        else:
            # Update QB using Q to evaluate
            target = r
            if not done and ns is not None:
                best_a = int(np.argmax(self.QB[ns]))
                target += self.gamma * self.Q[ns, best_a]
            self.QB[s, a] += self.alpha * (target - self.QB[s, a])


# ── C. n-step Q-learning ─────────────────────────────────────────────────
def make_nstep_q(n):
    class NStepQ(_Base):
        _n = n

        def __init__(self, ns_states, n_actions, **kwargs):
            super().__init__(ns_states, n_actions, **kwargs)
            self._buf = []   # (state, action, reward) tuples

        def start_episode(self):
            super().start_episode()
            self._buf = []

        def update(self, s, a, r, ns, done):
            self._buf.append((s, a, r))
            if len(self._buf) >= self._n or done:
                # Compute n-step return from oldest buffered transition
                G = sum(self.gamma**i * self._buf[i][2] for i in range(len(self._buf)))
                if not done and ns is not None:
                    G += self.gamma**len(self._buf) * np.max(self.Q[ns])
                s0, a0, _ = self._buf[0]
                self.Q[s0, a0] += self.alpha * (G - self.Q[s0, a0])
                self._buf.pop(0)

    NStepQ.__name__ = f"NStepQ_{n}"
    return NStepQ


# ── D. SARSA(lambda) with replacing traces ───────────────────────────────
def make_sarsa_lambda_replacing(lam):
    class SarsaLambdaReplacing(_Base):
        _lam = lam

        def __init__(self, ns_states, n_actions, **kwargs):
            super().__init__(ns_states, n_actions, **kwargs)
            self.E = np.zeros((ns_states, n_actions))
            self._pending = None

        def start_episode(self):
            super().start_episode()
            self.E.fill(0.0)
            self._pending = None

        def act(self, s):
            if self._pending is not None:
                a, self._pending = self._pending, None
                return a
            return self._eps_greedy(s)

        def update(self, s, a, r, ns, done):
            target = r
            if not done and ns is not None:
                next_a = self._eps_greedy(ns)
                self._pending = next_a
                target += self.gamma * self.Q[ns, next_a]
            delta = target - self.Q[s, a]
            # Replacing traces: set to 1 (not +=1) — caps at 1, reduces noise amplification
            self.E[s, a] = 1.0
            self.Q += self.alpha * delta * self.E
            self.E *= self.gamma * self._lam

    SarsaLambdaReplacing.__name__ = f"SarsaLambdaR_{lam}"
    return SarsaLambdaReplacing


# ── E. Episode baseline subtraction ─────────────────────────────────────
class BaselineQL(_Base):
    """Q-learning with per-episode reward baseline.
    Subtract running episode mean from each reward before the TD update.
    Lecture 4: baseline subtraction reduces variance without changing expected gradient.
    Helps agent distinguish signal from noise in 1:35 SNR environment.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ep_rewards = []
        self._ep_mean    = 0.0

    def start_episode(self):
        super().start_episode()
        self._ep_rewards = []
        self._ep_mean    = 0.0

    def update(self, s, a, r, ns, done):
        self._ep_rewards.append(r)
        self._ep_mean = np.mean(self._ep_rewards)
        shaped_r = r - self._ep_mean   # center reward around episode mean
        target = shaped_r
        if not done and ns is not None:
            target += self.gamma * np.max(self.Q[ns])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])


# ── run all ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"{'Variant':<40}  {'score':>7}  {'return':>8}  {'drawdown':>9}")
    print("-" * 70)

    def run(label, cls):
        r = run_mdp_eval(cls)
        star = " <-- BETTER" if r["score"] > 0.0389 else ""
        print(f"{label:<40}  {r['score']:>7.4f}  {r['mean_return']:>8.4f}"
              f"  {r['mean_max_drawdown']:>9.4f}{star}")
        return r["score"]

    run("Current Q-learning (baseline)", type("QL", (_Base,), {
        "update": lambda self, s, a, r, ns, done: (
            self.Q.__setitem__((s, a),
                self.Q[s, a] + self.alpha * (
                    (r + (self.gamma * np.max(self.Q[ns]) if not done and ns is not None else 0))
                    - self.Q[s, a]
                )
            )
        )
    }))

    print()
    run("A. Expected SARSA", ExpectedSARSA)
    print()
    run("B. Double Q-learning", DoubleQLearning)
    print()
    for n in [2, 3, 4, 6, 8]:
        run(f"C. n-step Q-learning  n={n}", make_nstep_q(n))
    print()
    for lam in [0.1, 0.3, 0.5, 0.7, 0.9]:
        run(f"D. SARSA(λ) replacing traces λ={lam}", make_sarsa_lambda_replacing(lam))
    print()
    run("E. Baseline subtraction Q-learning", BaselineQL)
