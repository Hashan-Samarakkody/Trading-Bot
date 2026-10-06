"""
IM07_sweep_swucb.py — compare Sliding Window UCB against current D-UCB

Sweeps tau in {30,40,50,60,70,80} and c in {0.005,0.01,0.02,0.05}
on both graded seeds and 40 held-out seeds.

Reference: Garivier & Moulines (2008) arXiv:0805.3415
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from bandit_env import TradingBanditEnv, STRATEGIES
from agent_base import BanditAgent

GRADED_SEEDS   = [7, 17, 27, 37, 47]
HOLDOUT_SEEDS  = list(range(200_000, 200_040))
N_PULLS        = 200


class SWUCBAgent(BanditAgent):
    """Sliding-Window UCB (Garivier & Moulines 2008).

    Keeps last tau (arm, reward) pairs. Stats computed only from that window.
    Provably better than D-UCB when discount gamma is misconfigured; more
    robust because the window is a hard cut rather than exponential decay.
    """

    def __init__(self, n_arms, tau=60, c=0.01):
        super().__init__(n_arms)
        self.tau = tau
        self.c = c
        self.history = []   # list of (arm, reward) — rolling window

    def _window_stats(self):
        window = self.history[-self.tau:] if len(self.history) >= self.tau else self.history
        counts = np.zeros(self.n_arms)
        sums   = np.zeros(self.n_arms)
        for arm, reward in window:
            counts[arm] += 1
            sums[arm]   += reward
        return counts, sums

    def select_arm(self) -> int:
        counts, sums = self._window_stats()
        untried = np.flatnonzero(counts == 0)
        if untried.size > 0:
            return int(untried[0])
        t = len(self.history)
        means = sums / counts
        bonus = self.c * np.sqrt(np.log(min(t, self.tau) + 1) / counts)
        return int(np.argmax(means + bonus))

    def update(self, arm: int, reward: float) -> None:
        self.history.append((arm, reward))


def score_agent(agent_cls, seeds, kwargs=None):
    kwargs = kwargs or {}
    totals = []
    for seed in seeds:
        agent = agent_cls(n_arms=len(STRATEGIES), **kwargs)
        env = TradingBanditEnv(total_pulls=N_PULLS)
        env.reset(seed=seed)
        total = 0.0
        for _ in range(N_PULLS):
            arm = agent.select_arm()
            r   = env.pull(arm)
            agent.update(arm, r)
            total += r
        totals.append(total)
    return np.mean(totals), np.min(totals), totals


# current D-UCB baseline 
from IM07_agent_bandit import Agent as DUCBAgent   # noqa: E402
ducb_graded_mean, ducb_graded_min, ducb_graded_all = score_agent(DUCBAgent, GRADED_SEEDS)
ducb_hold_mean,   ducb_hold_min,   _               = score_agent(DUCBAgent, HOLDOUT_SEEDS)
print(f"D-UCB  graded mean={ducb_graded_mean:.4f}  min={ducb_graded_min:.4f}"
      f"  held-out mean={ducb_hold_mean:.4f}  min={ducb_hold_min:.4f}")
print()

# SW-UCB sweep 
TAU_GRID = [30, 40, 50, 60, 70, 80]
C_GRID   = [0.005, 0.01, 0.02, 0.05]

best = {"hold_mean": -999}
results = []

for tau in TAU_GRID:
    for c in C_GRID:
        kw = {"tau": tau, "c": c}
        gm, gmin, gall = score_agent(SWUCBAgent, GRADED_SEEDS,   kw)
        hm, hmin, _    = score_agent(SWUCBAgent, HOLDOUT_SEEDS,  kw)
        results.append((hm, gm, gmin, hmin, tau, c))
        marker = " <-- NEW BEST" if hm > best["hold_mean"] else ""
        if hm > best["hold_mean"]:
            best = {"hold_mean": hm, "tau": tau, "c": c,
                    "graded_mean": gm, "graded_min": gmin}
        print(f"tau={tau:2d}  c={c:.3f}  graded={gm:.4f}(min {gmin:.3f})"
              f"  held-out={hm:.4f}(min {hmin:.3f}){marker}")

print()
print(f"Best SW-UCB: tau={best['tau']} c={best['c']}"
      f"  graded={best['graded_mean']:.4f}  held-out={best['hold_mean']:.4f}")
print(f"D-UCB:                          graded={ducb_graded_mean:.4f}"
      f"  held-out={ducb_hold_mean:.4f}")

if best["hold_mean"] > ducb_hold_mean:
    print(f"\nSW-UCB wins on held-out by {best['hold_mean'] - ducb_hold_mean:+.4f}"
          f" — safe to replace D-UCB in submission.")
else:
    print(f"\nD-UCB holds. Keep current agent.")
