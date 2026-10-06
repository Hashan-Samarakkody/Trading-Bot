"""
IM07_test_cusum_ducb.py — CUSUM change-detection layer on top of D-UCB

When CUSUM detects a regime shift in the reward stream, reset all D-UCB stats.
Uses CUSUM on the overall reward stream (not per-arm) to detect phase changes.

Sweeps CUSUM threshold h in {0.01, 0.02, 0.05, 0.10} and epsilon in {0.005, 0.01, 0.02}.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from bandit_env import TradingBanditEnv, STRATEGIES
from agent_base import BanditAgent

GRADED_SEEDS  = [7, 17, 27, 37, 47]
HOLDOUT_SEEDS = list(range(200_000, 200_040))
N_PULLS       = 200


class CUSUMDUCBAgent(BanditAgent):
    """D-UCB with CUSUM change-detection reset.

    CUSUM accumulates evidence of mean shift in the reward stream.
    When it exceeds threshold h, all D-UCB stats are reset and CUSUM restarted.
    This combines D-UCB's data efficiency with explicit change detection.
    """

    def __init__(self, n_arms, alpha=0.15, gamma_d=0.99, c=0.01, h=0.05, eps_cusum=0.01):
        super().__init__(n_arms)
        self.alpha    = alpha
        self.gamma_d  = gamma_d
        self.c        = c
        self.h        = h          # CUSUM detection threshold
        self.eps      = eps_cusum  # expected mean shift size

        self._reset_ducb()
        self._reset_cusum()

    def _reset_ducb(self):
        self.values     = np.zeros(self.n_arms)
        self.n_eff      = np.zeros(self.n_arms)
        self.raw_counts = np.zeros(self.n_arms, dtype=int)

    def _reset_cusum(self):
        # Two-sided CUSUM: track upward and downward shifts
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.reward_history = []

    def _update_cusum(self, reward):
        self.reward_history.append(reward)
        if len(self.reward_history) < 10:   # wait for baseline
            return False
        # Use mean of first 10 pulls as reference mu0
        mu0 = np.mean(self.reward_history[:10])
        self.cusum_pos = max(0, self.cusum_pos + reward - mu0 - self.eps)
        self.cusum_neg = max(0, self.cusum_neg - reward + mu0 - self.eps)
        return self.cusum_pos > self.h or self.cusum_neg > self.h

    def select_arm(self) -> int:
        untried = np.flatnonzero(self.raw_counts == 0)
        if untried.size > 0:
            return int(untried[0])
        total = self.n_eff.sum()
        bonus = self.c * np.sqrt(np.log(total + 1.0) / self.n_eff)
        return int(np.argmax(self.values + bonus))

    def update(self, arm: int, reward: float) -> None:
        self.n_eff *= self.gamma_d
        self.values[arm] += self.alpha * (reward - self.values[arm])
        self.n_eff[arm]  += 1.0
        self.raw_counts[arm] += 1

        if self._update_cusum(reward):
            self._reset_ducb()
            self._reset_cusum()


def score_agent(agent_cls, seeds, kwargs=None):
    kwargs = kwargs or {}
    totals = []
    for seed in seeds:
        agent = agent_cls(n_arms=len(STRATEGIES), **kwargs)
        env   = TradingBanditEnv(total_pulls=N_PULLS)
        env.reset(seed=seed)
        total = 0.0
        for _ in range(N_PULLS):
            arm = agent.select_arm()
            r   = env.pull(arm)
            agent.update(arm, r)
            total += r
        totals.append(total)
    return np.mean(totals), np.min(totals)


# D-UCB baseline
from IM07_agent_bandit import Agent as DUCBAgent   # noqa: E402
dg, _ = score_agent(DUCBAgent, GRADED_SEEDS)
dh, _ = score_agent(DUCBAgent, HOLDOUT_SEEDS)
print(f"D-UCB baseline  graded={dg:.4f}  held-out={dh:.4f}\n")

best = {"hold": -999}
for h in [0.01, 0.02, 0.05, 0.10]:
    for eps_c in [0.005, 0.01, 0.02]:
        kw = {"h": h, "eps_cusum": eps_c}
        gm, gmin = score_agent(CUSUMDUCBAgent, GRADED_SEEDS,  kw)
        hm, hmin = score_agent(CUSUMDUCBAgent, HOLDOUT_SEEDS, kw)
        marker = " <-- BEST" if hm > best["hold"] else ""
        if hm > best["hold"]:
            best = {"hold": hm, "graded": gm, "h": h, "eps": eps_c}
        print(f"h={h:.2f} eps={eps_c:.3f}  graded={gm:.4f}(min {gmin:.3f})"
              f"  held-out={hm:.4f}(min {hmin:.3f}){marker}")

print()
print(f"Best CUSUM-DUCB: h={best['h']} eps={best['eps']}"
      f"  graded={best['graded']:.4f}  held-out={best['hold']:.4f}")
print(f"D-UCB:           graded={dg:.4f}  held-out={dh:.4f}")
if best["hold"] > dh:
    print(f"\nCUSUM-DUCB wins on held-out by {best['hold'] - dh:+.4f}")
else:
    print(f"\nD-UCB holds on held-out by {dh - best['hold']:+.4f}")
