"""
IM07_thompson_bandit.py  --  Task T6 (owner: R3, design by R1)

Lecture 6's principle 4 (probability matching), implemented as a discounted
Gaussian Thompson sampler, so the report can compare principles 3 and 4 on
the same environment rather than quoting the lecture's regret bounds and
stopping there.

Why a Gaussian posterior and not Beta
-------------------------------------
The lecture's worked example is Beta-Bernoulli, because its rewards are
wins and losses in {0,1}. Here a reward is a weekly return -- a real number
of either sign, roughly +-0.03. A Beta posterior cannot represent that. The
natural conjugate choice for a real-valued reward is a Gaussian posterior
over the mean, which is also what the lecture's "Bayesian UCB example:
independent Gaussians" slide assumes.

Why the posterior must be discounted
------------------------------------
Standard Thompson sampling accumulates sufficient statistics forever, so a
posterior sharpened over the bull phase would stay sharp -- and wrong --
through the choppy and bear phases. Lecture 6 notes the Bayesian failure
mode directly: "worse performance if [the prior] is not [accurate]. A
confident wrong prior can suppress the best arm for a long time." After a
regime switch, yesterday's correct posterior IS a confident wrong prior.

The same forgetting used in the UCB agent is therefore applied here: an
exponentially weighted mean, and an effective observation count that decays
globally on every pull. A decayed count widens the posterior, which restores
that arm's chance of winning a draw -- the probability-matching analogue of
UCB's growing bonus.

Expected comparison
-------------------
Lecture 6 rates Thompson above UCB in the stationary case: UCB is
"logarithmic", Thompson is "logarithmic -- optimal", attaining the
Lai-Robbins bound. Whether that ordering survives non-stationarity on a
200-pull budget is the question this script actually answers.

Run:  python experiments/IM07_thompson_bandit.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from IM07_agent_bandit import Agent as DiscountedUCBAgent  # noqa: E402
from agent_base import BanditAgent  # noqa: E402
from bandit_env import STRATEGIES, TradingBanditEnv  # noqa: E402
from baseline_agents import EpsilonGreedyBanditAgent, RandomBanditAgent  # noqa: E402
from eval_harness import BANDIT_EVAL_SEEDS, BANDIT_N_PULLS  # noqa: E402

BEST_FIXED_ARM = 0.560
ORACLE = 2.356
HOLDOUT_SEEDS = list(range(1000, 1040))


class DiscountedThompsonAgent(BanditAgent):
    """Gaussian Thompson sampling with a discounted posterior.

    Posterior over each arm's mean reward is N(mu_a, sigma0^2 / n_eff_a):
    the more (effective) observations an arm has, the tighter the belief.
    Sampling one value per arm and acting greedily on the sample IS the
    lecture's four-step recipe -- update, sample, evaluate, act greedily --
    with all the exploration living in the sampling step.
    """

    def __init__(self, n_arms, alpha=0.15, gamma_d=0.99, sigma0=0.03, prior_n=1.0):
        super().__init__(n_arms)
        self.alpha = alpha
        self.gamma_d = gamma_d
        self.sigma0 = sigma0          # scale of a weekly return
        self.prior_n = prior_n        # pseudo-count, keeps the prior proper
        self.mu = np.zeros(n_arms, dtype=float)
        self.n_eff = np.zeros(n_arms, dtype=float)
        self._rng = np.random.default_rng()

    def select_arm(self) -> int:
        # Posterior sd shrinks as evidence accumulates and grows again as the
        # discount erodes it -- the forgetting mechanism, seen from the
        # Bayesian side.
        sd = self.sigma0 / np.sqrt(self.n_eff + self.prior_n)
        sample = self._rng.normal(self.mu, sd)
        return int(np.argmax(sample))

    def update(self, arm: int, reward: float) -> None:
        self.n_eff *= self.gamma_d                      # forget, globally
        self.mu[arm] += self.alpha * (reward - self.mu[arm])
        self.n_eff[arm] += 1.0


def score(agent_cls, seeds, **kwargs):
    totals = []
    for seed in seeds:
        agent = agent_cls(n_arms=len(STRATEGIES), **kwargs)
        env = TradingBanditEnv(total_pulls=BANDIT_N_PULLS)
        env.reset(seed=seed)
        total = 0.0
        for _ in range(BANDIT_N_PULLS):
            arm = agent.select_arm()
            reward = env.pull(arm)
            agent.update(arm, reward)
            total += reward
        totals.append(total)
    return np.array(totals)


def main():
    print("=" * 94)
    print("Lecture 6 principles, compared on this environment")
    print("=" * 94)
    print(f"{'agent':<34}{'principle':>12}{'scored':>9}{'worst':>8}"
          f"{'held-out':>10}{'ho worst':>10}")
    print("-" * 94)

    rows = [
        ("RandomBanditAgent (baseline)", "--", RandomBanditAgent, {}),
        ("EpsilonGreedy fixed (baseline)", "1 naive", EpsilonGreedyBanditAgent, {}),
        ("Discounted UCB  [SUBMITTED]", "3 optimism", DiscountedUCBAgent, {}),
        ("Discounted Thompson", "4 prob.match", DiscountedThompsonAgent, {}),
    ]

    results = {}
    for label, principle, cls, kwargs in rows:
        scored = score(cls, BANDIT_EVAL_SEEDS, **kwargs)
        held = score(cls, HOLDOUT_SEEDS, **kwargs)
        results[label] = (scored, held)
        print(f"{label:<34}{principle:>12}{scored.mean():9.3f}{scored.min():8.3f}"
              f"{held.mean():10.3f}{held.min():10.3f}")

    print()
    print(f"  reference: random 0.338 | best FIXED arm {BEST_FIXED_ARM} | oracle {ORACLE}")

    ucb = results["Discounted UCB  [SUBMITTED]"]
    ths = results["Discounted Thompson"]

    print()
    print("=" * 94)
    print("Principle 3 vs principle 4")
    print("=" * 94)
    print(f"  Discounted UCB      scored {ucb[0].mean():6.3f}   held-out {ucb[1].mean():6.3f}"
          f"   worst held-out {ucb[1].min():6.3f}")
    print(f"  Discounted Thompson scored {ths[0].mean():6.3f}   held-out {ths[1].mean():6.3f}"
          f"   worst held-out {ths[1].min():6.3f}")
    print()
    winner = "Thompson" if ths[1].mean() > ucb[1].mean() else "UCB"
    print(f"  Better held-out mean: {winner}")
    print()
    print("  Lecture 6 rates Thompson above UCB in the STATIONARY case (it attains")
    print("  the Lai-Robbins bound, UCB only matches its order). Whether that")
    print("  ordering survives a 200-pull non-stationary run is what the numbers")
    print("  above decide -- and the held-out column, not the scored column, is")
    print("  the one to read, since the UCB parameters were tuned on the scored")
    print("  seeds and the Thompson ones were not.")


if __name__ == "__main__":
    main()
