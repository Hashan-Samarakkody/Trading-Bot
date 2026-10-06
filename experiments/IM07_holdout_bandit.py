"""
holdout_bandit.py  --  Task T5b (owner: R4 Experimental Scientist)
=================================================================

experiments/IM07_sweep_bandit.py tuned three parameters on exactly the 5 seeds
the harness scores. That is a legitimate thing to do -- those are the seeds
we are graded on -- but it is NOT evidence that the configuration is good.
It may simply have fitted five particular realisations.

The project specification raises this concern itself: it says a single
200-pull run was noisy enough that "hyperparameter search can beat the
true-best algorithm on one specific realization just by overfitting to that
realization's luck", and that averaging 5 seeds is the fix. Five is better
than one; it is not infinity.

So this script re-scores the leading configurations on 40 seeds the sweep
never touched. If a configuration's held-out mean collapses relative to its
in-sample mean, it was overfit and must not be shipped.

This is the Phase 1 analogue of the Phase 2 train/held-out split, and it is
the evidence the report needs in order to claim the tuning was honest.

Run:  python experiments/IM07_holdout_bandit.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from IM07_agent_bandit import Agent  # noqa: E402
from bandit_env import STRATEGIES, TradingBanditEnv  # noqa: E402
from eval_harness import BANDIT_EVAL_SEEDS, BANDIT_N_PULLS  # noqa: E402

BEST_FIXED_ARM = 0.560
ORACLE = 2.356

# Seeds disjoint from BANDIT_EVAL_SEEDS = [7, 17, 27, 37, 47].
HOLDOUT_SEEDS = list(range(1000, 1040))

# (alpha, gamma_d, c, label) -- the leaders from sweep_bandit.py
CANDIDATES = [
    (0.15, 0.99, 0.010, "best in-sample mean"),
    (0.10, 0.97, 0.010, "2nd in-sample mean"),
    (0.10, 1.00, 0.020, "high worst-seed"),
    (0.40, 1.00, 0.050, "most robust in-sample"),
    (0.15, 0.99, 0.020, "runner-up, robust"),
    (0.10, 1.00, 0.000, "c=0, no exploration bonus"),
]


def score_on_seeds(alpha, gamma_d, c, seeds):
    """Replicate the harness loop exactly, but over an arbitrary seed list."""
    totals = []
    for seed in seeds:
        agent = Agent(n_arms=len(STRATEGIES), alpha=alpha, gamma_d=gamma_d, c=c)
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
    print("=" * 86)
    print("Out-of-sample validation of the swept parameters")
    print("=" * 86)
    print(f"in-sample seeds  : {BANDIT_EVAL_SEEDS}  (the 5 the harness scores)")
    print(f"held-out seeds   : {len(HOLDOUT_SEEDS)} seeds, "
          f"{HOLDOUT_SEEDS[0]}..{HOLDOUT_SEEDS[-1]}  (never used for tuning)")
    print()
    print(f"{'alpha':>6}{'gamma_d':>9}{'c':>7}"
          f"{'in-sample':>12}{'held-out':>11}{'ho worst':>10}{'gap':>9}   label")
    print("-" * 86)

    results = []
    for alpha, gamma_d, c, label in CANDIDATES:
        in_sample = score_on_seeds(alpha, gamma_d, c, BANDIT_EVAL_SEEDS)
        held_out = score_on_seeds(alpha, gamma_d, c, HOLDOUT_SEEDS)
        gap = in_sample.mean() - held_out.mean()
        results.append((held_out.mean(), alpha, gamma_d, c, label))
        print(f"{alpha:6.2f}{gamma_d:9.2f}{c:7.3f}"
              f"{in_sample.mean():12.3f}{held_out.mean():11.3f}"
              f"{held_out.min():10.3f}{gap:+9.3f}   {label}")

    results.sort(reverse=True)
    best_ho, alpha, gamma_d, c, label = results[0]

    print()
    print("=" * 86)
    print("Verdict")
    print("=" * 86)
    print(f"  best held-out configuration : alpha={alpha}, gamma_d={gamma_d}, c={c}")
    print(f"  held-out mean               : {best_ho:6.3f}")
    print(f"  best FIXED arm              : {BEST_FIXED_ARM:6.3f}")
    print(f"  oracle ceiling              : {ORACLE:6.3f}")
    print(f"  fraction of oracle          : {best_ho / ORACLE:6.1%}")
    print()
    print("  A small positive gap (in-sample slightly above held-out) is expected")
    print("  and harmless. A large gap, or a held-out mean below 0.560, would mean")
    print("  the sweep fitted the five scored seeds rather than the problem.")


if __name__ == "__main__":
    main()
