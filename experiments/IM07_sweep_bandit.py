"""
sweep_bandit.py  --  Task T5 (owner: R4 Experimental Scientist)
==============================================================

Sweeps the three discounted-UCB parameters over the 5 fixed evaluation
seeds and reports MEAN and PER-SEED SPREAD, not the mean alone.

Why the spread matters (gate V8): the harness scores the mean of 5 seeds.
A configuration whose mean is good but whose worst seed is near zero has
not learned to track regimes -- it has got lucky on 4 of 5 realisations.
The project specification admits this risk outright, which is why it
averages 5 seeds rather than 1. Reporting min/max alongside the mean is
how we avoid presenting a lucky configuration as a real result.

Baselines to beat, measured in experiments/IM07_arm_ev.py:
    random baseline        0.340
    best FIXED arm         0.560   <-- the real bar
    oracle ceiling         2.356

Run:  python experiments/IM07_sweep_bandit.py
"""

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from IM07_agent_bandit import Agent  # noqa: E402
from eval_harness import run_bandit_eval  # noqa: E402

RANDOM_BASELINE = 0.340
BEST_FIXED_ARM = 0.560
ORACLE = 2.356

ALPHAS = [0.05, 0.10, 0.15, 0.25, 0.40]
GAMMAS = [0.90, 0.94, 0.97, 0.99, 1.00]
CS = [0.0, 0.01, 0.02, 0.05, 0.10]


def evaluate(alpha, gamma_d, c):
    """Score one configuration through the real harness."""
    result = run_bandit_eval(
        Agent, agent_kwargs={"alpha": alpha, "gamma_d": gamma_d, "c": c}
    )
    per_seed = np.array([float(x) for x in result["per_seed_rewards"]])
    return result["score"], per_seed


def main():
    rows = []
    total = len(ALPHAS) * len(GAMMAS) * len(CS)
    for i, (alpha, gamma_d, c) in enumerate(itertools.product(ALPHAS, GAMMAS, CS), 1):
        score, per_seed = evaluate(alpha, gamma_d, c)
        rows.append((score, per_seed.min(), per_seed.max(), alpha, gamma_d, c))
        print(f"\r  swept {i}/{total}", end="", flush=True)
    print()

    rows.sort(reverse=True)

    print()
    print("=" * 78)
    print("Top 15 configurations, ranked by mean score over the 5 fixed seeds")
    print("=" * 78)
    print(f"{'mean':>8}{'worst':>9}{'best':>9}   {'alpha':>6}{'gamma_d':>9}{'c':>7}   verdict")
    for score, lo, hi, alpha, gamma_d, c in rows[:15]:
        verdict = "beats fixed arm" if score > BEST_FIXED_ARM else "below fixed arm"
        if lo < 0.2:
            verdict += " (FRAGILE: one seed collapses)"
        print(f"{score:8.3f}{lo:9.3f}{hi:9.3f}   "
              f"{alpha:6.2f}{gamma_d:9.2f}{c:7.3f}   {verdict}")

    print()
    print("=" * 78)
    print("Most ROBUST configurations, ranked by worst-seed score")
    print("=" * 78)
    by_worst = sorted(rows, key=lambda r: r[1], reverse=True)
    print(f"{'mean':>8}{'worst':>9}{'best':>9}   {'alpha':>6}{'gamma_d':>9}{'c':>7}")
    for score, lo, hi, alpha, gamma_d, c in by_worst[:10]:
        print(f"{score:8.3f}{lo:9.3f}{hi:9.3f}   {alpha:6.2f}{gamma_d:9.2f}{c:7.3f}")

    best = rows[0]
    print()
    print("=" * 78)
    print("Context")
    print("=" * 78)
    print(f"  random baseline            {RANDOM_BASELINE:6.3f}")
    print(f"  best FIXED arm (buy_hold)  {BEST_FIXED_ARM:6.3f}   <-- the real bar")
    print(f"  best swept configuration   {best[0]:6.3f}   "
          f"(alpha={best[3]}, gamma_d={best[4]}, c={best[5]})")
    print(f"  oracle ceiling             {ORACLE:6.3f}")
    print(f"  fraction of oracle reached {best[0] / ORACLE:6.1%}")


if __name__ == "__main__":
    main()
