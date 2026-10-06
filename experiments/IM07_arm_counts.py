"""
arm_counts.py  --  Task T11 / gate V3 (owner: R4 Experimental Scientist)
=======================================================================

The single most important Phase 1 figure.

Lecture 6 decomposes regret as

    L_t = sum_a  E[N_t(a)] * delta_a

-- regret is nothing but counts multiplied by gaps, so "a good algorithm
ensures small counts for large gaps." That makes N_t(a) the quantity the
theory actually cares about, and it is directly observable.

So: bucket the run into 20-pull epochs and record which arm was selected in
each. If the agent is genuinely tracking the market, the modal arm must
change at the two measured regime boundaries:

    pull  80 : buy_hold       -> mean_reversion
    pull 140 : mean_reversion -> stay_cash

A good score alone cannot distinguish an adaptive agent from one that got
lucky. This plot can, which is why it is a release gate and not a nicety.

Run:  python experiments/IM07_arm_counts.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from IM07_agent_bandit import Agent  # noqa: E402
from bandit_env import STRATEGIES, TradingBanditEnv  # noqa: E402
from eval_harness import BANDIT_EVAL_SEEDS, BANDIT_N_PULLS  # noqa: E402

EPOCH = 20
# Measured in experiments/IM07_arm_ev.py.
PHASE_BOUNDARIES = [80, 140]
PHASE_BEST = {0: "buy_hold", 1: "mean_reversion", 2: "stay_cash"}


def phase_of(pull: int) -> int:
    if pull < PHASE_BOUNDARIES[0]:
        return 0
    if pull < PHASE_BOUNDARIES[1]:
        return 1
    return 2


def run_and_trace(seed: int):
    """One full replicate, recording the arm chosen at every pull."""
    agent = Agent(n_arms=len(STRATEGIES))
    env = TradingBanditEnv(total_pulls=BANDIT_N_PULLS)
    env.reset(seed=seed)
    chosen = np.empty(BANDIT_N_PULLS, dtype=int)
    rewards = np.empty(BANDIT_N_PULLS)
    for t in range(BANDIT_N_PULLS):
        arm = agent.select_arm()
        reward = env.pull(arm)
        agent.update(arm, reward)
        chosen[t] = arm
        rewards[t] = reward
    return chosen, rewards


def main():
    n_epochs = BANDIT_N_PULLS // EPOCH
    counts = np.zeros((n_epochs, len(STRATEGIES)))
    optimal_hits = np.zeros(BANDIT_N_PULLS)

    for seed in BANDIT_EVAL_SEEDS:
        chosen, _ = run_and_trace(seed)
        for t, arm in enumerate(chosen):
            counts[t // EPOCH, arm] += 1
            optimal_hits[t] += (STRATEGIES[arm] == PHASE_BEST[phase_of(t)])

    counts /= len(BANDIT_EVAL_SEEDS)          # avg selections per epoch per seed
    optimal_rate = optimal_hits / len(BANDIT_EVAL_SEEDS)

    print("=" * 92)
    print("Arm selections per 20-pull epoch, averaged over the 5 evaluation seeds")
    print("=" * 92)
    print(f"{'pulls':>10}{'phase':>7}  " + "".join(f"{s:>16s}" for s in STRATEGIES)
          + "   modal arm")
    print("-" * 92)
    for e in range(n_epochs):
        lo, hi = e * EPOCH, (e + 1) * EPOCH - 1
        phase = phase_of(lo)
        bar = "".join(f"{counts[e, a]:16.1f}" for a in range(len(STRATEGIES)))
        modal = STRATEGIES[int(np.argmax(counts[e]))]
        mark = "  <-- switch" if lo in PHASE_BOUNDARIES else ""
        ok = "OK " if modal == PHASE_BEST[phase] else "   "
        print(f"{lo:5d}-{hi:<4d}{phase:>7}  {bar}   {ok}{modal}{mark}")

    print()
    print("=" * 92)
    print("Gate V3: does the modal arm change at the measured regime boundaries?")
    print("=" * 92)
    modal_per_epoch = [STRATEGIES[int(np.argmax(counts[e]))] for e in range(n_epochs)]
    passed = True
    for phase in (0, 1, 2):
        epochs = [e for e in range(n_epochs) if phase_of(e * EPOCH) == phase]
        modals = [modal_per_epoch[e] for e in epochs]
        want = PHASE_BEST[phase]
        hit = sum(m == want for m in modals)
        status = "PASS" if hit > len(epochs) / 2 else "FAIL"
        passed &= status == "PASS"
        print(f"  phase {phase}: best arm = {want:<15s} "
              f"modal in {hit}/{len(epochs)} epochs   {status}")

    # Fraction of pulls spent on the currently-optimal arm, per phase.
    print()
    print("  Fraction of pulls on the phase-optimal arm:")
    for phase, (lo, hi) in enumerate([(0, 80), (80, 140), (140, 200)]):
        print(f"    phase {phase} (pulls {lo:3d}-{hi - 1:3d}): "
              f"{optimal_rate[lo:hi].mean():6.1%}")

    print()
    print(f"  GATE V3: {'PASS -- agent demonstrably tracks the regime switches' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
