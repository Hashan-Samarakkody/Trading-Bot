"""
arm_ev.py  --  Tasks T1 / T2 (owner: R2 Quant Analyst, verified by R4)

Before choosing a bandit algorithm we need to know what the game is worth.
This script answers three questions empirically, so that nothing in the
report rests on a formula derived by hand:

  Q1. At exactly which pull indices does the regime phase change?
  Q2. What is the expected reward of each arm, within each phase?
  Q3. What are the achievable ceilings over one 200-pull replicate --
      the oracle (always on the phase-best arm) and the best FIXED arm?

Q3 is the number that matters: it converts the published baseline score of
0.34 from "the target" into "the floor", and tells us what a genuinely good
Phase 1 agent should be scoring.

Run:  python experiments/IM07_arm_ev.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bandit_env import STRATEGIES, TradingBanditEnv  # noqa: E402
from eval_harness import BANDIT_N_PULLS  # noqa: E402

N_SAMPLES = 40_000   # pulls per (phase, arm) cell
EV_SEED = 20_251_005


def phase_of_pull(env: TradingBanditEnv, pull: int) -> int:
    """Replicate the env's own phase schedule for a given pull index."""
    n_epochs = max(1, env.total_pulls // env.epoch_length)
    epoch = (pull // env.epoch_length) % n_epochs
    return min(int(epoch / n_epochs * len(env.PHASE_MIXES)), len(env.PHASE_MIXES) - 1)


def find_phase_boundaries(env: TradingBanditEnv):
    """Q1: the pull indices at which the phase label changes."""
    schedule = [phase_of_pull(env, p) for p in range(env.total_pulls)]
    boundaries = [p for p in range(1, len(schedule)) if schedule[p] != schedule[p - 1]]
    spans = []
    start = 0
    for b in boundaries + [env.total_pulls]:
        spans.append((start, b - 1, schedule[start]))
        start = b
    return boundaries, spans


def measure_arm_ev(n_samples: int = N_SAMPLES):
    """Q2: mean reward of every arm inside every phase.

    The env advances `pull_count` on every pull, which would walk us out of
    the phase under test. We therefore pin `pull_count` to a representative
    pull of the target phase before each draw, so each sample is drawn from
    that phase's regime mix and nothing else.
    """
    env = TradingBanditEnv(total_pulls=BANDIT_N_PULLS)
    n_phases = len(env.PHASE_MIXES)
    _, spans = find_phase_boundaries(env)
    phase_start = {phase: lo for lo, _, phase in spans}

    means = np.zeros((n_phases, env.n_arms))
    stderrs = np.zeros((n_phases, env.n_arms))

    for phase in range(n_phases):
        pinned_pull = phase_start[phase]
        for arm in range(env.n_arms):
            env.reset(seed=EV_SEED + 1000 * phase + arm)
            rewards = np.empty(n_samples)
            for i in range(n_samples):
                env.pull_count = pinned_pull       # stay inside this phase
                rewards[i] = env.pull(arm)
            means[phase, arm] = rewards.mean()
            stderrs[phase, arm] = rewards.std(ddof=1) / np.sqrt(n_samples)

    return means, stderrs, spans


def ceilings(means: np.ndarray, spans):
    """Q3: expected cumulative reward over one 200-pull replicate."""
    oracle = 0.0
    per_arm_fixed = np.zeros(means.shape[1])
    for lo, hi, phase in spans:
        n = hi - lo + 1
        oracle += n * means[phase].max()
        per_arm_fixed += n * means[phase]
    return oracle, per_arm_fixed


def main():
    env = TradingBanditEnv(total_pulls=BANDIT_N_PULLS)
    boundaries, spans = find_phase_boundaries(env)

    print("=" * 74)
    print("Q1  Phase schedule over a 200-pull replicate")
    print("=" * 74)
    print(f"epoch_length={env.epoch_length}, total_pulls={env.total_pulls}")
    print(f"switch points (pull index): {boundaries}")
    for lo, hi, phase in spans:
        mix = env.PHASE_MIXES[phase]
        mix_s = " ".join(f"{k}={v:.2f}" for k, v in mix.items())
        print(f"  pulls {lo:3d}-{hi:3d}  ({hi - lo + 1:3d} pulls)  phase {phase}  {mix_s}")

    print()
    print("=" * 74)
    print(f"Q2  Expected reward per arm per phase  ({N_SAMPLES:,} samples per cell)")
    print("=" * 74)
    means, stderrs, spans = measure_arm_ev()
    header = "phase  " + "".join(f"{name:>17s}" for name in STRATEGIES)
    print(header)
    for phase in range(means.shape[0]):
        row = f"  {phase}    "
        for arm in range(means.shape[1]):
            row += f"{means[phase, arm]:+9.5f}+-{stderrs[phase, arm]:.5f}"
        best = STRATEGIES[int(np.argmax(means[phase]))]
        print(row + f"   best={best}")

    print()
    print("=" * 74)
    print("Q3  Achievable ceilings over one 200-pull replicate")
    print("=" * 74)
    oracle, per_arm_fixed = ceilings(means, spans)
    print(f"  Oracle (always on the phase-best arm) : {oracle:7.3f}")
    for arm, name in enumerate(STRATEGIES):
        print(f"  Fixed arm: {name:<15s}            : {per_arm_fixed[arm]:7.3f}")
    best_fixed_arm = int(np.argmax(per_arm_fixed))
    best_fixed = per_arm_fixed[best_fixed_arm]
    print(f"  Best FIXED arm ({STRATEGIES[best_fixed_arm]})           : {best_fixed:7.3f}")
    print(f"  Published random baseline                : {0.34:7.3f}")
    if oracle > 0:
        print(f"  best-fixed / oracle                      : {best_fixed / oracle:7.1%}")


if __name__ == "__main__":
    main()
