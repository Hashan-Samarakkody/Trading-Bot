"""
sweep_mdp.py  --  Task T7b (owner: R4, design by R1)
====================================================

Phase 2 hyperparameter search, with one methodological rule:

    TUNE ON SEEDS 200000..200049.
    The graded seeds (100000..100049) are never used to choose anything.

The harness scores seeds 100000..100049. Tuning directly on those would make
the reported held-out score meaningless -- it would no longer be held out.
So a third, disjoint seed block is used as a validation set, and the graded
block is touched only once, at the end, to report the final number. This is
the same discipline the specification applies to the agent (train on 0..299,
score on 100000+), applied one level up to the hyperparameters.

What the diagnosis established (experiments/IM07_diagnose_mdp.py)
-----------------------------------------------------------
  * Per-step reward: mean -0.0004, sd 0.0142. Signal-to-noise about 1:35,
    so the reward is overwhelmingly noise and exploration is expensive.
  * Plausible |Q|: about 0.008 at gamma=0.95, about 0.024 at gamma=0.99.
    The first attempt used q_init = 0.01, which is BELOW the plausible Q at
    gamma=0.99 -- pessimistic, not optimistic. The rescaling was applied in
    the wrong direction.
  * SARSA(lambda) learned "stay flat" in 19/30 states: not a tie, a genuinely
    over-cautious policy. Accumulating traces multiply the effective step
    size, so it needs a smaller alpha than the one-step methods.

The grid below is built around those three facts rather than around round
numbers.

Run:  python experiments/IM07_sweep_mdp.py
"""

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from IM07_agent_mdp import QLearningAgent, SarsaAgent, SarsaLambdaAgent  # noqa: E402
from eval_harness import (  # noqa: E402
    DRAWDOWN_PENALTY_LAMBDA,
    MDP_EPISODE_LENGTH,
    MDP_EVAL_SEED_START,
    MDP_N_EVAL_EPISODES,
    MDP_N_TRAIN_EPISODES,
    MDP_TRAIN_SEED_START,
    _max_drawdown,
    _pin_agent_randomness,
)
from trading_env import TradingEnv  # noqa: E402

VALIDATION_SEED_START = 200_000          # disjoint from train AND from graded
BASELINE_HELD_OUT = 0.0279               # QLearningReferenceAgent, measured

GAMMAS = [0.95, 0.99]
ALPHAS = [0.05, 0.10, 0.20]
EPS_STARTS = [0.2, 0.5, 1.0]
EPS_ENDS = [0.01, 0.10]
Q_INITS = [0.0, 0.03]


def score_on(env, agent, seed_start, n_episodes):
    returns, drawdowns = [], []
    for i in range(n_episodes):
        state = env.reset(seed=seed_start + i)
        done = False
        while not done:
            state, _, done, _ = env.step(agent.act(state))
        summary = env.episode_summary()
        returns.append(env.portfolio_value - 1.0)
        drawdowns.append(_max_drawdown(summary.portfolio_values))
    mean_ret = float(np.mean(returns))
    mean_dd = float(np.mean(drawdowns))
    return mean_ret, mean_dd, mean_ret - DRAWDOWN_PENALTY_LAMBDA * mean_dd


def train_once(agent_cls, **kwargs):
    _pin_agent_randomness()
    env = TradingEnv(episode_length=MDP_EPISODE_LENGTH)
    agent = agent_cls(TradingEnv.N_STATES, TradingEnv.N_ACTIONS, **kwargs)
    for ep in range(MDP_N_TRAIN_EPISODES):
        state = env.reset(seed=MDP_TRAIN_SEED_START + ep)
        agent.start_episode()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
    agent.set_eval_mode(True)
    return env, agent


def sweep(agent_cls, label, extra_grid=None):
    grid = list(itertools.product(GAMMAS, ALPHAS, EPS_STARTS, EPS_ENDS, Q_INITS))
    extras = extra_grid or [{}]
    rows = []
    total = len(grid) * len(extras)
    done_n = 0
    for (gamma, alpha, eps_s, eps_e, q0), extra in itertools.product(grid, extras):
        kwargs = dict(gamma=gamma, alpha=alpha, eps_start=eps_s,
                      eps_end=eps_e, q_init=q0, **extra)
        env, agent = train_once(agent_cls, **kwargs)
        val = score_on(env, agent, VALIDATION_SEED_START, MDP_N_EVAL_EPISODES)
        rows.append((val[2], val[0], val[1], kwargs))
        done_n += 1
        print(f"\r  {label}: {done_n}/{total}", end="", flush=True)
    print()
    rows.sort(key=lambda r: r[0], reverse=True)
    return rows


def report(label, rows, n=5):
    print()
    print(f"  Top {n} for {label} (scored on VALIDATION seeds 200000+)")
    print(f"  {'score':>8}{'return':>9}{'drawdown':>10}   params")
    for score, ret, dd, kwargs in rows[:n]:
        ps = " ".join(f"{k}={v}" for k, v in kwargs.items())
        print(f"  {score:8.4f}{ret:9.4f}{dd:10.4f}   {ps}")
    return rows[0]


def main():
    print("=" * 96)
    print("Phase 2 hyperparameter sweep")
    print("=" * 96)
    print(f"  train seeds      : {MDP_TRAIN_SEED_START}..{MDP_TRAIN_SEED_START + MDP_N_TRAIN_EPISODES - 1}")
    print(f"  VALIDATION seeds : {VALIDATION_SEED_START}..{VALIDATION_SEED_START + MDP_N_EVAL_EPISODES - 1}  (tuning)")
    print(f"  graded seeds     : {MDP_EVAL_SEED_START}..{MDP_EVAL_SEED_START + MDP_N_EVAL_EPISODES - 1}  (touched once, at the end)")
    print(f"  baseline to beat : {BASELINE_HELD_OUT:.4f}")
    print()

    best = {}
    for label, cls, extra in [
        ("Q-learning", QLearningAgent, None),
        ("SARSA", SarsaAgent, None),
        # SARSA(lambda) needs a smaller alpha: traces multiply the effective
        # step size, which is why it over-corrected into an all-flat policy.
        ("SARSA(lambda)", SarsaLambdaAgent, [{"lam": 0.5}, {"lam": 0.9}]),
    ]:
        rows = sweep(cls, label, extra)
        best[label] = report(label, rows)

    print()
    print("=" * 96)
    print("Final: best-by-validation configurations, scored ONCE on the graded seeds")
    print("=" * 96)
    print(f"  {'variant':<16}{'validation':>12}{'GRADED':>10}{'return':>9}{'drawdn':>9}   params")
    for label, cls in [("Q-learning", QLearningAgent),
                       ("SARSA", SarsaAgent),
                       ("SARSA(lambda)", SarsaLambdaAgent)]:
        val_score, _, _, kwargs = best[label]
        env, agent = train_once(cls, **kwargs)
        graded = score_on(env, agent, MDP_EVAL_SEED_START, MDP_N_EVAL_EPISODES)
        ps = " ".join(f"{k}={v}" for k, v in kwargs.items())
        flag = "  PASS" if graded[2] > BASELINE_HELD_OUT else "  below baseline"
        print(f"  {label:<16}{val_score:12.4f}{graded[2]:10.4f}{graded[0]:9.4f}"
              f"{graded[1]:9.4f}   {ps}{flag}")


if __name__ == "__main__":
    main()
