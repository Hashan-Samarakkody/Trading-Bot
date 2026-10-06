"""
diagnose_mdp.py  --  diagnosis for the V6 failure (owner: R1 + R3)

compare_mdp.py showed three things that need explaining before any tuning:

  1. SARSA(lambda) returns EXACTLY 0.0000 -- it never takes a position at
     all. An exact zero is a bug signature, not a bad hyperparameter.
  2. Tuned Q-learning (gamma=0.99, decaying epsilon, optimistic init) scores
     BELOW the untuned baseline (gamma=0.95, fixed epsilon=0.1, Q0=0).
  3. Every variant has a NEGATIVE train-minus-held-out gap, i.e. all of them
     do better on seeds they never trained on.

This script inspects the learned Q-tables and policies directly instead of
guessing, because each of the three has a different cause and tuning would
only mask them.

Run:  python experiments/IM07_diagnose_mdp.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from IM07_agent_mdp import QLearningAgent, SarsaAgent, SarsaLambdaAgent  # noqa: E402
from baseline_agents import QLearningReferenceAgent  # noqa: E402
from eval_harness import (  # noqa: E402
    MDP_EPISODE_LENGTH,
    MDP_N_TRAIN_EPISODES,
    MDP_TRAIN_SEED_START,
    _pin_agent_randomness,
)
from trading_env import TradingEnv  # noqa: E402

ACTIONS = ["flat", "hold", "long"]


def train(agent_cls, **kwargs):
    _pin_agent_randomness()
    env = TradingEnv(episode_length=MDP_EPISODE_LENGTH)
    agent = agent_cls(TradingEnv.N_STATES, TradingEnv.N_ACTIONS, **kwargs)
    visits = np.zeros((TradingEnv.N_STATES, TradingEnv.N_ACTIONS))
    for ep in range(MDP_N_TRAIN_EPISODES):
        state = env.reset(seed=MDP_TRAIN_SEED_START + ep)
        agent.start_episode()
        done = False
        while not done:
            action = agent.act(state)
            visits[state, action] += 1
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
    return agent, visits


def describe(label, agent, visits):
    Q = agent.Q
    greedy = Q.argmax(axis=1)
    counts = np.bincount(greedy, minlength=3)
    seen = visits.sum(axis=1) > 0

    print(f"--- {label}")
    print(f"    Q range            : [{Q.min():+.5f}, {Q.max():+.5f}]")
    print(f"    Q finite           : {np.isfinite(Q).all()}")
    print(f"    greedy policy      : flat={counts[0]:2d}  hold={counts[1]:2d}  long={counts[2]:2d}"
          f"   (of {TradingEnv.N_STATES} states)")
    print(f"    states ever visited: {int(seen.sum())}/{TradingEnv.N_STATES}")
    # Spread between best and worst action, averaged over visited states --
    # if this is ~0 the argmax is meaningless and ties go to action 0 (flat).
    spread = (Q.max(axis=1) - Q.min(axis=1))[seen]
    print(f"    mean action spread : {spread.mean():.3e}   (if ~0, argmax is a tie -> always flat)")
    print(f"    final epsilon      : {getattr(agent, 'epsilon', float('nan')):.4f}")
    print()


def main():
    print("=" * 84)
    print("1. What did each agent actually learn?")
    print("=" * 84)
    for label, cls in [
        ("QLearningReference (baseline)", QLearningReferenceAgent),
        ("Q-learning (tuned)", QLearningAgent),
        ("SARSA (tuned)", SarsaAgent),
        ("SARSA(lambda)", SarsaLambdaAgent),
    ]:
        agent, visits = train(cls)
        describe(label, agent, visits)

    print("=" * 84)
    print("2. Reward scale -- is the optimistic Q0 on the right scale?")
    print("=" * 84)
    env = TradingEnv(episode_length=MDP_EPISODE_LENGTH)
    rewards = []
    for ep in range(30):
        env.reset(seed=MDP_TRAIN_SEED_START + ep)
        done = False
        while not done:
            _, r, done, _ = env.step(2)  # always long: the largest-magnitude rewards
            rewards.append(r)
    rewards = np.array(rewards)
    print(f"    per-step reward: mean={rewards.mean():+.6f}  sd={rewards.std():.6f}  "
          f"max|r|={np.abs(rewards).max():.6f}")
    for gamma in (0.95, 0.99):
        horizon = 1.0 / (1.0 - gamma)
        print(f"    gamma={gamma}: effective horizon {horizon:5.1f} steps, "
              f"plausible |Q| ~ {abs(rewards.mean()) * min(horizon, 60):.5f}")
    print()
    print("    A sensible optimistic Q0 must sit just ABOVE the plausible |Q|,")
    print("    not orders of magnitude above it (which never washes out) and not")
    print("    below it (which is not optimistic at all).")


if __name__ == "__main__":
    main()
