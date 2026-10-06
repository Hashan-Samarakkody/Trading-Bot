"""
compare_mdp.py  --  Task T10, gates V6 / V7 (owner: R4 Experimental Scientist)
 =======

The required SARSA-vs-Q-learning comparison, plus the train/held-out
generalisation measurement.

The question, stated precisely
------------------------------
Lecture 5's cliff-walking example: Q-learning learns q*, the optimal
edge-hugging path, but behaves epsilon-greedily while learning and
occasionally falls off; SARSA evaluates the policy it actually follows, so
it learns the edge is dangerous FOR IT and keeps its distance. "Reward
while learning: SARSA wins. Policy learned: Q-learning wins."

This project is the same situation with two complications pulling in
opposite directions:

  * The score is  mean_return - 0.5 * mean_max_drawdown,  so it explicitly
    PAYS for the risk-aversion SARSA learns. That favours SARSA.

  * But the harness calls set_eval_mode(True), which sets epsilon to 0
    before scoring. The exploration that made SARSA cautious is switched
    off at exactly the moment it would be tested. Lecture 5's own closing
    note: "turn epsilon down over time and Q-learning wins on both counts."
    That favours Q-learning.

So which wins is a genuine empirical question on this environment rather
than a known result, and the honest thing to do is measure it.

What is reported
----------------
For every variant, on BOTH the training seeds (0..49, which the agent
trained on) and the held-out seeds (100000..100049, which it never saw):
mean return, mean max drawdown, and the final score.

The gap between the two is the overfitting evidence the specification asks
for. Drawdown is reported separately from return because the score mixes
them, and a change in score is uninformative unless you can see which term
moved.

Run:  python experiments/IM07_compare_mdp.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from IM07_agent_mdp import QLearningAgent, SarsaAgent, SarsaLambdaAgent  # noqa: E402
from baseline_agents import QLearningReferenceAgent  # noqa: E402
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

BASELINE_SCORE = 0.03


def evaluate_on(env, agent, seeds):
    """Score a trained, frozen agent over an arbitrary set of seeds."""
    returns, drawdowns = [], []
    for seed in seeds:
        state = env.reset(seed=seed)
        done = False
        while not done:
            action = agent.act(state)
            state, _, done, _ = env.step(action)
        summary = env.episode_summary()
        returns.append(env.portfolio_value - 1.0)
        drawdowns.append(_max_drawdown(summary.portfolio_values))
    mean_ret = float(np.mean(returns))
    mean_dd = float(np.mean(drawdowns))
    return mean_ret, mean_dd, mean_ret - DRAWDOWN_PENALTY_LAMBDA * mean_dd


def train_and_assess(agent_cls, **kwargs):
    """Train exactly as the harness does, then score on train and held-out seeds."""
    _pin_agent_randomness()
    env = TradingEnv(episode_length=MDP_EPISODE_LENGTH)
    agent = agent_cls(TradingEnv.N_STATES, TradingEnv.N_ACTIONS, **kwargs)

    t0 = time.time()
    learning_curve = []
    for ep in range(MDP_N_TRAIN_EPISODES):
        state = env.reset(seed=MDP_TRAIN_SEED_START + ep)
        agent.start_episode()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        learning_curve.append(env.portfolio_value - 1.0)
    train_time = time.time() - t0

    agent.set_eval_mode(True)

    # Same number of episodes on each side so the two are comparable.
    train_seeds = range(MDP_TRAIN_SEED_START, MDP_TRAIN_SEED_START + MDP_N_EVAL_EPISODES)
    held_seeds = range(MDP_EVAL_SEED_START, MDP_EVAL_SEED_START + MDP_N_EVAL_EPISODES)

    on_train = evaluate_on(env, agent, train_seeds)
    on_held = evaluate_on(env, agent, held_seeds)
    total_time = time.time() - t0
    return on_train, on_held, np.array(learning_curve), total_time


def main():
    variants = [
        ("QLearningReference (baseline)", QLearningReferenceAgent, {}),
        ("Q-learning (tuned)", QLearningAgent, {}),
        ("SARSA (tuned)", SarsaAgent, {}),
        ("SARSA(lambda=0.9)", SarsaLambdaAgent, {}),
    ]

    print("=" * 100)
    print("SARSA vs Q-learning, and train vs held-out generalisation")
    print("=" * 100)
    print(f"{'variant':<32}{'set':>10}{'return':>10}{'drawdown':>11}{'score':>9}")
    print("-" * 100)

    results = {}
    for label, cls, kwargs in variants:
        on_train, on_held, curve, elapsed = train_and_assess(cls, **kwargs)
        results[label] = (on_train, on_held, curve, elapsed)
        tr_r, tr_d, tr_s = on_train
        ho_r, ho_d, ho_s = on_held
        print(f"{label:<32}{'train':>10}{tr_r:10.4f}{tr_d:11.4f}{tr_s:9.4f}")
        print(f"{'':<32}{'held-out':>10}{ho_r:10.4f}{ho_d:11.4f}{ho_s:9.4f}"
              f"   gap={tr_s - ho_s:+.4f}   [{elapsed:.1f}s]")
        print()

    print("=" * 100)
    print("Gate V6: beat the untuned baseline on held-out seeds")
    print("=" * 100)
    base_held = results["QLearningReference (baseline)"][1][2]
    print(f"  baseline held-out score : {base_held:.4f}  (spec quotes ~{BASELINE_SCORE})")
    best_label, best_score = None, -np.inf
    for label, (_, on_held, _, _) in results.items():
        if label.startswith("QLearningReference"):
            continue
        score = on_held[2]
        status = "PASS" if score > base_held else "FAIL"
        print(f"  {label:<32} {score:7.4f}   {status}")
        if score > best_score:
            best_label, best_score = label, score

    print()
    print("=" * 100)
    print("The cliff-walking question, answered on this environment")
    print("=" * 100)
    q_ret, q_dd, q_sc = results["Q-learning (tuned)"][1]
    s_ret, s_dd, s_sc = results["SARSA (tuned)"][1]
    print(f"  Q-learning : return {q_ret:+.4f}   drawdown {q_dd:.4f}   score {q_sc:+.4f}")
    print(f"  SARSA      : return {s_ret:+.4f}   drawdown {s_dd:.4f}   score {s_sc:+.4f}")
    print()
    if s_dd < q_dd:
        print(f"  SARSA carries the LOWER drawdown ({s_dd:.4f} vs {q_dd:.4f}) -- the "
              f"cautious\n  policy the cliff-walking example predicts for an on-policy learner.")
    else:
        print(f"  Q-learning carries the lower drawdown ({q_dd:.4f} vs {s_dd:.4f}) -- the "
              f"cliff-walking\n  intuition does NOT transfer here, which is itself worth explaining.")
    print(f"  On the risk-adjusted score the winner is: "
          f"{'SARSA' if s_sc > q_sc else 'Q-learning'}")
    print()
    print(f"  Best overall variant: {best_label}  (held-out score {best_score:.4f})")
    print()
    print("  NOTE: set_eval_mode(True) zeroes epsilon before scoring, so whatever")
    print("  caution SARSA learned is measured with exploration switched off.")


if __name__ == "__main__":
    main()
