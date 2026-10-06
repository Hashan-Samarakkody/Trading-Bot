"""
IM07_plots.py  --  Task T11 (owner: R4 Experimental Scientist)
=============================================================

Renders the figures the specification requires in the final report:

  fig1_bandit_learning_curve.png   cumulative reward vs pull, Phase 1,
                                   against the oracle and best-fixed-arm lines
  fig2_bandit_arm_tracking.png     share of pulls per arm per epoch -- the
                                   evidence that the agent tracks the regimes
  fig3_mdp_learning_curve.png      per-episode return over 300 training
                                   episodes, Q-learning vs SARSA vs SARSA(lambda)
  fig4_mdp_generalisation.png      training vs held-out score per variant

Figure 2 is the important one. Lecture 6 decomposes regret as
L_t = sum_a E[N_t(a)] * delta_a, so the per-arm selection count IS the
quantity the theory is about -- and unlike a score, it can distinguish an
agent that adapts from one that got lucky.

Run:  python experiments/IM07_plots.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from IM07_agent_bandit import Agent as UCBAgent  # noqa: E402
from IM07_agent_mdp import QLearningAgent, SarsaAgent, SarsaLambdaAgent  # noqa: E402
from bandit_env import STRATEGIES, TradingBanditEnv  # noqa: E402
from baseline_agents import EpsilonGreedyBanditAgent, RandomBanditAgent  # noqa: E402
from eval_harness import (  # noqa: E402
    BANDIT_EVAL_SEEDS,
    BANDIT_N_PULLS,
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

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report")
os.makedirs(OUT, exist_ok=True)

BOUNDARIES = [80, 140]
ORACLE_PER_PULL = {0: 0.0187, 1: 0.0143, 2: 0.0000}
BEST_FIXED_TOTAL = 0.560


def phase_of(p):
    return 0 if p < BOUNDARIES[0] else (1 if p < BOUNDARIES[1] else 2)


def bandit_trace(agent_cls, seed, **kw):
    agent = agent_cls(n_arms=len(STRATEGIES), **kw)
    env = TradingBanditEnv(total_pulls=BANDIT_N_PULLS)
    env.reset(seed=seed)
    arms = np.empty(BANDIT_N_PULLS, dtype=int)
    rew = np.empty(BANDIT_N_PULLS)
    for t in range(BANDIT_N_PULLS):
        a = agent.select_arm()
        r = env.pull(a)
        agent.update(a, r)
        arms[t], rew[t] = a, r
    return arms, rew


def mark_switches(ax):
    for b in BOUNDARIES:
        ax.axvline(b, color="0.35", ls="--", lw=1)
    ax.text(40, ax.get_ylim()[1] * 0.96, "bull", ha="center", fontsize=9, color="0.3")
    ax.text(110, ax.get_ylim()[1] * 0.96, "choppy", ha="center", fontsize=9, color="0.3")
    ax.text(170, ax.get_ylim()[1] * 0.96, "bear", ha="center", fontsize=9, color="0.3")


def fig1_bandit_curve():
    agents = [
        ("Discounted UCB (submitted)", UCBAgent, {}, "C0"),
        ("epsilon-greedy fixed (baseline)", EpsilonGreedyBanditAgent, {}, "C3"),
        ("random (baseline)", RandomBanditAgent, {}, "C7"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))

    for label, cls, kw, colour in agents:
        curves = [np.cumsum(bandit_trace(cls, s, **kw)[1]) for s in BANDIT_EVAL_SEEDS]
        curves = np.array(curves)
        mean = curves.mean(axis=0)
        ax.plot(mean, label=label, color=colour, lw=2)
        ax.fill_between(range(BANDIT_N_PULLS), curves.min(axis=0), curves.max(axis=0),
                        color=colour, alpha=0.12)

    oracle = np.cumsum([ORACLE_PER_PULL[phase_of(p)] for p in range(BANDIT_N_PULLS)])
    ax.plot(oracle, color="k", ls=":", lw=1.5, label="oracle ceiling (2.356)")
    ax.axhline(BEST_FIXED_TOTAL, color="C2", ls="-.", lw=1.2,
               label=f"best FIXED arm ({BEST_FIXED_TOTAL})")

    ax.set_xlabel("pull")
    ax.set_ylabel("cumulative reward")
    ax.set_title("Phase 1: cumulative reward (mean of 5 scored seeds, band = min-max)")
    mark_switches(ax)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT, "fig1_bandit_learning_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig2_arm_tracking():
    epoch = 20
    n_epochs = BANDIT_N_PULLS // epoch
    counts = np.zeros((n_epochs, len(STRATEGIES)))
    for s in BANDIT_EVAL_SEEDS:
        arms, _ = bandit_trace(UCBAgent, s)
        for t, a in enumerate(arms):
            counts[t // epoch, a] += 1
    counts /= counts.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(n_epochs)
    x = np.arange(n_epochs) * epoch + epoch / 2
    for a, name in enumerate(STRATEGIES):
        ax.bar(x, counts[:, a], width=epoch * 0.9, bottom=bottom, label=name)
        bottom += counts[:, a]

    for b in BOUNDARIES:
        ax.axvline(b, color="k", ls="--", lw=1.5)
    ax.set_xlabel("pull")
    ax.set_ylabel("share of pulls")
    ax.set_ylim(0, 1)
    ax.set_title("Phase 1: which arm the agent chooses, per 20-pull epoch\n"
                 "(dashed = measured regime switches at pulls 80 and 140)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT, "fig2_bandit_arm_tracking.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def train_mdp(cls, **kw):
    _pin_agent_randomness()
    env = TradingEnv(episode_length=MDP_EPISODE_LENGTH)
    agent = cls(TradingEnv.N_STATES, TradingEnv.N_ACTIONS, **kw)
    curve = []
    for ep in range(MDP_N_TRAIN_EPISODES):
        state = env.reset(seed=MDP_TRAIN_SEED_START + ep)
        agent.start_episode()
        done = False
        while not done:
            a = agent.act(state)
            nxt, r, done, _ = env.step(a)
            agent.update(state, a, r, nxt, done)
            state = nxt
        curve.append(env.portfolio_value - 1.0)
    agent.set_eval_mode(True)
    return agent, env, np.array(curve)


def score_block(env, agent, start):
    rets, dds = [], []
    for i in range(MDP_N_EVAL_EPISODES):
        s = env.reset(seed=start + i)
        done = False
        while not done:
            s, _, done, _ = env.step(agent.act(s))
        rets.append(env.portfolio_value - 1.0)
        dds.append(_max_drawdown(env.episode_summary().portfolio_values))
    return float(np.mean(rets)) - DRAWDOWN_PENALTY_LAMBDA * float(np.mean(dds))


def smooth(x, k=25):
    return np.convolve(x, np.ones(k) / k, mode="valid")


def fig3_and_4():
    variants = [
        ("Q-learning (submitted)", QLearningAgent, {}, "C0"),
        ("SARSA", SarsaAgent, {}, "C1"),
        ("SARSA(lambda=0.5)", SarsaLambdaAgent, {"lam": 0.5}, "C2"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    scores = {}
    for label, cls, kw, colour in variants:
        agent, env, curve = train_mdp(cls, **kw)
        ax.plot(np.arange(len(smooth(curve))) + 12, smooth(curve),
                label=label, color=colour, lw=2)
        scores[label] = (score_block(env, agent, MDP_TRAIN_SEED_START),
                         score_block(env, agent, MDP_EVAL_SEED_START))
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("training episode")
    ax.set_ylabel("episode return (25-episode moving average)")
    ax.set_title("Phase 2: training learning curves over 300 episodes")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p3 = os.path.join(OUT, "fig3_mdp_learning_curve.png")
    fig.savefig(p3, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(scores)
    xs = np.arange(len(labels))
    tr = [scores[k][0] for k in labels]
    ho = [scores[k][1] for k in labels]
    ax.bar(xs - 0.18, tr, 0.36, label="training seeds 0-49")
    ax.bar(xs + 0.18, ho, 0.36, label="held-out seeds 100000-100049")
    ax.axhline(0.0279, color="C3", ls="--", lw=1.2, label="untuned baseline 0.0279")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("score (return - 0.5 x drawdown)")
    ax.set_title("Phase 2: training vs held-out generalisation")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p4 = os.path.join(OUT, "fig4_mdp_generalisation.png")
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    return p3, p4, scores


def main():
    print("Rendering figures into report/ ...")
    print(f"  {os.path.basename(fig1_bandit_curve())}")
    print(f"  {os.path.basename(fig2_arm_tracking())}")
    p3, p4, scores = fig3_and_4()
    print(f"  {os.path.basename(p3)}")
    print(f"  {os.path.basename(p4)}")
    print()
    print("  Phase 2 scores used in fig 4:")
    for k, (tr, ho) in scores.items():
        print(f"    {k:<26} train {tr:+.4f}   held-out {ho:+.4f}   gap {tr - ho:+.4f}")


if __name__ == "__main__":
    main()
