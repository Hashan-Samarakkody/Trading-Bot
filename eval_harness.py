"""
eval_harness.py
===============
The fairness core of the whole project. Every submission -- baseline or
student -- is scored the exact same way:

Phase 1 (bandit):
    5 independent replicate runs (fresh agent, fresh environment each time)
    on 5 fixed seeds, N_PULLS pulls per replicate. Non-stationary within
    each replicate -- score is the average cumulative reward across the 5
    replicates (averaging over several fixed seeds, not just one, is what
    keeps this fair: see BANDIT_EVAL_SEEDS below).

Phase 2 (MDP / trading):
    1. TRAIN the submitted agent on a fixed sequence of training seeds
       (same seeds for every student -- nobody gets an easier training set).
    2. Freeze exploration (agent.set_eval_mode(True)).
    3. EVALUATE on a disjoint, held-out set of seeds the agent never trained
       on. This is the part that matters: it rewards agents that learned a
       genuinely good policy, not ones that overfit / memorized the training
       seeds.
    Score = mean excess return over the eval set, minus a drawdown penalty.
    The drawdown penalty exists so "always max leverage and hope" doesn't
    top the leaderboard -- risk-adjusted performance is the point.
"""

import random
import numpy as np
from trading_env import TradingEnv
from bandit_env import TradingBanditEnv

# ---- fixed configuration: DO NOT let students change these ---- #
# Averaged over several fixed seeds, not just one: a single 200-pull run is
# noisy enough that hyperparameter search can "beat" the true-best algorithm
# on one specific realization just by overfitting to that realization's luck
# (verified empirically while building this kit -- see the instructor guide).
# Averaging over multiple fixed seeds is the same fix Phase 2 already uses
# (50 eval episodes, not 1), applied to Phase 1.
BANDIT_EVAL_SEEDS = [7, 17, 27, 37, 47]
BANDIT_N_PULLS = 200

MDP_N_TRAIN_EPISODES = 300
MDP_TRAIN_SEED_START = 0          # seeds 0..299
MDP_N_EVAL_EPISODES = 50
MDP_EVAL_SEED_START = 100_000     # seeds 100000..100049, disjoint from training
MDP_EPISODE_LENGTH = 60
DRAWDOWN_PENALTY_LAMBDA = 0.5

# Reproducibility fix: student agents commonly seed their own randomness with
# `np.random.default_rng()` (no argument), which pulls OS entropy and is NOT
# affected by np.random.seed(). Left alone, that means identical code can
# score differently between two submissions -- not fair, and it invites
# "just resubmit until you get lucky" instead of building a robust agent.
# We fix this by forcing every `np.random.default_rng()` call made during
# evaluation to return a generator seeded the same way every time. This
# patches process-wide, which is safe here because every submission already
# runs in its own fresh subprocess (see run_submission.py).
AGENT_RNG_SEED = 12345
_ORIGINAL_DEFAULT_RNG = np.random.default_rng  # captured before any patching


def _pin_agent_randomness(seed: int = AGENT_RNG_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # Use the ORIGINAL default_rng, not np.random.default_rng, which may
    # already be patched from a previous call in this same process (e.g.
    # evaluating several agents back-to-back in one script). Deriving from
    # the original every time guarantees each call starts from a truly
    # fresh, identically-seeded generator rather than continuing to draw
    # from a generator some earlier evaluation already advanced.
    _pinned_generator = _ORIGINAL_DEFAULT_RNG(seed)

    # Only BARE `np.random.default_rng()` calls (no seed) get the pinned
    # generator. A call that passes an explicit seed must still honour it --
    # the environments seed themselves that way (`env.reset(seed=...)`), and
    # swallowing their seed would silently destroy the fixed train/eval seed
    # sets: every episode would instead be drawn from one shared stream that
    # the agent also draws from, so an agent making MORE random calls would
    # face a DIFFERENT market than one making fewer. Two agents with the same
    # policy must see the same prices; that is the whole point of the harness.
    def _default_rng(*args, **kwargs):
        if args or kwargs:
            return _ORIGINAL_DEFAULT_RNG(*args, **kwargs)
        return _pinned_generator

    np.random.default_rng = _default_rng


def run_bandit_eval(agent_class, agent_kwargs=None) -> dict:
    _pin_agent_randomness()
    agent_kwargs = agent_kwargs or {}
    from bandit_env import STRATEGIES

    per_seed_rewards = []
    for seed in BANDIT_EVAL_SEEDS:
        agent = agent_class(n_arms=len(STRATEGIES), **agent_kwargs)
        env = TradingBanditEnv(total_pulls=BANDIT_N_PULLS)
        env.reset(seed=seed)
        cumulative_reward = 0.0
        for _ in range(BANDIT_N_PULLS):
            arm = agent.select_arm()
            reward = env.pull(arm)
            agent.update(arm, reward)
            cumulative_reward += reward
        per_seed_rewards.append(cumulative_reward)

    mean_reward = float(np.mean(per_seed_rewards))
    return {
        "phase": "bandit",
        "per_seed_rewards": per_seed_rewards,
        "cumulative_reward": mean_reward,
        "score": mean_reward,
    }


def _max_drawdown(portfolio_values) -> float:
    values = np.array(portfolio_values)
    running_max = np.maximum.accumulate(values)
    drawdowns = 1.0 - values / running_max
    return float(drawdowns.max())


def run_mdp_eval(agent_class, agent_kwargs=None) -> dict:
    _pin_agent_randomness()
    agent_kwargs = agent_kwargs or {}
    env = TradingEnv(episode_length=MDP_EPISODE_LENGTH)
    agent = agent_class(TradingEnv.N_STATES, TradingEnv.N_ACTIONS, **agent_kwargs)

    # ---- 1. training on fixed seeds ----
    for ep in range(MDP_N_TRAIN_EPISODES):
        seed = MDP_TRAIN_SEED_START + ep
        state = env.reset(seed=seed)
        agent.start_episode()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

    # ---- 2. freeze exploration, evaluate on held-out seeds ----
    agent.set_eval_mode(True)
    final_returns = []
    max_drawdowns = []
    for ep in range(MDP_N_EVAL_EPISODES):
        seed = MDP_EVAL_SEED_START + ep
        state = env.reset(seed=seed)
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
        summary = env.episode_summary()
        final_returns.append(env.portfolio_value - 1.0)
        max_drawdowns.append(_max_drawdown(summary.portfolio_values))

    mean_return = float(np.mean(final_returns))
    mean_drawdown = float(np.mean(max_drawdowns))
    score = mean_return - DRAWDOWN_PENALTY_LAMBDA * mean_drawdown

    return {
        "phase": "mdp",
        "mean_return": mean_return,
        "mean_max_drawdown": mean_drawdown,
        "score": score,
    }


if __name__ == "__main__":
    from baseline_agents import (
        RandomBanditAgent, EpsilonGreedyBanditAgent,
        RandomMDPAgent, BuyAndHoldMDPAgent, QLearningReferenceAgent,
    )

    print("=== Phase 1: Bandit baselines ===")
    for cls in [RandomBanditAgent, EpsilonGreedyBanditAgent]:
        result = run_bandit_eval(cls)
        print(f"{cls.__name__:25s} score={result['score']:.4f}")

    print("\n=== Phase 2: MDP baselines ===")
    for cls in [RandomMDPAgent, BuyAndHoldMDPAgent, QLearningReferenceAgent]:
        result = run_mdp_eval(cls)
        print(f"{cls.__name__:25s} score={result['score']:.4f}  "
              f"(mean_return={result['mean_return']:.4f}, "
              f"mean_drawdown={result['mean_max_drawdown']:.4f})")
