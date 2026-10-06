"""
bandit_env.py
=============
Phase 1 of the semester: a non-stationary K-armed bandit built on the SAME
market simulator as the full trading environment (trading_env.py), so the
concepts transfer directly into Phase 2.

Framing: each "arm" is a fixed, pre-built trading strategy (not a raw
action). The student's bandit agent doesn't trade directly -- it repeatedly
chooses WHICH strategy to deploy for one trading week. This mirrors a real
capital-allocation problem: "which of my K models do I trust this week?"

Each pull simulates TWO consecutive weeks under one regime draw:
  - a LOOKBACK week, which is all the strategy is allowed to see, and
  - a TRADING week, whose return is the reward.
The strategy commits to a position from the lookback week alone and is then
paid on the week that follows. This is deliberate: if a strategy were
allowed to decide using the same week it gets paid on, "go long iff this
week went up" would be a risk-free money machine and the bandit would
collapse to "always pull that arm" (no exploration problem left to teach).

Non-stationarity: the market walks through three PHASES over the run --
bull-heavy, then choppy/range-bound, then bear-heavy -- stepping at
`epoch_length` boundaries. Each phase has a DIFFERENT best arm, and each
phase badly punishes a different one:

    phase        best arm        worst arm
    bull-heavy   buy_hold        stay_cash  (0, while others earn)
    choppy       mean_reversion  momentum   (loses what it chases)
    bear-heavy   stay_cash       buy_hold   (rides the whole fall)

No single arm is anywhere near optimal across the whole run: the best fixed
arm earns roughly a quarter of what an agent that tracks the switches can.
That is the point of the phase -- an agent that locks in on whatever looked
best in the first 40 pulls gets punished, while one that keeps exploring
(sliding-window UCB, discounted Thompson sampling, an epsilon that never
fully decays) can follow the market.

The choppy regime is genuinely range-bound: a week that runs up tends to
give some of it back the week after (see REVERSION_STRENGTH). Bull and bear
regimes trend instead. That contrast is what makes arm choice matter -- if
every regime trended, a trend-following arm would quietly dominate all of
them and there would be nothing to learn. (Phase 2's trading_env.py keeps
the simpler formulation, with no within-regime reversion; the two phases
share the regime parameters, not every dynamic.)
"""

import numpy as np

WEEK_LENGTH = 5  # trading days per "pull"

# How strongly a choppy week gives back the previous week's move.
# 0.0 = no reversion (pure random walk), 1.0 = fully retraces on average.
REVERSION_STRENGTH = 1.0


def _week_return(prices) -> float:
    return (prices[-1] / prices[0]) - 1.0


def _strategy_momentum(lookback_prices):
    """Go long if LAST week's return was positive, else flat."""
    return 1 if _week_return(lookback_prices) > 0 else 0


def _strategy_mean_reversion(lookback_prices):
    """Go long if LAST week fell (bet on a bounce-back), else flat."""
    return 1 if _week_return(lookback_prices) < 0 else 0


def _strategy_buy_hold(lookback_prices):
    return 1


def _strategy_stay_cash(lookback_prices):
    return 0


def _strategy_random(lookback_prices, rng):
    return int(rng.integers(0, 2))


STRATEGIES = ["momentum", "mean_reversion", "buy_hold", "stay_cash", "random"]


class TradingBanditEnv:
    """
    API:
        env.reset(seed)
        reward = env.pull(arm_index)   # arm_index in [0, K-1]
        env.n_arms -> K

    The regime mix is piecewise-constant: it is fixed within an epoch of
    `epoch_length` pulls and steps to the next PHASE_MIX at each phase
    boundary, so the best arm changes a few times over a run --
    non-stationary bandit, not classic i.i.d.
    """

    # Market phases, walked through in order across the run. Deliberately
    # near-pure at each end: with a mild mix the arms' expected rewards sit
    # so close together that "always pull the single best-on-average arm"
    # is within noise of the best possible adaptive policy, and a one-line
    # agent that hardcodes one arm tops the leaderboard.
    PHASE_MIXES = [
        {"bull": 0.85, "bear": 0.05, "choppy": 0.10},   # trending up
        {"bull": 0.05, "bear": 0.05, "choppy": 0.90},   # range-bound
        {"bull": 0.05, "bear": 0.85, "choppy": 0.10},   # trending down
    ]

    def __init__(self, epoch_length: int = 20, total_pulls: int = 200):
        self.n_arms = len(STRATEGIES)
        self.epoch_length = epoch_length
        self.total_pulls = total_pulls
        self.reset()

    def reset(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        self.pull_count = 0
        return None  # bandits have no state to return

    def _regime_weights(self):
        """Which market phase we're in right now. Steps at epoch boundaries
        and walks through PHASE_MIXES across the run. Students never see this
        schedule, only its effect on which arm is currently paying."""
        n_epochs = max(1, self.total_pulls // self.epoch_length)
        epoch = (self.pull_count // self.epoch_length) % n_epochs
        phase = int(epoch / n_epochs * len(self.PHASE_MIXES))
        phase = min(phase, len(self.PHASE_MIXES) - 1)
        return dict(self.PHASE_MIXES[phase])

    def _simulate_two_weeks(self):
        """One regime draw, two consecutive weeks of prices under it.

        Returns (lookback_prices, trading_prices). The regime is held fixed
        across both weeks so that last week genuinely carries information
        about next week -- that is what gives momentum/mean-reversion a real
        (but noisy, and regime-dependent) edge to be discovered.

        In bull/bear regimes the drift simply persists, so last week's
        direction tends to continue. In the choppy regime the second week's
        drift is tilted AGAINST last week's move (REVERSION_STRENGTH), so
        the move tends to partly retrace -- that is the regime where
        mean-reversion is the right arm and momentum is the wrong one.
        """
        from trading_env import REGIME_PARAMS  # reuse same regime params
        weights = self._regime_weights()
        regime = self._rng.choice(list(weights.keys()), p=list(weights.values()))
        drift, vol = REGIME_PARAMS[regime]

        lookback = [100.0]
        for _ in range(WEEK_LENGTH):
            lookback.append(lookback[-1] * np.exp(self._rng.normal(drift, vol)))

        trading_drift = drift
        if regime == "choppy":
            lookback_move = np.log(lookback[-1] / lookback[0])
            trading_drift = drift - REVERSION_STRENGTH * lookback_move / WEEK_LENGTH

        trading = [lookback[-1]]
        for _ in range(WEEK_LENGTH):
            trading.append(trading[-1] * np.exp(self._rng.normal(trading_drift, vol)))

        return np.array(lookback), np.array(trading), regime

    def pull(self, arm_index: int) -> float:
        """Deploy strategy `arm_index` for one week; return that week's return.

        The strategy sees only `lookback_prices` when choosing its position,
        and is paid on `trading_prices` -- no peeking at the week it trades.
        """
        assert 0 <= arm_index < self.n_arms
        lookback_prices, trading_prices, regime = self._simulate_two_weeks()
        strategy_name = STRATEGIES[arm_index]

        if strategy_name == "momentum":
            position = _strategy_momentum(lookback_prices)
        elif strategy_name == "mean_reversion":
            position = _strategy_mean_reversion(lookback_prices)
        elif strategy_name == "buy_hold":
            position = _strategy_buy_hold(lookback_prices)
        elif strategy_name == "stay_cash":
            position = _strategy_stay_cash(lookback_prices)
        else:
            position = _strategy_random(lookback_prices, self._rng)

        week_return = position * _week_return(trading_prices)
        self.pull_count += 1
        return week_return


if __name__ == "__main__":
    env = TradingBanditEnv()
    env.reset(seed=1)
    totals = np.zeros(env.n_arms)
    for i in range(200):
        arm = i % env.n_arms  # round robin, just to sanity-check it runs
        totals[arm] += env.pull(arm)
    for name, total in zip(STRATEGIES, totals):
        print(f"{name:15s} cumulative reward: {total:.4f}")
