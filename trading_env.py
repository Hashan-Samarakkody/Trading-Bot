"""
trading_env.py
================
A small, tabular-friendly single-player trading environment.

Design goals (why it's built this way):
- State space is DISCRETE and small (30 states) so plain Q-learning / SARSA
  tables work with no function approximation.
- Prices are SYNTHETIC (regime-switching geometric Brownian motion), so we
  can generate unlimited training episodes AND freeze a fixed set of seeds
  for fair leaderboard evaluation (no lookahead bias, no lucky real-market draws).
- Regimes (bull / bear / choppy) switch mid-episode via a Markov chain, so a
  policy that only learns "always buy" gets punished eventually. This is the
  main source of difficulty and what separates good agents from lucky ones.

Episode = one simulated "quarter" of `episode_length` trading days.
Action space = {0: SELL/STAY FLAT, 1: HOLD, 2: BUY/STAY LONG} -- single unit
position only (no shorting, no position sizing) to keep the state/action
space tabular-friendly. Trading has a small transaction cost.
"""

import numpy as np
from dataclasses import dataclass, field

REGIMES = ["bull", "bear", "choppy"]

# (drift, volatility) per regime, per-day
REGIME_PARAMS = {
    "bull":   (0.0045, 0.010),
    "bear":   (-0.0045, 0.012),
    "choppy": (0.0000, 0.018),
}

# Markov transition matrix between regimes (rows sum to 1)
# Regimes are "sticky" (tend to persist) but do switch.
REGIME_TRANSITION = {
    "bull":   {"bull": 0.94, "bear": 0.02, "choppy": 0.04},
    "bear":   {"bull": 0.02, "bear": 0.94, "choppy": 0.04},
    "choppy": {"bull": 0.06, "bear": 0.06, "choppy": 0.88},
}

TRANSACTION_COST = 0.001  # 0.1% of trade value, charged on any position change


@dataclass
class EpisodeResult:
    returns: list = field(default_factory=list)      # daily portfolio log-returns
    portfolio_values: list = field(default_factory=list)
    regimes: list = field(default_factory=list)


def _discretize_momentum(momentum_5d: float) -> int:
    """5 buckets: strong down / down / flat / up / strong up."""
    thresholds = [-0.02, -0.005, 0.005, 0.02]
    for i, t in enumerate(thresholds):
        if momentum_5d < t:
            return i
    return len(thresholds)


def _discretize_vol(vol_5d: float) -> int:
    """3 buckets: low / medium / high realized volatility."""
    if vol_5d < 0.010:
        return 0
    elif vol_5d < 0.020:
        return 1
    return 2


class TradingEnv:
    """
    Gym-like API: reset(seed) -> state (int), step(action) -> (state, reward, done, info)

    State encoding (single integer 0-29):
        state = momentum_bucket * 6 + vol_bucket * 2 + position
        momentum_bucket in [0..4]  (5 values)
        vol_bucket      in [0..2]  (3 values)
        position        in [0,1]   (2 values)  -> total 5*3*2 = 30 states
    """

    N_STATES = 30
    N_ACTIONS = 3  # 0 = flat/sell, 1 = hold, 2 = long/buy

    def __init__(self, episode_length: int = 60, lookback: int = 5):
        self.episode_length = episode_length
        self.lookback = lookback
        self._rng = np.random.default_rng()
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        self.t = 0
        self.position = 0  # 0 = flat, 1 = long
        self.regime = self._rng.choice(REGIMES)
        self.regime_history = [self.regime]

        # simulate the full price path up front (deterministic given seed)
        n = self.episode_length + self.lookback
        prices = [100.0]
        regimes = []
        regime = self.regime
        for _ in range(n):
            regimes.append(regime)
            drift, vol = REGIME_PARAMS[regime]
            shock = self._rng.normal(drift, vol)
            prices.append(prices[-1] * np.exp(shock))
            # advance regime (Markov step)
            probs = REGIME_TRANSITION[regime]
            regime = self._rng.choice(list(probs.keys()), p=list(probs.values()))
        self.prices = np.array(prices)
        self.path_regimes = regimes

        self.cash = 1.0
        self.shares = 0.0
        self.portfolio_value = 1.0
        self.episode_returns = []
        self.portfolio_values = [1.0]

        return self._get_state()

    # ------------------------------------------------------------------ #
    def _get_state(self) -> int:
        idx = self.t + self.lookback
        window = self.prices[idx - self.lookback: idx + 1]
        momentum = (window[-1] / window[0]) - 1.0
        rets = np.diff(np.log(window))
        vol = rets.std() if len(rets) > 1 else 0.0
        m_bucket = _discretize_momentum(momentum)
        v_bucket = _discretize_vol(vol)
        return m_bucket * 6 + v_bucket * 2 + self.position

    # ------------------------------------------------------------------ #
    def step(self, action: int):
        assert action in (0, 1, 2), "action must be 0 (flat), 1 (hold), 2 (long)"
        idx = self.t + self.lookback
        price_now = self.prices[idx]
        price_next = self.prices[idx + 1]

        target_position = 0 if action == 0 else (self.position if action == 1 else 1)
        cost = 0.0
        if target_position != self.position:
            cost = TRANSACTION_COST * self.portfolio_value
            self.position = target_position

        # portfolio return this step
        asset_return = (price_next / price_now) - 1.0
        pnl = (self.position * asset_return) * self.portfolio_value - cost
        self.portfolio_value += pnl
        step_return = pnl / (self.portfolio_value - pnl + 1e-9)

        self.episode_returns.append(step_return)
        self.portfolio_values.append(self.portfolio_value)

        self.t += 1
        done = self.t >= self.episode_length

        next_state = self._get_state() if not done else None
        info = {"portfolio_value": self.portfolio_value, "regime": self.path_regimes[idx]}
        return next_state, step_return, done, info

    # ------------------------------------------------------------------ #
    def episode_summary(self) -> EpisodeResult:
        return EpisodeResult(
            returns=self.episode_returns,
            portfolio_values=self.portfolio_values,
            # offset by lookback: path_regimes[0:lookback] covers the warm-up
            # window used only to compute the first state, not any traded step
            regimes=self.path_regimes[self.lookback: self.lookback + self.episode_length],
        )


if __name__ == "__main__":
    # smoke test with a random agent
    env = TradingEnv()
    s = env.reset(seed=42)
    total_reward = 0.0
    done = False
    while not done:
        a = np.random.choice([0, 1, 2])
        s, r, done, info = env.step(a)
        total_reward += r
    print(f"Random agent final portfolio value: {info['portfolio_value']:.4f}")
    print(f"Number of discrete states: {TradingEnv.N_STATES}, actions: {TradingEnv.N_ACTIONS}")
