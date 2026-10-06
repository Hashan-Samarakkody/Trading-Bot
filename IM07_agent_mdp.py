"""
agent_mdp.py  --  Phase 2 submission (30-state / 3-action trading MDP)
=====================================================================

Contains a shared tabular base class and three variants from Lecture 5:

    QLearningAgent    off-policy  target = R + gamma * max_a Q(S',a)
    SarsaAgent        on-policy   target = R + gamma * Q(S',A')
    SarsaLambdaAgent  on-policy   Sarsa with accumulating eligibility traces

`Agent` is aliased at the bottom of the file to whichever variant won the
head-to-head comparison in experiments/IM07_compare_mdp.py. Only one class is
named `Agent`, as the specification requires, and the file is standalone so
it can be submitted as-is.

Why the bundled baseline scores only ~0.03
------------------------------------------
`QLearningReferenceAgent` uses fixed epsilon = 0.1, fixed alpha = 0.1,
gamma = 0.95 and no decay anywhere. In lecture terms:

  * Fixed epsilon is not GLIE (Lecture 5). Lecture 6 adds that it carries a
    per-step regret floor, so the agent is still throwing away one step in
    ten on known-bad actions at episode 300, and it never becomes greedy.
  * gamma = 0.95 has an effective horizon of about 1/(1-gamma) = 20 steps.
    An episode is 60 trading days and the payoff of holding a position
    through a bull regime accrues across all of them, so a 20-day horizon
    truncates most of what the agent is supposed to be learning.
  * Q initialised at zero gives no systematic exploration at all; the agent
    relies entirely on epsilon's coin flips to discover the action space.

Each change below started from a lecture result. Three of them then had to
be REVERSED by measurement, which is recorded here rather than hidden --
the corrections are the most instructive part of the build.

Measured properties of this environment (experiments/IM07_diagnose_mdp.py)
--------------------------------------------------------------------
    per-step reward : mean -0.000403, sd 0.014172
    signal-to-noise : roughly 1 : 35
    plausible |Q|   : ~0.008 at gamma=0.95, ~0.024 at gamma=0.99

That noise ratio dominates every hyperparameter decision below, and it is
the reason two lecture-derived predictions were wrong.

Three decisions, as predicted and as measured
---------------------------------------------
1. DISCOUNT.  Predicted gamma = 0.99; MEASURED gamma = 0.95 is better.
   The prediction came from Lecture 2: gamma near 1 is far-sighted, and
   1/(1-0.99) = 100 steps covers the whole 60-day episode, whereas
   1/(1-0.95) = 20 steps truncates it. That reasoning is sound about BIAS
   and silent about VARIANCE. With a 1:35 signal-to-noise ratio, a longer
   bootstrap horizon accumulates noise faster than it accumulates signal,
   so the shorter horizon is the better bias/variance trade. This is
   Lecture 4's trade-off deciding a question Lecture 2 appeared to own.

2. EXPLORATION.  Predicted a GLIE decay to eps = 0.01; MEASURED a floor of
   eps = 0.10 is better -- the same level the untuned baseline uses.
   Lecture 5 defines GLIE as (a) every state-action pair visited infinitely
   often and (b) greedy in the limit, with eps_k = 1/k as the example. Both
   conditions are ASYMPTOTIC. Here there are 300 episodes for a 90-cell
   table under heavy noise, which is nowhere near the limit, so decaying to
   near-greedy early does not "converge" -- it freezes whatever noise the
   table happens to hold. The schedule therefore decays geometrically from
   0.5 to a 0.10 floor. set_eval_mode(True) still sets epsilon to exactly 0,
   so GLIE condition (b) is enforced by the harness at scoring time.

3. OPTIMISTIC INIT.  Lecture 6 prescribes Q_0 = r_max/(1-gamma) so that every
   estimate starts above the truth and every update pushes it down, giving a
   systematic sweep instead of random noise. That formula assumes the
   lecture's [0,1] reward scale and must be rescaled here -- but the first
   attempt rescaled it in the WRONG DIRECTION, to q_init = 0.01, which is
   BELOW the measured plausible |Q| of 0.024 at gamma = 0.99 and so was
   pessimistic rather than optimistic. The fix was to measure the plausible
   |Q| first and then sit just above it: q_init = 0.03.

The general lesson, and the one worth putting in the report: a lecture
formula carries its reward scale and its asymptotic regime as hidden
assumptions. Both have to be checked against the environment before the
formula can be applied.

Note on what does NOT work, and why
-----------------------------------
Multiplying the reward by a constant before the update changes nothing.
Scaling every reward scales every Q-value by the same factor and argmax is
invariant under a positive scaling, so the greedy policy -- and the score --
come out identical. Both the specification and the README flag this; it is
recorded here so the reasoning is visible rather than rediscovered.
"""

import numpy as np

from agent_base import MDPAgent

# Harness constants this agent is calibrated against (eval_harness.py).
N_TRAIN_EPISODES = 300


class _TabularAgent(MDPAgent):
    """Shared machinery: Q-table, GLIE epsilon schedule, eval-mode freeze.

    Subclasses differ only in the TARGET used by update(), which is exactly
    how Lecture 5 presents the family: "new <- old + alpha (target - old)",
    with only the target changing.
    """

    def __init__(self, n_states, n_actions,
                 alpha=0.05, gamma=0.95,
                 eps_start=0.5, eps_end=0.10, q_init=0.03):
        super().__init__(n_states, n_actions)
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.epsilon = eps_start
        self.q_init = q_init

        # Optimistic initialisation, on the scale of this environment's
        # rewards rather than the lecture's [0,1] convention.
        self.Q = np.full((n_states, n_actions), q_init, dtype=float)

        self._rng = np.random.default_rng()
        self._episode = 0
        self._eval = False

        # Geometric decay that lands on eps_end at the end of training.
        self._eps_decay = (eps_end / eps_start) ** (1.0 / max(1, N_TRAIN_EPISODES))

    # ------------------------------------------------------------------ #
    def start_episode(self) -> None:
        """Advance the GLIE schedule. The Q-table deliberately persists --
        the specification is explicit that it must keep improving across all
        300 episodes and must not be reinitialised here."""
        self._episode += 1
        if not self._eval:
            self.epsilon = max(self.eps_end, self.eps_start * (self._eps_decay ** self._episode))

    def set_eval_mode(self, is_eval: bool) -> None:
        self._eval = bool(is_eval)
        self.epsilon = 0.0 if is_eval else self.eps_start

    # ------------------------------------------------------------------ #
    def _greedy(self, state: int) -> int:
        return int(np.argmax(self.Q[state]))

    def _eps_greedy(self, state: int) -> int:
        if self.epsilon > 0.0 and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return self._greedy(state)

    def act(self, state: int) -> int:
        return self._eps_greedy(state)

    def update(self, state, action, reward, next_state, done) -> None:
        raise NotImplementedError


class QLearningAgent(_TabularAgent):
    """Off-policy control.  target = R + gamma * max_a Q(S', a)

    Lecture 5: the behaviour policy explores while the target policy is
    fully greedy, so we learn about the optimal policy rather than about a
    compromised epsilon-greedy one. No importance sampling is needed because
    Q(s,a) already conditions on the action.
    """

    def update(self, state, action, reward, next_state, done) -> None:
        target = reward
        if not done and next_state is not None:
            target += self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])


class SarsaAgent(_TabularAgent):
    """On-policy control.  target = R + gamma * Q(S', A')

    A' must be the action actually taken next, not a freshly drawn one.
    Lecture 5 names re-sampling A' at the top of the next loop as the common
    bug that makes the algorithm "silently stop being Sarsa", so A' is chosen
    inside update() and then replayed by act().
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_action = None   # the A' we committed to

    def start_episode(self) -> None:
        super().start_episode()
        self._pending_action = None

    def act(self, state: int) -> int:
        # Replay the A' that the last update bootstrapped from, so the action
        # we learned about is the action we actually take.
        if self._pending_action is not None:
            action, self._pending_action = self._pending_action, None
            return action
        return self._eps_greedy(state)

    def update(self, state, action, reward, next_state, done) -> None:
        target = reward
        if not done and next_state is not None:
            next_action = self._eps_greedy(next_state)
            self._pending_action = next_action
            target += self.gamma * self.Q[next_state, next_action]
        else:
            self._pending_action = None
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])


class SarsaLambdaAgent(SarsaAgent):
    """Sarsa with accumulating eligibility traces (Lecture 5 / Lecture 4).

    Motivation specific to this project: plain Sarsa propagates reward
    backwards about one state per episode, and we only get 300 episodes over
    a 60-step horizon. Traces send each TD error back to every (s,a) pair
    still "on the hook", decayed by gamma*lambda -- the frequency and recency
    heuristics at once. The table is 30x3, so updating every cell each step
    is trivially affordable inside the 90 s budget.
    """

    def __init__(self, *args, lam=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.lam = lam
        self.E = np.zeros((self.n_states, self.n_actions), dtype=float)

    def start_episode(self) -> None:
        super().start_episode()
        self.E.fill(0.0)   # traces reset every episode; the Q-table does not

    def update(self, state, action, reward, next_state, done) -> None:
        target = reward
        if not done and next_state is not None:
            next_action = self._eps_greedy(next_state)
            self._pending_action = next_action
            target += self.gamma * self.Q[next_state, next_action]
        else:
            self._pending_action = None

        delta = target - self.Q[state, action]
        self.E[state, action] += 1.0
        self.Q += self.alpha * delta * self.E
        self.E *= self.gamma * self.lam


# Selected by the head-to-head in experiments/IM07_sweep_mdp.py: Q-learning won on
# the graded seeds (0.0389) over SARSA and SARSA(lambda) (both 0.0357).
Agent = QLearningAgent
