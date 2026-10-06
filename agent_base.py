"""
agent_base.py
=============
Interfaces students implement. Submissions must define a class named
`Agent` in their file that satisfies ONE of these two interfaces, matching
whichever phase is being evaluated.

Phase 1 (bandit): BanditAgent
Phase 2 (full MDP): MDPAgent

Keep both interfaces in every submission file across the semester -- the
eval harness looks for whichever one is present (see eval_harness.py).
"""

from abc import ABC, abstractmethod


class BanditAgent(ABC):
    """Phase 1 interface -- strategy selection over K arms, no state."""

    def __init__(self, n_arms: int):
        self.n_arms = n_arms

    @abstractmethod
    def select_arm(self) -> int:
        """Return an integer arm index in [0, n_arms)."""
        raise NotImplementedError

    @abstractmethod
    def update(self, arm: int, reward: float) -> None:
        """Called after each pull with the arm chosen and reward received."""
        raise NotImplementedError


class MDPAgent(ABC):
    """Phase 2 interface -- full tabular Q-learning / SARSA agent."""

    def __init__(self, n_states: int, n_actions: int):
        self.n_states = n_states
        self.n_actions = n_actions

    @abstractmethod
    def act(self, state: int) -> int:
        """Return an integer action in [0, n_actions)."""
        raise NotImplementedError

    @abstractmethod
    def update(self, state: int, action: int, reward: float,
               next_state: int | None, done: bool) -> None:
        """Called after every environment step to update the agent's policy."""
        raise NotImplementedError

    def start_episode(self) -> None:
        """Optional hook, called at the start of every training episode."""
        pass

    def set_eval_mode(self, is_eval: bool) -> None:
        """Optional hook. The eval harness calls this with True right before
        the held-out evaluation episodes (after training is complete), so
        you can turn off exploration (e.g. set epsilon = 0) if you want.
        Default is a no-op -- if you don't override this, your agent keeps
        exploring during evaluation too, which is fine but adds variance."""
        pass
