"""
test_sarsa_correctness.py  --  gate V5 (owner: R3 Engineer)

Lecture 5 names one specific bug by name:

    "Common bug to avoid: choosing A' twice. If you re-sample the next
     action at the top of the following iteration, the value you
     bootstrapped from is not the action you took, and the algorithm
     silently stops being Sarsa."

Silently is the operative word -- a broken Sarsa still trains, still scores,
and still looks like Sarsa in a report. Since 30% of the marks are for
correctness of the RL implementation, this is worth an explicit test rather
than an eyeball.

The test instruments the agent: every action returned by act() is recorded,
and every action bootstrapped from inside update() is recorded. For genuine
Sarsa the two sequences must coincide -- the A' used in the target at step t
must be the A taken at step t+1.

The same test is run against the Q-learning agent, where the property must
NOT hold (Q-learning bootstraps from argmax, not from what it does next).
A test that passes for both would be testing nothing.

Run:  python experiments/IM07_test_sarsa_correctness.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from IM07_agent_mdp import QLearningAgent, SarsaAgent, SarsaLambdaAgent  # noqa: E402
from trading_env import TradingEnv  # noqa: E402


def trace_episode(agent_cls, seed=0, **kwargs):
    """Run one episode, recording actions taken and actions bootstrapped from."""
    env = TradingEnv(episode_length=60)
    agent = agent_cls(TradingEnv.N_STATES, TradingEnv.N_ACTIONS, **kwargs)

    taken, bootstrapped = [], []

    # Wrap update() so we can see which next-action it used in the target.
    original_update = agent.update

    def spy_update(state, action, reward, next_state, done):
        before = getattr(agent, "_pending_action", None)
        original_update(state, action, reward, next_state, done)
        after = getattr(agent, "_pending_action", None)
        bootstrapped.append(after if before is None else after)

    agent.update = spy_update

    state = env.reset(seed=seed)
    agent.start_episode()
    done = False
    while not done:
        action = agent.act(state)
        taken.append(action)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state

    return taken, bootstrapped


def check(agent_cls, expect_match: bool, label: str, **kwargs):
    taken, bootstrapped = trace_episode(agent_cls, **kwargs)

    # bootstrapped[t] is the A' used in the target at step t; it should equal
    # taken[t+1], the action actually taken at the next step.
    pairs = [(bootstrapped[t], taken[t + 1])
             for t in range(len(taken) - 1)
             if bootstrapped[t] is not None]
    if not pairs:
        matches = 0
        rate = 0.0
    else:
        matches = sum(a == b for a, b in pairs)
        rate = matches / len(pairs)

    ok = (rate == 1.0) if expect_match else (rate < 1.0)
    status = "PASS" if ok else "FAIL"
    want = "must be 100%" if expect_match else "must be < 100%"
    print(f"  {label:<22s} bootstrapped A' == next action taken: "
          f"{matches:3d}/{len(pairs):3d} = {rate:6.1%}   ({want})   {status}")
    return ok


def main():
    print("=" * 88)
    print("Gate V5: does each agent bootstrap from the action it actually takes?")
    print("=" * 88)

    all_ok = True
    all_ok &= check(SarsaAgent, True, "SarsaAgent")
    all_ok &= check(SarsaLambdaAgent, True, "SarsaLambdaAgent")
    all_ok &= check(QLearningAgent, False, "QLearningAgent")

    print()
    print("  SARSA is on-policy: the target uses the action it will take, so the")
    print("  two sequences must agree exactly.")
    print("  Q-learning is off-policy: it bootstraps from max_a Q(S',a) and keeps")
    print("  no pending action, so it must NOT agree -- otherwise the two")
    print("  algorithms would be identical and the comparison meaningless.")
    print()
    print(f"  GATE V5: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
