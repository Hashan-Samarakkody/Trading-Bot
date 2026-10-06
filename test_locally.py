"""
test_locally.py
================
Run your own agent through the exact same evaluation the server will use,
without needing the server at all. Use this to iterate quickly.

Usage:
    python3 test_locally.py my_agent.py bandit
    python3 test_locally.py my_agent.py mdp
"""

import sys
import importlib.util


def load_agent_class(path: str):
    spec = importlib.util.spec_from_file_location("student_agent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent


def main():
    if len(sys.argv) != 3:
        print("usage: python3 test_locally.py <your_agent_file.py> <bandit|mdp>")
        sys.exit(1)

    path, phase = sys.argv[1], sys.argv[2]
    agent_class = load_agent_class(path)

    if phase == "bandit":
        from eval_harness import run_bandit_eval
        result = run_bandit_eval(agent_class)
    elif phase == "mdp":
        from eval_harness import run_mdp_eval
        result = run_mdp_eval(agent_class)
    else:
        print("phase must be 'bandit' or 'mdp'")
        sys.exit(1)

    print("\n=== Result ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    # rough context: how does this compare to the baselines?
    print("\n=== For reference, baseline scores ===")
    if phase == "bandit":
        from baseline_agents import RandomBanditAgent, EpsilonGreedyBanditAgent
        from eval_harness import run_bandit_eval as rbe
        for cls in [RandomBanditAgent, EpsilonGreedyBanditAgent]:
            r = rbe(cls)
            print(f"  {cls.__name__:25s} score={r['score']:.4f}")
    else:
        from baseline_agents import RandomMDPAgent, BuyAndHoldMDPAgent, QLearningReferenceAgent
        from eval_harness import run_mdp_eval as rme
        for cls in [RandomMDPAgent, BuyAndHoldMDPAgent, QLearningReferenceAgent]:
            r = rme(cls)
            print(f"  {cls.__name__:25s} score={r['score']:.4f}")


if __name__ == "__main__":
    main()
