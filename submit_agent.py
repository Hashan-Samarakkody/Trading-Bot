"""
submit_agent.py
================
Reads your agent file and submits it to the leaderboard server.

Usage:
    python3 submit_agent.py <agent_file.py> <student_id> <bandit|mdp> [server_url]

Example:
    python3 submit_agent.py agent.py alice bandit
    python3 submit_agent.py agent.py alice mdp http://10.0.0.5:8000
"""
import sys
import json
import urllib.request


def main():
    if len(sys.argv) < 4:
        print("usage: python3 submit_agent.py <agent_file.py> <student_id> <bandit|mdp> [server_url]")
        sys.exit(1)

    agent_path, student_id, phase = sys.argv[1], sys.argv[2], sys.argv[3]
    server_url = sys.argv[4] if len(sys.argv) > 4 else "http://localhost:8000"

    with open(agent_path) as f:
        code = f.read()

    payload = json.dumps({"student_id": student_id, "phase": phase, "code": code}).encode()
    req = urllib.request.Request(
        f"{server_url}/submit", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )

    print(f"Submitting {agent_path} as '{student_id}' (phase: {phase}) to {server_url} ...")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
    except Exception as e:
        print(f"Submission failed to reach the server: {e}")
        sys.exit(1)

    if result.get("ok"):
        print(f"Success! Score: {result['result']['score']:.4f}")
        print(f"Leaderboard: {server_url}/leaderboard.html?phase={phase}")
    else:
        print(f"Submission was rejected: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
