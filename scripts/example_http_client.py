from __future__ import annotations

import argparse

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal SupplyMind HTTP API client.")
    parser.add_argument("--base_url", default="http://127.0.0.1:7860")
    parser.add_argument("--task_id", default="scarcity_negotiation")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    reset = requests.post(
        f"{args.base_url}/reset",
        params={"task_id": args.task_id, "seed": args.seed},
        timeout=30,
    )
    reset.raise_for_status()
    observation = reset.json()
    print({"event": "reset", "task_id": observation["task_id"], "round": observation["round_index"]})

    done = False
    while not done:
        action = requests.get(f"{args.base_url}/heuristic-action", timeout=30).json()
        step = requests.post(f"{args.base_url}/step", json=action, timeout=30)
        step.raise_for_status()
        payload = step.json()
        observation = payload["observation"]
        done = payload["done"]
        print(
            {
                "round": observation["round_index"],
                "reward": payload["reward"]["step_reward"],
                "components": payload["reward"]["components"],
                "done": done,
            }
        )

    print({"event": "done", "summary": payload["info"].get("episode_summary")})


if __name__ == "__main__":
    main()
