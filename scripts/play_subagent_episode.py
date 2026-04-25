from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from supplymind_env.environment import V3SupplyMindEnv
from supplymind_env.subagent import SUBAGENT_SYSTEM_PROMPT, prompted_subagent_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one prompted SupplyMind subagent episode.")
    parser.add_argument("--task", default="scarcity_negotiation")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", default=str(ROOT / "results" / "subagent_episode.json"))
    args = parser.parse_args()

    env = V3SupplyMindEnv(default_task_id=args.task)
    observation = env.reset(args.task, args.seed)
    timeline = []

    while not env.done:
        action = prompted_subagent_policy(observation)
        result = env.step(action)
        timeline.append(
            {
                "round": observation.round_index,
                "action": action.model_dump(mode="json"),
                "reward": result.reward.model_dump(mode="json"),
                "events": result.observation.feedback.recent_events,
                "components": result.reward.components,
                "done": result.done,
            }
        )
        observation = result.observation

    payload = {
        "task": args.task,
        "seed": args.seed,
        "system_prompt": SUBAGENT_SYSTEM_PROMPT,
        "episode_summary": env.last_episode_summary,
        "timeline": timeline,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "episode_summary": env.last_episode_summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
