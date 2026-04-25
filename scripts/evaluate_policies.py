from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from supplymind_env.environment import V3SupplyMindEnv
from supplymind_env.models import V3Action, V3Observation
from supplymind_env.policies import baseline_policy, heuristic_policy, no_op_policy
from supplymind_env.seed_catalog import EVAL_SEEDS, TASK_IDS
from supplymind_env.grading import grade_episode
from supplymind_env.solver import privileged_reference_policy, rollout_reference


PolicyFn = Callable[[V3Observation], V3Action]


POLICIES: dict[str, PolicyFn] = {
    "no_op": no_op_policy,
    "reactive_baseline": baseline_policy,
    "negotiation_heuristic": heuristic_policy,
    "privileged_reference": privileged_reference_policy,
}


def run_episode(task_id: str, seed: int, policy: PolicyFn) -> dict[str, object]:
    env = V3SupplyMindEnv(default_task_id=task_id)
    observation = env.reset_internal(task_id=task_id, internal_seed=seed, public_seed=seed)
    rewards: list[float] = []
    done = False
    while not done:
        result = env.step(policy(observation))
        rewards.append(result.reward.step_reward)
        observation = result.observation
        done = result.done
    summary = result.info["episode_summary"]
    return {
        "task_id": observation.task_id,
        "internal_task_id": task_id,
        "seed": seed,
        "raw_reward": summary["raw_reward"],
        "score": summary["graded_score"],
        "baseline_reward": summary["baseline_reward"],
        "heuristic_reward": summary["heuristic_reward"],
        "target_reward": summary["target_reward"],
        "step_rewards": rewards,
    }


def run_reference_episode(task_id: str, seed: int) -> dict[str, object]:
    raw_reward = rollout_reference(task_id, seed)
    task_result = grade_episode(task_id, seed, raw_reward)
    return {
        "task_id": V3SupplyMindEnv(default_task_id=task_id).reset_internal(task_id=task_id, internal_seed=seed, public_seed=seed).task_id,
        "internal_task_id": task_id,
        "seed": seed,
        "raw_reward": raw_reward,
        "score": task_result.score,
        "baseline_reward": task_result.baseline_reward,
        "heuristic_reward": task_result.heuristic_reward,
        "target_reward": task_result.target_reward,
        "step_rewards": [],
    }


def evaluate() -> dict[str, object]:
    policy_results: dict[str, list[dict[str, object]]] = {}
    for policy_name, policy in POLICIES.items():
        rows: list[dict[str, object]] = []
        for task_id in TASK_IDS:
            for seed in EVAL_SEEDS[task_id]:
                if policy_name == "privileged_reference":
                    rows.append(run_reference_episode(task_id, seed))
                else:
                    rows.append(run_episode(task_id, seed, policy))
        policy_results[policy_name] = rows
    summary = {}
    for policy_name, rows in policy_results.items():
        summary[policy_name] = {
            "mean_score": round(mean(float(row["score"]) for row in rows), 4),
            "mean_reward": round(mean(float(row["raw_reward"]) for row in rows), 3),
            "episodes": len(rows),
        }
    return {"summary": summary, "episodes": policy_results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/policy_eval.json")
    args = parser.parse_args()

    payload = evaluate()
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
