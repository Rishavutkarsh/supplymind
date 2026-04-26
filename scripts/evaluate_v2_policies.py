from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from supplymind_env_v2.environment import V2SupplyMindEnv
from supplymind_env_v2.generator import BENCHMARK_PROFILE_IDS, TRAINING_PROFILE_IDS
from supplymind_env_v2.grading import cached_reference_reward, grade_episode
from supplymind_env_v2.policies import heuristic_joint_policy, naive_joint_policy, no_op_policy


POLICIES = {
    "no_op": no_op_policy,
    "naive_joint": naive_joint_policy,
    "heuristic_joint": heuristic_joint_policy,
}
SEEDS = (101, 103, 105)
PROFILE_IDS = BENCHMARK_PROFILE_IDS


def run_policy(task_id: str, seed: int, policy) -> dict:
    env = V2SupplyMindEnv(default_task_id=task_id)
    observation = env.reset_internal(task_id, seed)
    while not env.done:
        result = env.step(policy(observation))
        observation = result.observation
    summary = result.info["episode_summary"]
    return {
        "task_id": task_id,
        "seed": seed,
        "raw_reward": summary["raw_reward"],
        "score": summary["graded_score"],
        "center_reward": summary["center_reward"],
        "average_warehouse_reward": summary["average_warehouse_reward"],
        "baseline_reward": summary["baseline_reward"],
        "target_reward": summary["target_reward"],
    }


def main() -> None:
    results = {}
    for policy_name, policy in POLICIES.items():
        rows = []
        for task_id in PROFILE_IDS:
            for seed in SEEDS:
                rows.append(run_policy(task_id, seed, policy))
        results[policy_name] = rows
    ref_rows = []
    for task_id in PROFILE_IDS:
        for seed in SEEDS:
            raw = cached_reference_reward(task_id, seed)
            task_result = grade_episode(task_id, seed, raw, 0.0, 0.0)
            ref_rows.append({"task_id": task_id, "seed": seed, "raw_reward": raw, "score": task_result.score})
    results["privileged_reference"] = ref_rows
    summary = {
        name: {
            "mean_score": round(mean(float(row["score"]) for row in rows), 4),
            "mean_reward": round(mean(float(row["raw_reward"]) for row in rows), 3),
            "episodes": len(rows),
        }
        for name, rows in results.items()
    }
    out = ROOT / "results" / "v2_policy_eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "profile_scope": "benchmark",
                "benchmark_profile_ids": BENCHMARK_PROFILE_IDS,
                "training_profile_ids": TRAINING_PROFILE_IDS,
                "summary": summary,
                "episodes": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
