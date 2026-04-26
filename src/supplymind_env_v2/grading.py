from __future__ import annotations

from functools import lru_cache

from .environment import V2SupplyMindEnv
from .models import V2TaskResult
from .policies import heuristic_joint_policy, naive_joint_policy
from .solver import RolloutStats, rollout_reference_stats


STRICT_SCORE_EPSILON = 1e-4
BASELINE_SCORE_ANCHOR = 0.05
REFERENCE_SCORE_ANCHOR = 0.95


def grade_episode(task_id: str, seed: int, raw_reward: float, center_reward: float, average_warehouse_reward: float) -> V2TaskResult:
    baseline = cached_rollout_policy(task_id, seed, "baseline")
    reference = cached_reference_stats(task_id, seed)
    target_reward = max(reference.global_reward, baseline.global_reward + 20.0)
    target_center_reward = max(reference.center_reward, baseline.center_reward + 5.0)
    target_warehouse_reward = max(reference.average_warehouse_reward, baseline.average_warehouse_reward + 5.0)
    return V2TaskResult(
        task_id=task_id,
        raw_reward=raw_reward,
        baseline_reward=baseline.global_reward,
        target_reward=target_reward,
        score=normalize_score(raw_reward, baseline.global_reward, target_reward),
        center_reward=center_reward,
        average_warehouse_reward=average_warehouse_reward,
        baseline_center_reward=baseline.center_reward,
        target_center_reward=target_center_reward,
        center_role_score=normalize_score(center_reward, baseline.center_reward, target_center_reward),
        baseline_warehouse_reward=baseline.average_warehouse_reward,
        target_warehouse_reward=target_warehouse_reward,
        warehouse_role_score=normalize_score(average_warehouse_reward, baseline.average_warehouse_reward, target_warehouse_reward),
    )


def rollout_policy(task_id: str, seed: int, policy_name: str) -> RolloutStats:
    env = V2SupplyMindEnv(default_task_id=task_id)
    observation = env.reset_internal(task_id, seed)
    policy = naive_joint_policy if policy_name == "baseline" else heuristic_joint_policy
    while not env.done:
        result = env.step(policy(observation), grade_terminal=False)
        observation = result.observation
    warehouse_rewards = [value for key, value in env.agent_rewards.items() if key != "center"]
    return RolloutStats(
        global_reward=env.cumulative_reward,
        center_reward=env.agent_rewards["center"],
        average_warehouse_reward=sum(warehouse_rewards) / len(warehouse_rewards) if warehouse_rewards else 0.0,
    )


@lru_cache(maxsize=512)
def cached_rollout_policy(task_id: str, seed: int, policy_name: str) -> RolloutStats:
    return rollout_policy(task_id, seed, policy_name)


@lru_cache(maxsize=512)
def cached_reference_stats(task_id: str, seed: int) -> RolloutStats:
    return rollout_reference_stats(task_id, seed)


def normalize_score(raw_reward: float, baseline_reward: float, target_reward: float) -> float:
    lower = STRICT_SCORE_EPSILON
    upper = 1.0 - STRICT_SCORE_EPSILON
    if target_reward <= baseline_reward:
        return REFERENCE_SCORE_ANCHOR if raw_reward >= target_reward else BASELINE_SCORE_ANCHOR
    progress = (raw_reward - baseline_reward) / (target_reward - baseline_reward)
    if progress <= 1:
        score = BASELINE_SCORE_ANCHOR + progress * (REFERENCE_SCORE_ANCHOR - BASELINE_SCORE_ANCHOR)
    else:
        score = REFERENCE_SCORE_ANCHOR + min(progress - 1, 1.0) * (upper - REFERENCE_SCORE_ANCHOR)
    return max(lower, min(upper, score))
