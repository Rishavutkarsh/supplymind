from __future__ import annotations

from functools import lru_cache

from .environment import V2SupplyMindEnv
from .models import V2TaskResult
from .policies import heuristic_joint_policy, naive_joint_policy
from .solver import rollout_reference


STRICT_SCORE_EPSILON = 1e-4
BASELINE_SCORE_ANCHOR = 0.05
REFERENCE_SCORE_ANCHOR = 0.95


def grade_episode(task_id: str, seed: int, raw_reward: float, center_reward: float, average_warehouse_reward: float) -> V2TaskResult:
    baseline_reward = cached_rollout_policy(task_id, seed, "baseline")
    target_reward = max(cached_reference_reward(task_id, seed), baseline_reward + 20.0)
    return V2TaskResult(
        task_id=task_id,
        raw_reward=raw_reward,
        baseline_reward=baseline_reward,
        target_reward=target_reward,
        score=normalize_score(raw_reward, baseline_reward, target_reward),
        center_reward=center_reward,
        average_warehouse_reward=average_warehouse_reward,
    )


def rollout_policy(task_id: str, seed: int, policy_name: str) -> float:
    env = V2SupplyMindEnv(default_task_id=task_id)
    observation = env.reset_internal(task_id, seed)
    policy = naive_joint_policy if policy_name == "baseline" else heuristic_joint_policy
    while not env.done:
        result = env.step(policy(observation), grade_terminal=False)
        observation = result.observation
    return env.cumulative_reward


@lru_cache(maxsize=512)
def cached_rollout_policy(task_id: str, seed: int, policy_name: str) -> float:
    return rollout_policy(task_id, seed, policy_name)


@lru_cache(maxsize=512)
def cached_reference_reward(task_id: str, seed: int) -> float:
    return rollout_reference(task_id, seed)


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
