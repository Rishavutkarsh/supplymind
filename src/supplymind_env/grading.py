from __future__ import annotations

from functools import lru_cache

from .environment import V3SupplyMindEnv
from .models import V3TaskResult
from .policies import baseline_policy, heuristic_policy
from .solver import rollout_reference


STRICT_SCORE_EPSILON = 1e-4
BASELINE_SCORE_ANCHOR = 0.05


def grade_episode(task_id: str, seed: int, raw_reward: float) -> V3TaskResult:
    baseline_reward = cached_rollout_policy(task_id, seed, "baseline")
    heuristic_reward = cached_rollout_policy(task_id, seed, "heuristic")
    target_reward = cached_reference_reward(task_id, seed)
    target_reward = max(target_reward, baseline_reward + 20.0, heuristic_reward + 10.0)
    score = normalize_score(raw_reward, baseline_reward, target_reward)
    return V3TaskResult(
        task_id=task_id,
        raw_reward=raw_reward,
        baseline_reward=baseline_reward,
        target_reward=target_reward,
        score=score,
        heuristic_reward=heuristic_reward,
    )


def rollout_policy(task_id: str, seed: int, policy_name: str) -> float:
    env = V3SupplyMindEnv(default_task_id=task_id)
    observation = env.reset_internal(task_id=task_id, internal_seed=seed)
    policy = baseline_policy if policy_name == "baseline" else heuristic_policy
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
        return upper if raw_reward >= target_reward else BASELINE_SCORE_ANCHOR
    normalized = (raw_reward - baseline_reward) / (target_reward - baseline_reward)
    score = BASELINE_SCORE_ANCHOR + normalized * (upper - BASELINE_SCORE_ANCHOR)
    return max(lower, min(upper, score))
