from __future__ import annotations

from copy import deepcopy

from .config import price_band
from .environment import V2SupplyMindEnv
from .models import CenterAction, V2JointAction
from .policies import heuristic_joint_policy, naive_joint_policy


def rollout_reference(task_id: str, seed: int) -> float:
    return max(_rollout_planner(task_id, seed), _rollout_policy(task_id, seed, naive_joint_policy), _rollout_policy(task_id, seed, heuristic_joint_policy))


def privileged_reference_policy(observation) -> V2JointAction:
    return heuristic_joint_policy(observation)


def _rollout_policy(task_id: str, seed: int, policy) -> float:
    env = V2SupplyMindEnv(default_task_id=task_id)
    observation = env.reset_internal(task_id, seed)
    while not env.done:
        result = env.step(policy(observation), grade_terminal=False)
        observation = result.observation
    return env.cumulative_reward


def _rollout_planner(task_id: str, seed: int) -> float:
    env = V2SupplyMindEnv(default_task_id=task_id)
    observation = env.reset_internal(task_id, seed)
    while not env.done:
        action = _best_action(env, observation)
        result = env.step(action, grade_terminal=False)
        observation = result.observation
    return env.cumulative_reward


def _best_action(env: V2SupplyMindEnv, observation) -> V2JointAction:
    candidates = _candidate_actions(observation)
    best = candidates[0]
    best_value = float("-inf")
    for action in candidates:
        clone = deepcopy(env)
        before = clone.cumulative_reward
        result = clone.step(action, grade_terminal=False)
        if not result.done:
            clone.step(heuristic_joint_policy(result.observation), grade_terminal=False)
        value = clone.cumulative_reward - before
        if value > best_value:
            best_value = value
            best = action
    return best


def _candidate_actions(observation) -> list[V2JointAction]:
    base = heuristic_joint_policy(observation)
    candidates = [V2JointAction(), naive_joint_policy(observation), base]
    for summary in observation.center.warehouse_summaries:
        for sku, units in summary["inventory"].items():
            if units <= 1 and observation.center.depot_inventory.get(sku, 0) > 0 and observation.center.depot_trucks_available > 0:
                candidates.append(
                    V2JointAction(
                        warehouse_actions=base.warehouse_actions,
                        central_action=CenterAction(
                            central_replenishments=[
                                {
                                    "to_warehouse": summary["warehouse_id"],
                                    "sku": sku,
                                    "units": min(3, observation.center.depot_inventory[sku]),
                                    "unit_price": price_band(sku)["fair_wholesale_price"],
                                }
                            ]
                        ),
                    )
                )
    for sku, units in observation.center.depot_inventory.items():
        if units <= 2 and observation.center.remaining_rounds > 4:
            candidates.append(
                V2JointAction(
                    warehouse_actions=base.warehouse_actions,
                    central_action=CenterAction(central_procurements=[{"sku": sku, "units": 4, "max_unit_cost": price_band(sku)["procurement_cost"]}]),
                )
            )
    return candidates[:24]
