from __future__ import annotations

from .environment import V3SupplyMindEnv
from .models import V3Action, V3Observation
from .dynamics import visible_orders
from .policies import baseline_policy, heuristic_policy


def privileged_reference_policy(observation: V3Observation) -> V3Action:
    action = baseline_policy(observation)
    priority_policy = []
    if observation.demand_reports:
        top_report = max(
            observation.demand_reports,
            key=lambda item: (item.urgency, item.missed_units_last_round, item.at_risk_units, item.requested_units + item.forecast_units),
        )
        priority_policy.append({"sku": top_report.sku, "region": top_report.region, "priority": top_report.urgency})

    transfers = list(action.inventory_transfers)
    for report in sorted(observation.demand_reports, key=lambda item: (-item.urgency, -item.missed_units_last_round, -item.at_risk_units)):
        if report.urgency < 3 and report.missed_units_last_round <= 0:
            continue
        needed_units = max(1, min(2, report.requested_units + report.at_risk_units + report.missed_units_last_round))
        donors = [
            warehouse for warehouse in observation.warehouses
            if warehouse.inventory.get(report.sku, 0) >= needed_units + 3
        ]
        receivers = [
            warehouse for warehouse in observation.warehouses
            if warehouse.warehouse_id == report.warehouse_id
        ]
        if donors and receivers:
            donor = max(donors, key=lambda item: item.inventory.get(report.sku, 0))
            receiver = receivers[0]
            if donor.warehouse_id != receiver.warehouse_id:
                transfers.append(
                    {
                        "from_warehouse": donor.warehouse_id,
                        "to_warehouse": receiver.warehouse_id,
                        "sku": report.sku,
                        "units": needed_units,
                        "compensation": needed_units * 9.0,
                    }
                )
                break
    return V3Action(
        central_replenishments=action.central_replenishments,
        inventory_transfers=transfers[:2],
        driver_loans=action.driver_loans,
        offer_matches=action.offer_matches,
        priority_policy=priority_policy,
        defer_orders=action.defer_orders,
        coalition_deals=action.coalition_deals,
    )


def rollout_reference(task_id: str, seed: int) -> float:
    return max(
        _rollout_oracle(task_id, seed),
        _rollout_policy(task_id, seed, baseline_policy),
        _rollout_policy(task_id, seed, heuristic_policy),
    )


def _rollout_oracle(task_id: str, seed: int) -> float:
    env = V3SupplyMindEnv(default_task_id=task_id)
    observation = env.reset_internal(task_id=task_id, internal_seed=seed)
    while not env.done:
        result = env.step(_oracle_action(env, observation), grade_terminal=False)
        observation = result.observation
    return env.cumulative_reward


def _rollout_policy(task_id: str, seed: int, policy) -> float:
    env = V3SupplyMindEnv(default_task_id=task_id)
    observation = env.reset_internal(task_id=task_id, internal_seed=seed)
    while not env.done:
        result = env.step(policy(observation), grade_terminal=False)
        observation = result.observation
    return env.cumulative_reward


def _oracle_action(env: V3SupplyMindEnv, observation: V3Observation) -> V3Action:
    recipe = env._require_recipe()
    specs_by_region = {spec.region: spec for spec in recipe.warehouse_specs}
    replenishments = []
    committed_depot: dict[str, int] = {}
    used_trucks = 0
    for order in sorted(
        visible_orders(recipe, env.round_index, env.completed_orders, env.expired_orders),
        key=lambda item: (-item.priority, item.deadline_round, -item.units * item.value_per_unit),
    ):
        spec = specs_by_region.get(order.region)
        if spec is None:
            continue
        available = env.inventory[spec.warehouse_id].get(order.sku, 0)
        if available >= order.units:
            continue
        if used_trucks >= observation.central_depot.trucks_available:
            break
        depot_left = env.central_depot_inventory.get(order.sku, 0) - committed_depot.get(order.sku, 0)
        units = min(order.units - available, depot_left, 2)
        if units > 0:
            replenishments.append({"to_warehouse": spec.warehouse_id, "sku": order.sku, "units": units})
            committed_depot[order.sku] = committed_depot.get(order.sku, 0) + units
            used_trucks += 1
    return V3Action(central_replenishments=replenishments)
