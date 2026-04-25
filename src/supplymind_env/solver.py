from __future__ import annotations

import json
from copy import deepcopy

from .dynamics import generate_market_signals
from .environment import V3SupplyMindEnv
from .models import V3Action, V3Observation
from .dynamics import visible_orders
from .policies import baseline_policy, heuristic_policy


LOOKAHEAD_DEPTH = 2
MAX_CANDIDATES = 28


def privileged_reference_policy(observation: V3Observation) -> V3Action:
    action = heuristic_policy(observation)
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
        central_procurements=action.central_procurements,
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
        _rollout_bounded_planner(task_id, seed),
        _rollout_policy(task_id, seed, baseline_policy),
        _rollout_policy(task_id, seed, heuristic_policy),
    )


def _rollout_bounded_planner(task_id: str, seed: int) -> float:
    env = V3SupplyMindEnv(default_task_id=task_id)
    observation = env.reset_internal(task_id=task_id, internal_seed=seed)
    while not env.done:
        result = env.step(_bounded_lookahead_action(env, observation), grade_terminal=False)
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
    procurements_by_sku: dict[str, int] = {}
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
        if depot_left <= 2 and env.round_index + recipe.profile.depot_procurement_lead_time < recipe.profile.total_rounds:
            already = sum(procurements_by_sku.values())
            buy_units = min(3, recipe.profile.depot_procurement_cap - already)
            if buy_units > 0:
                procurements_by_sku[order.sku] = procurements_by_sku.get(order.sku, 0) + buy_units
    return V3Action(
        central_procurements=[{"sku": sku, "units": units} for sku, units in procurements_by_sku.items()],
        central_replenishments=replenishments,
    )


def _bounded_lookahead_action(env: V3SupplyMindEnv, observation: V3Observation) -> V3Action:
    candidates = _candidate_actions(env, observation)
    best_action = candidates[0]
    best_value = float("-inf")
    for action in candidates:
        value = _simulate_action_value(env, action, LOOKAHEAD_DEPTH)
        if value > best_value:
            best_value = value
            best_action = action
    return best_action


def _simulate_action_value(env: V3SupplyMindEnv, action: V3Action, depth: int) -> float:
    clone = deepcopy(env)
    start_reward = clone.cumulative_reward
    result = clone.step(action, grade_terminal=False)
    if not result.done and depth > 1:
        observation = result.observation
        for _ in range(depth - 1):
            next_action = _oracle_action(clone, observation)
            result = clone.step(next_action, grade_terminal=False)
            observation = result.observation
            if result.done:
                break
    return (clone.cumulative_reward - start_reward) + _terminal_potential(clone)


def _candidate_actions(env: V3SupplyMindEnv, observation: V3Observation) -> list[V3Action]:
    candidates: list[V3Action] = [
        V3Action(),
        baseline_policy(observation),
        heuristic_policy(observation),
        privileged_reference_policy(observation),
        _oracle_action(env, observation),
    ]
    candidates.extend(_shipment_candidates(env, observation))
    candidates.extend(_procurement_candidates(env, observation))
    candidates.extend(_offer_match_candidates(env, observation))
    candidates.extend(_direct_transfer_candidates(env, observation))
    return _dedupe_actions(candidates)[:MAX_CANDIDATES]


def _shipment_candidates(env: V3SupplyMindEnv, observation: V3Observation) -> list[V3Action]:
    recipe = env._require_recipe()
    specs_by_region = {spec.region: spec for spec in recipe.warehouse_specs}
    candidates: list[V3Action] = []
    committed_by_sku: dict[str, int] = {}
    used_trucks = 0
    for order in sorted(
        visible_orders(recipe, env.round_index, env.completed_orders, env.expired_orders),
        key=lambda item: (-item.priority, item.deadline_round, -item.units * item.value_per_unit),
    ):
        if used_trucks >= observation.central_depot.trucks_available:
            break
        spec = specs_by_region.get(order.region)
        if spec is None:
            continue
        available = env.inventory[spec.warehouse_id].get(order.sku, 0)
        missing = max(0, order.units - available)
        depot_left = env.central_depot_inventory.get(order.sku, 0) - committed_by_sku.get(order.sku, 0)
        units = min(missing, depot_left, 3)
        if units <= 0:
            continue
        candidates.append(V3Action(central_replenishments=[{"to_warehouse": spec.warehouse_id, "sku": order.sku, "units": units}]))
        committed_by_sku[order.sku] = committed_by_sku.get(order.sku, 0) + units
        used_trucks += 1
    return candidates


def _procurement_candidates(env: V3SupplyMindEnv, observation: V3Observation) -> list[V3Action]:
    recipe = env._require_recipe()
    if observation.remaining_rounds <= recipe.profile.depot_procurement_lead_time + 2:
        return []
    future_window = env.round_index + recipe.profile.depot_procurement_lead_time + 3
    demand_by_sku: dict[str, int] = {}
    for order in recipe.orders:
        if order.order_id in env.completed_orders or order.order_id in env.expired_orders:
            continue
        if env.round_index < order.created_round <= future_window:
            demand_by_sku[order.sku] = demand_by_sku.get(order.sku, 0) + order.units
    candidates: list[V3Action] = []
    inbound_by_sku: dict[str, int] = {}
    for _, sku, units in env.depot_procurement_returns:
        inbound_by_sku[sku] = inbound_by_sku.get(sku, 0) + units
    for sku, pressure in sorted(demand_by_sku.items(), key=lambda item: -item[1])[:4]:
        depot_units = env.central_depot_inventory.get(sku, 0) + inbound_by_sku.get(sku, 0)
        units = min(max(0, pressure - depot_units), recipe.profile.depot_procurement_cap, 4)
        if units > 0:
            candidates.append(V3Action(central_procurements=[{"sku": sku, "units": units}]))
    return candidates


def _offer_match_candidates(env: V3SupplyMindEnv, observation: V3Observation) -> list[V3Action]:
    recipe = env._require_recipe()
    signals = generate_market_signals(recipe, env.inventory, env.drivers_available, env.trust)
    offers = [signal for signal in signals if signal.signal_type == "inventory_offer" and signal.sku]
    requests = [signal for signal in signals if signal.signal_type == "inventory_request" and signal.sku]
    matches = []
    for request in sorted(requests, key=lambda signal: (-signal.urgency, -signal.units)):
        for offer in sorted(offers, key=lambda signal: signal.ask_price):
            if offer.sku != request.sku or offer.warehouse_id == request.warehouse_id:
                continue
            units = min(offer.units, request.units, 3)
            if units <= 0:
                continue
            matches.append(
                {
                    "offer_signal_id": offer.signal_id,
                    "request_signal_id": request.signal_id,
                    "units": units,
                    "compensation": max(offer.ask_price * units, 1.0),
                }
            )
            break
        if len(matches) >= 3:
            break
    candidates = [V3Action(offer_matches=[match]) for match in matches]
    if len(matches) >= 2:
        candidates.append(V3Action(offer_matches=matches[:2]))
    return candidates


def _direct_transfer_candidates(env: V3SupplyMindEnv, observation: V3Observation) -> list[V3Action]:
    candidates: list[V3Action] = []
    for report in sorted(observation.demand_reports, key=lambda item: (-item.urgency, -item.missed_units_last_round, -item.at_risk_units))[:4]:
        receiver = next((warehouse for warehouse in observation.warehouses if warehouse.warehouse_id == report.warehouse_id), None)
        if receiver is None:
            continue
        donors = [
            warehouse for warehouse in observation.warehouses
            if warehouse.warehouse_id != report.warehouse_id and warehouse.inventory.get(report.sku, 0) >= report.forecast_units + 4
        ]
        if not donors:
            continue
        donor = max(donors, key=lambda warehouse: warehouse.inventory.get(report.sku, 0))
        units = min(2, donor.inventory.get(report.sku, 0), max(1, report.requested_units + report.at_risk_units + report.missed_units_last_round))
        candidates.append(
            V3Action(
                inventory_transfers=[
                    {
                        "from_warehouse": donor.warehouse_id,
                        "to_warehouse": receiver.warehouse_id,
                        "sku": report.sku,
                        "units": units,
                        "compensation": units * 10.0,
                    }
                ]
            )
        )
    return candidates


def _terminal_potential(env: V3SupplyMindEnv) -> float:
    recipe = env._require_recipe()
    potential = 0.0
    open_orders = visible_orders(recipe, env.round_index, env.completed_orders, env.expired_orders)
    specs_by_region = {spec.region: spec for spec in recipe.warehouse_specs}
    for order in open_orders:
        spec = specs_by_region.get(order.region)
        if spec is None:
            continue
        available = env.inventory[spec.warehouse_id].get(order.sku, 0)
        if available < order.units:
            potential -= 0.25 * (order.units - available) * order.value_per_unit
    values = list(env.local_utility.values())
    if values:
        potential -= 0.03 * max(0.0, max(values) - min(values) - 35.0)
    return potential


def _dedupe_actions(actions: list[V3Action]) -> list[V3Action]:
    seen: set[str] = set()
    deduped: list[V3Action] = []
    for action in actions:
        key = json.dumps(action.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)
    return deduped
