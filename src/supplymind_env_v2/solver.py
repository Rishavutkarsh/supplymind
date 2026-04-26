from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from .config import price_band
from .environment import V2SupplyMindEnv
from .models import CenterAction, V2JointAction, WarehouseAction
from .policies import heuristic_joint_policy, naive_joint_policy


HORIZON = 7
BEAM_DEPTH = 2
BEAM_WIDTH = 2
MAX_CANDIDATES = 10
FOLLOW_UP_CANDIDATES = 4
HARD_BEAM_WIDTH = 3
HARD_MAX_CANDIDATES = 12
HARD_FOLLOW_UP_CANDIDATES = 5
MEDIUM_BEAM_WIDTH = 3
MEDIUM_MAX_CANDIDATES = 12
MEDIUM_FOLLOW_UP_CANDIDATES = 5


def rollout_reference(task_id: str, seed: int) -> float:
    return max(
        _rollout_planner(task_id, seed),
        _rollout_policy(task_id, seed, naive_joint_policy),
        _rollout_policy(task_id, seed, heuristic_joint_policy),
    )


def privileged_reference_policy(observation) -> V2JointAction:
    env = V2SupplyMindEnv(default_task_id=observation.task_id)
    env.reset_internal(observation.task_id, observation.scenario_info["seed"])
    while env.round_index < observation.round_index and not env.done:
        result = env.step(_best_action(env, env.state()), grade_terminal=False)
    return _best_action(env, observation)


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
        result = env.step(_best_action(env, observation), grade_terminal=False)
        observation = result.observation
    return env.cumulative_reward


def _best_action(env: V2SupplyMindEnv, observation) -> V2JointAction:
    candidates = _centralized_candidates(env, observation, HORIZON)
    best = candidates[0]
    best_value = float("-inf")
    for action in candidates:
        clone = deepcopy(env)
        before = clone.cumulative_reward
        result = clone.step(action, grade_terminal=False)
        value = clone.cumulative_reward - before
        if not result.done:
            value += _beam_value(clone, result.observation, BEAM_DEPTH)
        if value > best_value:
            best_value = value
            best = action
    return best


def _beam_value(env: V2SupplyMindEnv, observation, depth: int) -> float:
    start = env.cumulative_reward
    beam_width = _beam_width(env)
    follow_up_candidates = _follow_up_candidates(env)
    beam: list[tuple[float, V2SupplyMindEnv, object]] = [(env.cumulative_reward, env, observation)]
    for remaining in range(depth, 0, -1):
        expanded: list[tuple[float, V2SupplyMindEnv, object]] = []
        for _, state_env, state_observation in beam:
            if state_env.done:
                expanded.append((state_env.cumulative_reward, state_env, state_observation))
                continue
            candidates = _centralized_candidates(state_env, state_observation, max(3, remaining + 2))[:follow_up_candidates]
            for action in candidates:
                clone = deepcopy(state_env)
                result = clone.step(action, grade_terminal=False)
                expanded.append((clone.cumulative_reward, clone, result.observation))
        if not expanded:
            break
        expanded.sort(key=lambda item: item[0], reverse=True)
        beam = expanded[:beam_width]
    return max((item[0] for item in beam), default=start) - start


@dataclass(frozen=True)
class Demand:
    units: int = 0
    value: float = 0.0
    urgent_units: int = 0


@dataclass(frozen=True)
class Mode:
    name: str
    demand_cover: float
    safety_buffer: int
    accept_slack: int
    allow_procurement: bool = True
    allow_transfers: bool = True
    allow_liquidation: bool = True


MODES = (
    Mode("balanced", 0.75, 1, 1),
    Mode("service_aggressive", 1.00, 1, 0),
    Mode("burst_buffer", 1.10, 2, 1),
    Mode("cost_conservative", 0.55, 0, 1, allow_procurement=False),
    Mode("transfer_first", 0.90, 1, 1, allow_procurement=False),
    Mode("perishable_guard", 0.65, 0, 1, allow_liquidation=True),
)


def _centralized_candidates(env: V2SupplyMindEnv, observation, horizon: int) -> list[V2JointAction]:
    pressure = _future_demand(env, observation, horizon)
    candidates = [
        _centralized_action(observation, pressure, mode)
        for mode in MODES
    ]
    candidates.append(heuristic_joint_policy(observation))
    candidates.append(naive_joint_policy(observation))
    candidates.append(V2JointAction())
    return _dedupe(candidates)[:_candidate_limit(env)]


def _is_hard(env: V2SupplyMindEnv) -> bool:
    return "hard" in env.internal_task_id


def _is_medium(env: V2SupplyMindEnv) -> bool:
    return "medium" in env.internal_task_id


def _beam_width(env: V2SupplyMindEnv) -> int:
    if _is_hard(env):
        return HARD_BEAM_WIDTH
    if _is_medium(env):
        return MEDIUM_BEAM_WIDTH
    return BEAM_WIDTH


def _follow_up_candidates(env: V2SupplyMindEnv) -> int:
    if _is_hard(env):
        return HARD_FOLLOW_UP_CANDIDATES
    if _is_medium(env):
        return MEDIUM_FOLLOW_UP_CANDIDATES
    return FOLLOW_UP_CANDIDATES


def _candidate_limit(env: V2SupplyMindEnv) -> int:
    if _is_hard(env):
        return HARD_MAX_CANDIDATES
    if _is_medium(env):
        return MEDIUM_MAX_CANDIDATES
    return MAX_CANDIDATES


def _future_demand(env: V2SupplyMindEnv, observation, horizon: int) -> dict[tuple[str, str], Demand]:
    recipe = env._require_recipe()
    end_round = min(recipe.profile.total_rounds - 1, observation.round_index + horizon)
    demand: dict[tuple[str, str], Demand] = {}
    for order in recipe.orders:
        status = env.order_status.get(order.order_id)
        if status in {"fulfilled", "rejected", "expired"}:
            continue
        if order.created_round > end_round or order.deadline_round < observation.round_index:
            continue
        key = (order.warehouse_id, order.sku)
        current = demand.get(key, Demand())
        urgent = order.units if order.deadline_round <= observation.round_index + 2 else 0
        demand[key] = Demand(
            units=current.units + order.units,
            value=current.value + order.units * order.customer_value_per_unit,
            urgent_units=current.urgent_units + urgent,
        )
    return demand


def _centralized_action(observation, demand: dict[tuple[str, str], Demand], mode: Mode) -> V2JointAction:
    projected = {
        warehouse_id: dict(warehouse.inventory)
        for warehouse_id, warehouse in observation.warehouses.items()
    }
    depot_left = dict(observation.center.depot_inventory)

    shipments = _plan_shipments(observation, demand, mode, projected, depot_left)
    transfers = _plan_offer_matches(observation, demand, mode, projected)
    procurements = _plan_procurements(observation, demand, mode, projected, depot_left)
    liquidations = _plan_liquidations(observation, demand, mode, depot_left)
    warehouse_actions = _plan_warehouse_actions(observation, mode, projected, transfers)

    return V2JointAction(
        warehouse_actions=warehouse_actions,
        central_action=CenterAction(
            central_procurements=procurements,
            central_liquidations=liquidations,
            central_replenishments=shipments,
            offer_matches=transfers,
        ),
    )


def _plan_shipments(observation, demand: dict[tuple[str, str], Demand], mode: Mode, projected: dict[str, dict[str, int]], depot_left: dict[str, int]) -> list[dict]:
    if observation.center.depot_trucks_available <= 0:
        return []
    rows = []
    for (warehouse_id, sku), item in demand.items():
        current = projected.get(warehouse_id, {}).get(sku, 0)
        target = int(item.units * mode.demand_cover) + mode.safety_buffer
        gap = max(0, target - current)
        if gap <= 0 or depot_left.get(sku, 0) <= 0:
            continue
        rows.append(
            {
                "score": item.value + 5.0 * item.urgent_units - 1.5 * current,
                "warehouse_id": warehouse_id,
                "sku": sku,
                "units": gap,
            }
        )
    shipments: list[dict] = []
    used_trucks = 0
    for row in sorted(rows, key=lambda value: value["score"], reverse=True):
        if used_trucks >= observation.center.depot_trucks_available:
            break
        sku = row["sku"]
        units = min(row["units"], depot_left.get(sku, 0), 3)
        if units <= 0:
            continue
        shipments.append(
            {
                "to_warehouse": row["warehouse_id"],
                "sku": sku,
                "units": units,
                "unit_price": price_band(sku)["fair_wholesale_price"],
            }
        )
        depot_left[sku] -= units
        projected[row["warehouse_id"]][sku] = projected[row["warehouse_id"]].get(sku, 0) + units
        used_trucks += 1
    return shipments


def _plan_offer_matches(observation, demand: dict[tuple[str, str], Demand], mode: Mode, projected: dict[str, dict[str, int]]) -> list[dict]:
    if not mode.allow_transfers:
        return []
    offers: list[dict] = []
    requests: list[dict] = []
    warehouse_actions_for_signals: dict[str, WarehouseAction] = {}
    for warehouse_id, inventory in projected.items():
        offer_rows = []
        request_rows = []
        for sku, units in inventory.items():
            need = demand.get((warehouse_id, sku), Demand()).units
            if units >= need + 4:
                offer_rows.append({"sku": sku, "units": min(3, units - need - 2), "ask_price": price_band(sku)["fair_wholesale_price"]})
            elif need > units:
                request_rows.append({"sku": sku, "units": min(3, need - units), "max_price": price_band(sku)["max_wholesale_price"]})
        warehouse_actions_for_signals[warehouse_id] = WarehouseAction(inventory_offers=offer_rows, inventory_requests=request_rows)
        for offer in offer_rows:
            offers.append({"signal_id": f"{warehouse_id}:offer:{offer['sku']}", "warehouse_id": warehouse_id, **offer})
        for request in request_rows:
            requests.append({"signal_id": f"{warehouse_id}:request:{request['sku']}", "warehouse_id": warehouse_id, **request})

    matches = []
    for request in sorted(requests, key=lambda row: -price_band(row["sku"])["customer_value"]):
        same_sku_offers = [
            offer for offer in offers
            if offer["sku"] == request["sku"] and offer["warehouse_id"] != request["warehouse_id"] and offer["ask_price"] <= request["max_price"]
        ]
        same_sku_offers.sort(key=lambda row: row["ask_price"])
        for offer in same_sku_offers:
            units = min(2, offer["units"], request["units"], projected[offer["warehouse_id"]].get(offer["sku"], 0))
            if units <= 0:
                continue
            matches.append(
                {
                    "offer_signal_id": offer["signal_id"],
                    "request_signal_id": request["signal_id"],
                    "units": units,
                    "compensation": units * price_band(offer["sku"])["fair_wholesale_price"],
                }
            )
            projected[offer["warehouse_id"]][offer["sku"]] -= units
            projected[request["warehouse_id"]][request["sku"]] = projected[request["warehouse_id"]].get(request["sku"], 0) + units
            offer["units"] -= units
            request["units"] -= units
            break
        if len(matches) >= 3:
            break
    return matches


def _plan_procurements(observation, demand: dict[tuple[str, str], Demand], mode: Mode, projected: dict[str, dict[str, int]], depot_left: dict[str, int]) -> list[dict]:
    if not mode.allow_procurement or observation.center.remaining_rounds <= 4:
        return []
    network_stock = dict(depot_left)
    for inventory in projected.values():
        for sku, units in inventory.items():
            network_stock[sku] = network_stock.get(sku, 0) + units
    demand_by_sku: dict[str, int] = {}
    value_by_sku: dict[str, float] = {}
    for (_, sku), item in demand.items():
        demand_by_sku[sku] = demand_by_sku.get(sku, 0) + item.units
        value_by_sku[sku] = value_by_sku.get(sku, 0.0) + item.value
    rows = []
    for sku, units in demand_by_sku.items():
        gap = int(units * mode.demand_cover) + mode.safety_buffer - network_stock.get(sku, 0)
        if gap > 0:
            rows.append({"sku": sku, "gap": gap, "score": value_by_sku.get(sku, 0.0)})
    procurements = []
    for row in sorted(rows, key=lambda value: value["score"], reverse=True)[:1]:
        sku = row["sku"]
        procurements.append({"sku": sku, "units": min(5, row["gap"]), "max_unit_cost": price_band(sku)["procurement_cost"]})
    return procurements


def _plan_liquidations(observation, demand: dict[tuple[str, str], Demand], mode: Mode, depot_left: dict[str, int]) -> list[dict]:
    if not mode.allow_liquidation:
        return []
    milk = depot_left.get("fresh_milk", 0)
    if milk <= 3:
        return []
    age = observation.center.depot_inventory_age.get("fresh_milk", 0.0)
    milk_demand = sum(item.units for (_, sku), item in demand.items() if sku == "fresh_milk")
    if age < 3.0 and milk <= milk_demand + 2:
        return []
    excess = max(0, milk - max(2, milk_demand))
    return [] if excess <= 0 else [{"sku": "fresh_milk", "units": min(4, excess)}]


def _plan_warehouse_actions(observation, mode: Mode, projected: dict[str, dict[str, int]], matches: list[dict]) -> dict[str, WarehouseAction]:
    # Recreate offer/request signals required by same-step matches.
    offers_by_wh: dict[str, list[dict]] = {warehouse_id: [] for warehouse_id in observation.warehouses}
    requests_by_wh: dict[str, list[dict]] = {warehouse_id: [] for warehouse_id in observation.warehouses}
    for match in matches:
        offer_wh, _, sku = match["offer_signal_id"].split(":")
        request_wh, _, _ = match["request_signal_id"].split(":")
        offers_by_wh[offer_wh].append({"sku": sku, "units": match["units"], "ask_price": price_band(sku)["fair_wholesale_price"]})
        requests_by_wh[request_wh].append({"sku": sku, "units": match["units"], "max_price": price_band(sku)["max_wholesale_price"]})

    actions: dict[str, WarehouseAction] = {}
    for warehouse_id, warehouse in observation.warehouses.items():
        inventory = dict(projected[warehouse_id])
        driver_slots = warehouse.drivers_available
        decisions = []
        visible = sorted(
            [order for order in warehouse.local_orders if order.status == "pending"],
            key=lambda order: (order.deadline_round, -order.units * order.customer_value_per_unit),
        )
        for order in visible:
            can_wait = order.deadline_round > observation.round_index + mode.accept_slack
            can_fill = inventory.get(order.sku, 0) >= order.units and driver_slots > 0
            if can_fill:
                decisions.append({"order_id": order.order_id, "decision": "accept"})
                inventory[order.sku] -= order.units
                driver_slots -= 1
            elif not can_wait:
                decisions.append({"order_id": order.order_id, "decision": "reject"})

        responses = [
            {"proposal_id": proposal.proposal_id, "decision": "accept"}
            for proposal in warehouse.pending_transfer_proposals
            if inventory.get(proposal.sku, 0) - proposal.units >= warehouse.safety_stock.get(proposal.sku, 1)
        ]
        actions[warehouse_id] = WarehouseAction(
            order_decisions=decisions,
            inventory_offers=offers_by_wh[warehouse_id],
            inventory_requests=requests_by_wh[warehouse_id],
            transfer_responses=responses,
            local_priority=[{"sku": "insulin_pack", "priority": 3}, {"sku": "fresh_milk", "priority": 2}],
        )
    return actions


def _dedupe(actions: list[V2JointAction]) -> list[V2JointAction]:
    unique = []
    seen = set()
    for action in actions:
        key = action.model_dump_json()
        if key in seen:
            continue
        unique.append(action)
        seen.add(key)
    return unique
