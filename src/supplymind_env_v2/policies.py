from __future__ import annotations

from .config import price_band
from .models import CenterAction, V2JointAction, WarehouseAction


def no_op_policy(observation) -> V2JointAction:
    return V2JointAction()


def naive_joint_policy(observation) -> V2JointAction:
    warehouse_actions = {}
    for warehouse_id, warehouse in observation.warehouses.items():
        warehouse_actions[warehouse_id] = WarehouseAction(
            order_decisions=[{"order_id": order.order_id, "decision": "accept"} for order in warehouse.local_orders if order.status == "pending"],
            transfer_responses=[{"proposal_id": proposal.proposal_id, "decision": "accept"} for proposal in warehouse.pending_transfer_proposals],
        )
    return V2JointAction(warehouse_actions=warehouse_actions, central_action=CenterAction())


def heuristic_joint_policy(observation) -> V2JointAction:
    warehouse_actions = fixed_warehouse_actions(observation)
    center_action = fixed_center_action(observation, warehouse_actions)
    return V2JointAction(warehouse_actions=warehouse_actions, central_action=center_action)


def fixed_warehouse_actions(observation) -> dict[str, WarehouseAction]:
    warehouse_actions = {}
    for warehouse_id, warehouse in observation.warehouses.items():
        order_decisions = []
        for order in warehouse.local_orders:
            if order.status != "pending":
                continue
            enough_stock = warehouse.inventory.get(order.sku, 0) >= order.units
            enough_time = order.deadline_round >= observation.round_index + 1
            decision = "accept" if enough_time and enough_stock else "reject"
            order_decisions.append({"order_id": order.order_id, "decision": decision})
        offers = []
        requests = []
        for sku, units in warehouse.inventory.items():
            safety = warehouse.safety_stock.get(sku, 1)
            if units >= safety + 3:
                offers.append({"sku": sku, "units": min(3, units - safety), "ask_price": price_band(sku)["fair_wholesale_price"]})
            elif units <= safety:
                requests.append({"sku": sku, "units": safety + 2 - units, "max_price": price_band(sku)["max_wholesale_price"]})
        responses = []
        for proposal in warehouse.pending_transfer_proposals:
            remaining = warehouse.inventory.get(proposal.sku, 0) - proposal.units
            required = warehouse.safety_stock.get(proposal.sku, 1)
            fair = proposal.compensation >= proposal.units * price_band(proposal.sku)["fair_wholesale_price"]
            responses.append({"proposal_id": proposal.proposal_id, "decision": "accept" if remaining >= required and fair else "reject"})
        priorities = [{"sku": "insulin_pack", "priority": 3}, {"sku": "fresh_milk", "priority": 2}]
        warehouse_actions[warehouse_id] = WarehouseAction(
            order_decisions=order_decisions,
            inventory_offers=offers,
            inventory_requests=requests,
            transfer_responses=responses,
            local_priority=priorities,
        )
    return warehouse_actions


def fixed_center_action(observation, warehouse_actions: dict[str, WarehouseAction] | None = None) -> CenterAction:
    warehouse_actions = warehouse_actions or {}
    center_replenishments = _targeted_replenishments(observation)
    procurements = _targeted_procurements(observation)
    liquidations = _targeted_liquidations(observation)
    offer_matches = _match_warehouse_signals(warehouse_actions)
    return CenterAction(
        central_procurements=procurements,
        central_liquidations=liquidations,
        central_replenishments=center_replenishments,
        offer_matches=offer_matches,
    )


def _targeted_replenishments(observation) -> list[dict]:
    center_replenishments = []
    used_trucks = 0
    depot_left = dict(observation.center.depot_inventory)
    summaries = sorted(
        observation.center.warehouse_summaries,
        key=lambda item: (item["pending_orders"] + item["accepted_orders"], -sum(item["inventory"].values())),
        reverse=True,
    )
    for summary in summaries:
        if used_trucks >= observation.center.depot_trucks_available:
            break
        inventory = summary["inventory"]
        sku_pressure = sorted(
            observation.center.price_bands.items(),
            key=lambda item: (inventory.get(item[0], 0), -item[1]["customer_value"]),
        )
        for sku, band in sku_pressure:
            if inventory.get(sku, 0) <= 1 and depot_left.get(sku, 0) > 0:
                units = min(2, depot_left[sku])
                center_replenishments.append({"to_warehouse": summary["warehouse_id"], "sku": sku, "units": units, "unit_price": band["fair_wholesale_price"]})
                depot_left[sku] -= units
                used_trucks += 1
                break
    return center_replenishments


def _targeted_procurements(observation) -> list[dict]:
    procurements = []
    if observation.center.remaining_rounds <= 5:
        return procurements
    for sku, units in observation.center.depot_inventory.items():
        if units > 2:
            continue
        band = observation.center.price_bands[sku]
        procurements.append({"sku": sku, "units": 3, "max_unit_cost": band["procurement_cost"]})
        break
    return procurements


def _targeted_liquidations(observation) -> list[dict]:
    if observation.round_index <= 1:
        return []
    fresh = observation.center.depot_inventory.get("fresh_milk", 0)
    if fresh <= 8:
        return []
    visible_milk_pressure = sum(
        summary["pending_orders"] + summary["accepted_orders"]
        for summary in observation.center.warehouse_summaries
        if summary["inventory"].get("fresh_milk", 0) <= 2
    )
    if visible_milk_pressure:
        return []
    return [{"sku": "fresh_milk", "units": min(4, fresh - 8)}]


def _match_warehouse_signals(warehouse_actions: dict[str, WarehouseAction]) -> list[dict]:
    offers = []
    requests = []
    for warehouse_id, action in warehouse_actions.items():
        for offer in action.inventory_offers:
            offers.append({"signal_id": f"{warehouse_id}:offer:{offer.sku}", "warehouse_id": warehouse_id, "sku": offer.sku, "units": offer.units, "price": offer.ask_price})
        for request in action.inventory_requests:
            requests.append({"signal_id": f"{warehouse_id}:request:{request.sku}", "warehouse_id": warehouse_id, "sku": request.sku, "units": request.units, "price": request.max_price})
    matches = []
    used_offer_units: dict[str, int] = {}
    used_request_units: dict[str, int] = {}
    for request in sorted(requests, key=lambda item: -price_band(item["sku"])["customer_value"]):
        for offer in sorted([item for item in offers if item["sku"] == request["sku"] and item["warehouse_id"] != request["warehouse_id"]], key=lambda item: item["price"]):
            offer_left = offer["units"] - used_offer_units.get(offer["signal_id"], 0)
            request_left = request["units"] - used_request_units.get(request["signal_id"], 0)
            units = min(offer_left, request_left, 2)
            if units <= 0 or offer["price"] > request["price"]:
                continue
            compensation = max(offer["price"] * units, price_band(offer["sku"])["fair_wholesale_price"] * units)
            matches.append({"offer_signal_id": offer["signal_id"], "request_signal_id": request["signal_id"], "units": units, "compensation": compensation})
            used_offer_units[offer["signal_id"]] = used_offer_units.get(offer["signal_id"], 0) + units
            used_request_units[request["signal_id"]] = used_request_units.get(request["signal_id"], 0) + units
            break
    return matches


__all__ = ["fixed_center_action", "fixed_warehouse_actions", "heuristic_joint_policy", "naive_joint_policy", "no_op_policy"]
