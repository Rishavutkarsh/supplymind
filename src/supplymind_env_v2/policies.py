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
    warehouse_actions = {}
    for warehouse_id, warehouse in observation.warehouses.items():
        order_decisions = []
        for order in warehouse.local_orders:
            if order.status != "pending":
                continue
            enough_stock = warehouse.inventory.get(order.sku, 0) >= order.units
            enough_time = order.deadline_round >= observation.round_index + 1
            order_decisions.append({"order_id": order.order_id, "decision": "accept" if enough_time and enough_stock else "reject"})
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

    center_replenishments = []
    used_trucks = 0
    for summary in sorted(observation.center.warehouse_summaries, key=lambda item: item["pending_orders"] + item["accepted_orders"], reverse=True):
        if used_trucks >= observation.center.depot_trucks_available:
            break
        inventory = summary["inventory"]
        for sku, band in observation.center.price_bands.items():
            if inventory.get(sku, 0) <= 1 and observation.center.depot_inventory.get(sku, 0) > 0:
                center_replenishments.append({"to_warehouse": summary["warehouse_id"], "sku": sku, "units": min(2, observation.center.depot_inventory[sku]), "unit_price": band["fair_wholesale_price"]})
                used_trucks += 1
                break

    procurements = []
    for sku, units in observation.center.depot_inventory.items():
        if units <= 2 and observation.center.remaining_rounds > 4:
            band = observation.center.price_bands[sku]
            procurements.append({"sku": sku, "units": 3, "max_unit_cost": band["procurement_cost"]})
            break

    return V2JointAction(
        warehouse_actions=warehouse_actions,
        central_action=CenterAction(
            central_procurements=procurements,
            central_replenishments=center_replenishments,
        ),
    )


__all__ = ["heuristic_joint_policy", "naive_joint_policy", "no_op_policy"]
