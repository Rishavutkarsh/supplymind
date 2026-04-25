from __future__ import annotations

from typing import Any

from .config import load_reward_config


def public_rules() -> dict[str, Any]:
    config = load_reward_config()
    return {
        "role_contract": {
            "joint": "Controls both center and warehouses for evaluation.",
            "center": "Controls only central procurement, wholesale replenishment, and transfer coordination.",
            "warehouse": "Controls only local order decisions, inventory offers/requests, transfer responses, and local priorities.",
        },
        "goal": {
            "official_score": "Maximize global welfare. Global welfare is not the sum of center and warehouse rewards because internal payments would be double-counted.",
            "center_reward": "Tracks central business incentive: realized wholesale margin, broker fees, depot costs, and shared network penalties.",
            "warehouse_reward": "Tracks local warehouse incentive: customer sales, compensation, purchase costs, local operations, missed demand, and leftover inventory.",
        },
        "item_prices": config["price_bands"],
        "order_penalties": {
            "rejected_order": f"{config['order_rewards']['reject_penalty_multiplier']} * order_value",
            "accepted_but_missed": f"{config['order_rewards']['accepted_missed_penalty_multiplier']} * order_value",
            "silent_expiry": f"{config['order_rewards']['silent_expiry_penalty_multiplier']} * order_value",
        },
        "operating_costs": {
            "depot_holding": f"{config['center_rewards']['depot_holding_cost_per_unit']} per depot unit per round",
            "warehouse_holding": f"{config['warehouse_rewards']['holding_cost_per_unit']} per warehouse unit per round",
            "terminal_leftover_inventory": f"{config['global_rewards']['terminal_inventory_penalty_multiplier']} * procurement_value_of_remaining_stock",
            "terminal_fairness": f"{config['global_rewards']['terminal_fairness_weight']} * mean_absolute_deviation(per_warehouse_service_rate)",
            "spoilage": "fresh_milk above the spoilage threshold is charged at procurement cost in the current MVP",
            "shipment_delivery_transfer": "route-cost based; route costs and route times are visible in state",
        },
        "transfer_costs": {
            "center_broker_fee": f"center-only: min({config['center_rewards']['transfer_broker_fee_per_unit']} * units, {config['center_rewards']['transfer_broker_fee_compensation_share']} * compensation)",
            "global_transfer_bonus": "0; global reward only benefits from transfers through later fulfillment, lower stockout, or lower waste",
            "rejected_transfer_penalty": config["center_rewards"]["rejected_transfer_penalty"],
            "direct_transfer_cost": "0.8 * source_to_destination_route_cost * units",
            "offer_match_transfer_cost": "0.5 * units",
        },
        "action_space": {
            "joint_action": {
                "warehouse_actions": {
                    "<warehouse_id>": {
                        "order_decisions": [{"order_id": "o17", "decision": "accept|reject"}],
                        "inventory_offers": [{"sku": "fresh_milk", "units": 2, "ask_price": 6.0}],
                        "inventory_requests": [{"sku": "insulin_pack", "units": 3, "max_price": 12.0}],
                        "transfer_responses": [{"proposal_id": "p1", "decision": "accept|reject"}],
                        "local_priority": [{"sku": "insulin_pack", "priority": 3}],
                    }
                },
                "central_action": {
                    "central_procurements": [{"sku": "fresh_milk", "units": 4, "max_unit_cost": 5.0}],
                    "central_replenishments": [{"to_warehouse": "north", "sku": "insulin_pack", "units": 2, "unit_price": 12.0}],
                    "inventory_transfer_proposals": [{"from_warehouse": "west", "to_warehouse": "east", "sku": "rice_bag_5kg", "units": 3, "compensation": 15.0}],
                    "offer_matches": [{"offer_signal_id": "west:offer:rice_bag_5kg", "request_signal_id": "east:request:rice_bag_5kg", "units": 2, "compensation": 12.0}],
                },
            }
        },
        "hidden_from_agent": [
            "future orders",
            "seed recipe internals",
            "reference planner",
            "baseline and target rewards before terminal grading",
        ],
    }


def compact_public_rules() -> dict[str, Any]:
    rules = public_rules()
    return {
        "rules_endpoint": "/v2/rules",
        "official_score": rules["goal"]["official_score"],
        "item_prices": rules["item_prices"],
        "order_penalties": rules["order_penalties"],
        "operating_costs": rules["operating_costs"],
        "transfer_costs": rules["transfer_costs"],
    }
