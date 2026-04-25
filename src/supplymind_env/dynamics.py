from __future__ import annotations

from collections import defaultdict

from .models import (
    HiddenRecipe,
    InventoryTransferProposal,
    MarketSignal,
    NegotiationEvent,
    OrderTemplate,
    WarehouseSpec,
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def visible_orders(recipe: HiddenRecipe, round_index: int, completed: set[str], expired: set[str]) -> list[OrderTemplate]:
    return [
        order
        for order in recipe.orders
        if order.created_round <= round_index and order.order_id not in completed and order.order_id not in expired
    ]


def transfer_cost(specs_by_id: dict[str, WarehouseSpec], proposal: InventoryTransferProposal) -> float:
    source = specs_by_id[proposal.from_warehouse]
    target = specs_by_id[proposal.to_warehouse]
    return 0.75 * source.delivery_cost_by_region[target.region] * proposal.units


def warehouse_message(
    spec: WarehouseSpec,
    inventory: dict[str, int],
    drivers_available: int,
    forecast: dict[str, int],
    trust: float,
) -> str:
    shortages = [
        sku for sku, expected in forecast.items()
        if inventory.get(sku, 0) < spec.safety_stock[sku] + expected
    ]
    surplus = [
        sku for sku, units in inventory.items()
        if units > spec.safety_stock[sku] + forecast.get(sku, 0) + 1
    ]
    if shortages:
        return f"{spec.label}: requests support for {', '.join(shortages)}; drivers={drivers_available}."
    if surplus:
        return f"{spec.label}: can offer limited {', '.join(surplus)} if compensation is fair."
    return f"{spec.label}: prefers to hold inventory; local risk is balanced."


def generate_market_signals(
    recipe: HiddenRecipe,
    inventory_by_warehouse: dict[str, dict[str, int]],
    drivers_available: dict[str, int],
    trust: dict[str, float],
) -> list[MarketSignal]:
    signals: list[MarketSignal] = []
    for spec in recipe.warehouse_specs:
        inventory = inventory_by_warehouse[spec.warehouse_id]
        forecast = recipe.private_forecasts[spec.warehouse_id]
        for sku, units in inventory.items():
            need_line = spec.safety_stock[sku] + forecast.get(sku, 0)
            surplus = units - need_line
            if surplus >= 2:
                signals.append(
                    MarketSignal(
                        signal_id=f"{spec.warehouse_id}:offer:{sku}",
                        warehouse_id=spec.warehouse_id,
                        signal_type="inventory_offer",
                        sku=sku,  # type: ignore[arg-type]
                        units=min(surplus, 4),
                        ask_price=round(_ask_price(spec, sku, trust[spec.warehouse_id]), 2),
                        urgency=1,
                        message=f"{spec.label} offers up to {min(surplus, 4)} {sku}.",
                    )
                )
            elif surplus <= -1:
                signals.append(
                    MarketSignal(
                        signal_id=f"{spec.warehouse_id}:request:{sku}",
                        warehouse_id=spec.warehouse_id,
                        signal_type="inventory_request",
                        sku=sku,  # type: ignore[arg-type]
                        units=min(abs(surplus), 4),
                        urgency=2 if abs(surplus) >= 2 else 1,
                        message=f"{spec.label} requests {min(abs(surplus), 4)} {sku}.",
                    )
                )
        if drivers_available[spec.warehouse_id] >= 2:
            signals.append(
                MarketSignal(
                    signal_id=f"{spec.warehouse_id}:driver_offer",
                    warehouse_id=spec.warehouse_id,
                    signal_type="driver_offer",
                    driver_count=1,
                    ask_price=round(3.0 + (1.0 - trust[spec.warehouse_id]) * 2.0, 2),
                    message=f"{spec.label} can lend 1 driver for the right price.",
                )
            )
        elif drivers_available[spec.warehouse_id] == 0:
            signals.append(
                MarketSignal(
                    signal_id=f"{spec.warehouse_id}:driver_request",
                    warehouse_id=spec.warehouse_id,
                    signal_type="driver_request",
                    driver_count=1,
                    urgency=2,
                    message=f"{spec.label} requests temporary driver capacity.",
                )
            )
    return signals


def _ask_price(spec: WarehouseSpec, sku: str, trust: float) -> float:
    base = {"fresh_milk": 4.0, "rice_bag_5kg": 3.0, "insulin_pack": 7.0, "usb_c_charger": 9.0}[sku]
    personality_markup = {
        "cooperative": 0.85,
        "risk_averse": 1.25,
        "selfish": 1.55,
        "opportunistic": 1.15,
    }[spec.personality]
    return base * personality_markup * (1.15 - 0.25 * trust)


def acceptance_decision(
    spec: WarehouseSpec,
    proposal: InventoryTransferProposal,
    inventory: dict[str, int],
    forecast: dict[str, int],
    trust: float,
) -> tuple[bool, float, str]:
    available = inventory.get(proposal.sku, 0)
    if proposal.units <= 0 or proposal.units > available:
        return False, -1.0, "insufficient inventory"

    remaining = available - proposal.units
    local_need = spec.safety_stock[proposal.sku] + forecast.get(proposal.sku, 0)
    risk_units = max(0, local_need - remaining)
    personality_margin = {
        "cooperative": 0.55,
        "risk_averse": 1.35,
        "selfish": 1.65,
        "opportunistic": 1.05,
    }[spec.personality]
    required_compensation = risk_units * 4.0 * personality_margin
    trust_discount = 1.0 - (0.35 * trust)
    required_compensation *= trust_discount
    if proposal.compensation + 1e-9 >= required_compensation:
        utility_delta = proposal.compensation - (risk_units * 3.0)
        return True, utility_delta, "accepted fair compensation"
    return False, -0.6, f"rejected; wanted compensation >= {required_compensation:.1f}"


def fairness_penalty(agent_rewards: dict[str, float], weight: float) -> float:
    if len(agent_rewards) < 2:
        return 0.0
    values = list(agent_rewards.values())
    spread = max(values) - min(values)
    return -weight * max(0.0, spread - 35.0)


def holding_and_waste_cost(
    inventory_by_warehouse: dict[str, dict[str, int]],
    profile,
) -> tuple[float, float]:
    holding = 0.0
    waste = 0.0
    for inventory in inventory_by_warehouse.values():
        for sku, units in inventory.items():
            holding -= units * profile.holding_cost_per_unit
            if sku == "fresh_milk" and units > 8:
                waste -= (units - 8) * profile.waste_cost_per_unit
    return holding, waste


def add_component(components: dict[str, float], key: str, value: float) -> None:
    components[key] = components.get(key, 0.0) + value


def aggregate_agent_rewards(events: list[NegotiationEvent]) -> dict[str, float]:
    rewards: dict[str, float] = defaultdict(float)
    for event in events:
        rewards[event.actor] += event.local_utility_delta
    return dict(rewards)
