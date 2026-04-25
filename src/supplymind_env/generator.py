from __future__ import annotations

import random

from .models import DifficultyProfile, HiddenRecipe, OrderTemplate, WarehouseSpec


PROFILES: dict[str, DifficultyProfile] = {
    "v1_cooperative_restock": DifficultyProfile(
        task_id="v1_cooperative_restock",
        warehouse_count=4,
        total_rounds=25,
        transfer_cap=7,
        truck_capacity=8,
        depot_replenishment_cap=5,
        depot_trucks=1,
        starting_trust=0.72,
        stockout_penalty_multiplier=0.75,
        holding_cost_per_unit=0.04,
        waste_cost_per_unit=0.9,
        fairness_penalty_weight=0.18,
        coalition_bonus=1.5,
    ),
    "v1_scarcity_negotiation": DifficultyProfile(
        task_id="v1_scarcity_negotiation",
        warehouse_count=5,
        total_rounds=30,
        transfer_cap=8,
        truck_capacity=8,
        depot_replenishment_cap=6,
        depot_trucks=1,
        starting_trust=0.62,
        stockout_penalty_multiplier=0.95,
        holding_cost_per_unit=0.055,
        waste_cost_per_unit=1.15,
        fairness_penalty_weight=0.28,
        coalition_bonus=2.5,
    ),
    "v1_crisis_coalition": DifficultyProfile(
        task_id="v1_crisis_coalition",
        warehouse_count=7,
        total_rounds=40,
        transfer_cap=10,
        truck_capacity=10,
        depot_replenishment_cap=8,
        depot_trucks=2,
        starting_trust=0.55,
        stockout_penalty_multiplier=1.05,
        holding_cost_per_unit=0.065,
        waste_cost_per_unit=1.35,
        fairness_penalty_weight=0.36,
        coalition_bonus=4.0,
    ),
}


def generate_recipe(task_id: str, seed: int) -> HiddenRecipe:
    profile = PROFILES[task_id]
    rng = random.Random(f"{task_id}:{seed}")
    specs = _warehouse_specs(profile.warehouse_count, rng)
    initial_inventory = _initial_inventory(specs, task_id, rng)
    central_depot_inventory = _central_depot_inventory(task_id, rng)
    initial_drivers = {spec.warehouse_id: 2 if profile.warehouse_count == 3 else rng.choice([1, 2]) for spec in specs}
    private_forecasts = _private_forecasts(specs, task_id, rng)
    orders = _orders(specs, task_id, profile.total_rounds, rng)
    return HiddenRecipe(
        task_id=task_id,
        seed=seed,
        profile=profile,
        warehouse_specs=tuple(specs),
        initial_inventory=initial_inventory,
        central_depot_inventory=central_depot_inventory,
        initial_drivers=initial_drivers,
        private_forecasts=private_forecasts,
        orders=tuple(orders),
    )


def _warehouse_specs(count: int, rng: random.Random) -> list[WarehouseSpec]:
    personalities = ["cooperative", "risk_averse", "selfish", "opportunistic", "risk_averse", "cooperative", "selfish"]
    rng.shuffle(personalities)
    base = [
        ("north", "North", "uptown", {"uptown": 1.0, "downtown": 2.4, "suburb": 3.0, "industrial": 2.8}),
        ("east", "East", "suburb", {"uptown": 2.7, "downtown": 2.1, "suburb": 1.0, "industrial": 2.5}),
        ("south", "South", "downtown", {"uptown": 2.2, "downtown": 1.0, "suburb": 2.5, "industrial": 2.4}),
        ("west", "West", "industrial", {"uptown": 2.8, "downtown": 2.3, "suburb": 2.1, "industrial": 1.0}),
        ("central", "Central", "midtown", {"uptown": 1.7, "downtown": 1.6, "suburb": 2.0, "industrial": 1.8, "midtown": 1.0, "riverside": 2.2, "campus": 1.9}),
        ("riverside", "Riverside", "riverside", {"uptown": 3.0, "downtown": 2.6, "suburb": 2.2, "industrial": 2.4, "midtown": 2.2, "riverside": 1.0, "campus": 2.8}),
        ("campus", "Campus", "campus", {"uptown": 2.4, "downtown": 2.7, "suburb": 2.0, "industrial": 3.1, "midtown": 1.9, "riverside": 2.8, "campus": 1.0}),
    ]
    regions = [item[2] for item in base[:count]]
    specs: list[WarehouseSpec] = []
    for index, (warehouse_id, label, region, costs) in enumerate(base[:count]):
        scoped_costs = {key: float(costs.get(key, 2.8)) for key in regions}
        delivery_time = {key: max(1, round(value)) for key, value in scoped_costs.items()}
        specs.append(
            WarehouseSpec(
                warehouse_id=warehouse_id,
                label=label,
                region=region,
                personality=personalities[index],
                safety_stock={
                    "fresh_milk": rng.choice([2, 3]),
                    "rice_bag_5kg": rng.choice([2, 3]),
                    "insulin_pack": rng.choice([1, 2]),
                    "usb_c_charger": rng.choice([1, 2]),
                },
                delivery_cost_by_region=scoped_costs,
                delivery_time_by_region=delivery_time,
            )
        )
    return specs


def _initial_inventory(specs: list[WarehouseSpec], task_id: str, rng: random.Random) -> dict[str, dict[str, int]]:
    base_units = 7 if task_id == "v1_cooperative_restock" else 6
    inventory: dict[str, dict[str, int]] = {}
    for spec in specs:
        inventory[spec.warehouse_id] = {
            "fresh_milk": max(1, base_units + rng.randint(-2, 3)),
            "rice_bag_5kg": max(1, base_units + rng.randint(-2, 3)),
            "insulin_pack": max(1, base_units - 2 + rng.randint(-1, 3)),
            "usb_c_charger": max(1, base_units - 3 + rng.randint(-1, 3)),
        }
    hot = rng.choice(specs).warehouse_id
    cold = rng.choice([spec.warehouse_id for spec in specs if spec.warehouse_id != hot])
    inventory[hot]["fresh_milk"] = max(1, inventory[hot]["fresh_milk"] - 3)
    inventory[cold]["fresh_milk"] += 3
    return inventory


def _central_depot_inventory(task_id: str, rng: random.Random) -> dict[str, int]:
    base = 8 if task_id == "v1_cooperative_restock" else 7
    if task_id == "v1_crisis_coalition":
        base = 9
    return {
        "fresh_milk": base + rng.randint(-1, 2),
        "rice_bag_5kg": base + rng.randint(-1, 2),
        "insulin_pack": max(3, base - 2 + rng.randint(-1, 2)),
        "usb_c_charger": max(2, base - 3 + rng.randint(-1, 2)),
    }


def _private_forecasts(specs: list[WarehouseSpec], task_id: str, rng: random.Random) -> dict[str, dict[str, int]]:
    multiplier = 1 if task_id == "v1_cooperative_restock" else 2
    return {
        spec.warehouse_id: {
            "fresh_milk": rng.randint(1, 3 + multiplier),
            "rice_bag_5kg": rng.randint(1, 3 + multiplier),
            "insulin_pack": rng.randint(1, 2 + multiplier),
            "usb_c_charger": rng.randint(0, 2 + multiplier),
        }
        for spec in specs
    }


def _orders(
    specs: list[WarehouseSpec],
    task_id: str,
    total_rounds: int,
    rng: random.Random,
) -> list[OrderTemplate]:
    regions = [spec.region for spec in specs]
    if "downtown" not in regions:
        regions.append("downtown")
    count_by_task = {
        "v1_cooperative_restock": 42,
        "v1_scarcity_negotiation": 58,
        "v1_crisis_coalition": 84,
    }
    orders: list[OrderTemplate] = []
    for index in range(count_by_task[task_id]):
        created = min(total_rounds - 1, int(index * total_rounds / count_by_task[task_id]) + rng.choice([0, 0, 1]))
        sku = rng.choices(
            ["fresh_milk", "rice_bag_5kg", "insulin_pack", "usb_c_charger"],
            weights=[0.34, 0.30, 0.22, 0.14],
            k=1,
        )[0]
        units = rng.choice([1, 1, 2, 2, 3])
        if task_id == "v1_crisis_coalition" and index % 6 == 0:
            units = rng.choice([3, 4])
        value_by_sku = {"rice_bag_5kg": 11.0, "fresh_milk": 13.0, "insulin_pack": 22.0, "usb_c_charger": 24.0}
        value = value_by_sku[sku]
        if task_id != "v1_cooperative_restock" and created >= total_rounds // 2:
            value += rng.choice([2.0, 3.5, 4.0])
        deadline = min(total_rounds, created + rng.choice([2, 3, 3, 4]))
        orders.append(
            OrderTemplate(
                order_id=f"o{index + 1}",
                created_round=created,
                region=rng.choice(regions),
                sku=sku,
                units=units,
                value_per_unit=value,
                deadline_round=deadline,
                priority=2 if value >= 12 else 1,
            )
        )
    return sorted(orders, key=lambda order: (order.created_round, order.order_id))
