from __future__ import annotations

import random

from .config import price_band
from .models import DifficultyProfile, HiddenRecipe, OrderTemplate, WarehouseSpec


PROFILES: dict[str, DifficultyProfile] = {
    "v2_cooperative_market": DifficultyProfile(
        task_id="v2_cooperative_market",
        warehouse_count=4,
        total_rounds=20,
        depot_trucks=1,
        depot_replenishment_cap=6,
        depot_procurement_cap=6,
        depot_procurement_lead_time=2,
        transfer_cap=8,
    ),
    "v2_scarcity_market": DifficultyProfile(
        task_id="v2_scarcity_market",
        warehouse_count=5,
        total_rounds=28,
        depot_trucks=1,
        depot_replenishment_cap=7,
        depot_procurement_cap=7,
        depot_procurement_lead_time=2,
        transfer_cap=10,
    ),
    "v2_crisis_market": DifficultyProfile(
        task_id="v2_crisis_market",
        warehouse_count=7,
        total_rounds=36,
        depot_trucks=2,
        depot_replenishment_cap=9,
        depot_procurement_cap=9,
        depot_procurement_lead_time=3,
        transfer_cap=12,
    ),
}


PUBLIC_TASK_IDS = ("cooperative_market", "scarcity_market", "crisis_market")
INTERNAL_BY_PUBLIC = {
    "cooperative_market": "v2_cooperative_market",
    "scarcity_market": "v2_scarcity_market",
    "crisis_market": "v2_crisis_market",
}
PUBLIC_BY_INTERNAL = {value: key for key, value in INTERNAL_BY_PUBLIC.items()}


def to_internal_task_id(task_id: str) -> str:
    return INTERNAL_BY_PUBLIC.get(task_id, task_id)


def to_public_task_id(task_id: str) -> str:
    return PUBLIC_BY_INTERNAL.get(task_id, task_id)


def generate_recipe(task_id: str, seed: int) -> HiddenRecipe:
    internal_task_id = to_internal_task_id(task_id)
    profile = PROFILES[internal_task_id]
    rng = random.Random(f"{internal_task_id}:{seed}")
    specs = _warehouse_specs(profile.warehouse_count)
    initial_inventory = _initial_inventory(specs, rng, profile)
    initial_drivers = {spec.warehouse_id: rng.choice([1, 2]) for spec in specs}
    central_depot_inventory = {
        "fresh_milk": rng.randint(7, 10),
        "rice_bag_5kg": rng.randint(7, 11),
        "insulin_pack": rng.randint(5, 8),
        "usb_c_charger": rng.randint(4, 7),
    }
    orders = _orders(specs, rng, profile)
    return HiddenRecipe(
        task_id=internal_task_id,
        seed=seed,
        profile=profile,
        warehouse_specs=tuple(specs),
        initial_inventory=initial_inventory,
        initial_drivers=initial_drivers,
        central_depot_inventory=central_depot_inventory,
        orders=tuple(orders),
    )


def _warehouse_specs(count: int) -> list[WarehouseSpec]:
    base = [
        ("north", "North", "uptown", {"uptown": 1.0, "suburb": 3.0, "downtown": 2.4, "industrial": 2.8, "midtown": 1.8, "riverside": 3.0, "campus": 2.4}),
        ("east", "East", "suburb", {"uptown": 2.7, "suburb": 1.0, "downtown": 2.1, "industrial": 2.5, "midtown": 2.0, "riverside": 2.2, "campus": 2.0}),
        ("south", "South", "downtown", {"uptown": 2.2, "suburb": 2.5, "downtown": 1.0, "industrial": 2.4, "midtown": 1.6, "riverside": 2.6, "campus": 2.7}),
        ("west", "West", "industrial", {"uptown": 2.8, "suburb": 2.1, "downtown": 2.3, "industrial": 1.0, "midtown": 1.8, "riverside": 2.4, "campus": 3.1}),
        ("central", "Central", "midtown", {"uptown": 1.7, "suburb": 2.0, "downtown": 1.6, "industrial": 1.8, "midtown": 1.0, "riverside": 2.2, "campus": 1.9}),
        ("riverside", "Riverside", "riverside", {"uptown": 3.0, "suburb": 2.2, "downtown": 2.6, "industrial": 2.4, "midtown": 2.2, "riverside": 1.0, "campus": 2.8}),
        ("campus", "Campus", "campus", {"uptown": 2.4, "suburb": 2.0, "downtown": 2.7, "industrial": 3.1, "midtown": 1.9, "riverside": 2.8, "campus": 1.0}),
    ][:count]
    regions = [item[2] for item in base]
    specs: list[WarehouseSpec] = []
    for warehouse_id, label, region, costs in base:
        scoped = {key: float(costs[key]) for key in regions}
        specs.append(
            WarehouseSpec(
                warehouse_id=warehouse_id,
                label=label,
                region=region,
                safety_stock={"fresh_milk": 2, "rice_bag_5kg": 2, "insulin_pack": 1, "usb_c_charger": 1},
                route_costs=scoped,
                route_times={key: max(1, round(value)) for key, value in scoped.items()},
            )
        )
    return specs


def _initial_inventory(specs: list[WarehouseSpec], rng: random.Random, profile: DifficultyProfile) -> dict[str, dict[str, int]]:
    base = 6 if profile.warehouse_count <= 4 else 5
    inventory: dict[str, dict[str, int]] = {}
    for spec in specs:
        inventory[spec.warehouse_id] = {
            "fresh_milk": max(1, base + rng.randint(-2, 3)),
            "rice_bag_5kg": max(1, base + rng.randint(-2, 3)),
            "insulin_pack": max(1, base - 2 + rng.randint(-1, 3)),
            "usb_c_charger": max(1, base - 3 + rng.randint(-1, 3)),
        }
    return inventory


def _orders(specs: list[WarehouseSpec], rng: random.Random, profile: DifficultyProfile) -> list[OrderTemplate]:
    count = {4: 34, 5: 48, 7: 72}[profile.warehouse_count]
    orders: list[OrderTemplate] = []
    for index in range(count):
        created = min(profile.total_rounds - 1, int(index * profile.total_rounds / count) + rng.choice([0, 0, 1]))
        warehouse = rng.choice(specs)
        sku = rng.choices(["fresh_milk", "rice_bag_5kg", "insulin_pack", "usb_c_charger"], weights=[0.34, 0.30, 0.22, 0.14], k=1)[0]
        units = rng.choice([1, 1, 2, 2, 3])
        sla = rng.choice([2, 3, 4, 5])
        if profile.task_id == "v2_crisis_market" and index % 7 == 0:
            units = rng.choice([3, 4])
            sla = rng.choice([1, 2, 3])
        orders.append(
            OrderTemplate(
                order_id=f"o{index + 1}",
                created_round=created,
                warehouse_id=warehouse.warehouse_id,
                sku=sku,
                units=units,
                customer_value_per_unit=price_band(sku)["customer_value"],
                deadline_round=min(profile.total_rounds, created + sla),
            )
        )
    return tuple(sorted(orders, key=lambda order: (order.created_round, order.order_id)))
