from __future__ import annotations

import random

from .config import price_band
from .models import DifficultyProfile, HiddenRecipe, OrderTemplate, WarehouseSpec


PROFILES: dict[str, DifficultyProfile] = {
    "v2_train_easy": DifficultyProfile(
        task_id="v2_train_easy",
        warehouse_count=3,
        total_rounds=12,
        depot_trucks=1,
        depot_replenishment_cap=4,
        depot_procurement_cap=4,
        depot_procurement_lead_time=1,
        transfer_cap=5,
    ),
    "v2_train_medium": DifficultyProfile(
        task_id="v2_train_medium",
        warehouse_count=4,
        total_rounds=18,
        depot_trucks=1,
        depot_replenishment_cap=6,
        depot_procurement_cap=6,
        depot_procurement_lead_time=2,
        transfer_cap=8,
    ),
    "v2_train_hard": DifficultyProfile(
        task_id="v2_train_hard",
        warehouse_count=5,
        total_rounds=24,
        depot_trucks=1,
        depot_replenishment_cap=7,
        depot_procurement_cap=7,
        depot_procurement_lead_time=2,
        transfer_cap=10,
    ),
    "v2_easy": DifficultyProfile(
        task_id="v2_easy",
        warehouse_count=3,
        total_rounds=18,
        depot_trucks=1,
        depot_replenishment_cap=5,
        depot_procurement_cap=5,
        depot_procurement_lead_time=1,
        transfer_cap=6,
    ),
    "v2_medium": DifficultyProfile(
        task_id="v2_medium",
        warehouse_count=4,
        total_rounds=26,
        depot_trucks=1,
        depot_replenishment_cap=7,
        depot_procurement_cap=7,
        depot_procurement_lead_time=2,
        transfer_cap=9,
    ),
    "v2_hard": DifficultyProfile(
        task_id="v2_hard",
        warehouse_count=5,
        total_rounds=34,
        depot_trucks=2,
        depot_replenishment_cap=9,
        depot_procurement_cap=9,
        depot_procurement_lead_time=3,
        transfer_cap=12,
    ),
}

TRAINING_PROFILE_IDS = ("v2_train_easy", "v2_train_medium", "v2_train_hard")
BENCHMARK_PROFILE_IDS = ("v2_easy", "v2_medium", "v2_hard")

PUBLIC_TASK_IDS = (
    "train_easy",
    "train_medium",
    "train_hard",
    "easy",
    "medium",
    "hard",
    "cooperative_market",
    "scarcity_market",
    "crisis_market",
)
INTERNAL_BY_PUBLIC = {
    "train_easy": "v2_train_easy",
    "train_medium": "v2_train_medium",
    "train_hard": "v2_train_hard",
    "easy": "v2_easy",
    "medium": "v2_medium",
    "hard": "v2_hard",
    "cooperative_market": "v2_easy",
    "scarcity_market": "v2_medium",
    "crisis_market": "v2_hard",
}
LEGACY_INTERNAL_ALIASES = {
    "v2_cooperative_market": "v2_easy",
    "v2_scarcity_market": "v2_medium",
    "v2_crisis_market": "v2_hard",
}
PUBLIC_BY_INTERNAL = {
    "v2_train_easy": "train_easy",
    "v2_train_medium": "train_medium",
    "v2_train_hard": "train_hard",
    "v2_easy": "easy",
    "v2_medium": "medium",
    "v2_hard": "hard",
}
SEED_POOLS: dict[str, tuple[int, ...]] = {
    "v2_train_easy": (101, 113, 127, 139, 151),
    "v2_train_medium": (211, 223, 239, 251, 263),
    "v2_train_hard": (307, 317, 331, 347, 359),
    "v2_easy": (401, 419, 431, 443, 457),
    "v2_medium": (503, 521, 541, 557, 569),
    "v2_hard": (601, 617, 631, 647, 661),
}


def to_internal_task_id(task_id: str) -> str:
    internal = INTERNAL_BY_PUBLIC.get(task_id, task_id)
    return LEGACY_INTERNAL_ALIASES.get(internal, internal)


def to_public_task_id(task_id: str) -> str:
    return PUBLIC_BY_INTERNAL.get(task_id, task_id)


def sample_seed(task_id: str, entropy: int) -> int:
    internal_task_id = to_internal_task_id(task_id)
    pool = SEED_POOLS.get(internal_task_id, (101, 103, 105, 107, 109))
    return pool[entropy % len(pool)]


def generate_recipe(task_id: str, seed: int) -> HiddenRecipe:
    internal_task_id = to_internal_task_id(task_id)
    profile = PROFILES[internal_task_id]
    rng = random.Random(f"{internal_task_id}:{seed}")
    motif = _motif(profile, seed)
    specs = _warehouse_specs(profile.warehouse_count)
    initial_inventory = _initial_inventory(specs, rng, profile, motif)
    initial_drivers = _initial_drivers(specs, rng, profile)
    central_depot_inventory = _central_depot_inventory(rng, motif)
    orders = _orders(specs, rng, profile, motif)
    public_forecasts = _public_forecasts(specs, profile, motif, orders)
    return HiddenRecipe(
        task_id=internal_task_id,
        seed=seed,
        profile=profile,
        warehouse_specs=tuple(specs),
        initial_inventory=initial_inventory,
        initial_drivers=initial_drivers,
        central_depot_inventory=central_depot_inventory,
        orders=tuple(orders),
        public_forecasts=tuple(public_forecasts),
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


def _motif(profile: DifficultyProfile, seed: int) -> str:
    easy = ("mild_understock", "perishable_pressure", "regional_shift")
    medium = ("regional_shift", "transfer_needed", "premium_burst", "rice_festival")
    hard = ("regional_shift", "transfer_needed", "premium_burst", "tight_sla", "perishable_pressure")
    if profile.task_id in {"v2_train_easy", "v2_easy"}:
        choices = easy
    elif profile.task_id in {"v2_train_medium", "v2_medium"}:
        choices = medium
    else:
        choices = hard
    return choices[seed % len(choices)]


def _central_depot_inventory(rng: random.Random, motif: str) -> dict[str, int]:
    depot = {
        "fresh_milk": rng.randint(4, 6),
        "rice_bag_5kg": rng.randint(7, 11),
        "insulin_pack": rng.randint(5, 8),
        "usb_c_charger": rng.randint(4, 7),
    }
    if motif == "perishable_pressure":
        depot["fresh_milk"] = min(6, depot["fresh_milk"] + 1)
        depot["insulin_pack"] = max(3, depot["insulin_pack"] - 1)
    elif motif == "premium_burst":
        depot["insulin_pack"] += 2
        depot["usb_c_charger"] += 2
    elif motif == "transfer_needed":
        depot["rice_bag_5kg"] = max(4, depot["rice_bag_5kg"] - 2)
    return depot


def _initial_drivers(specs: list[WarehouseSpec], rng: random.Random, profile: DifficultyProfile) -> dict[str, int]:
    if profile.task_id in {"v2_train_easy", "v2_easy"}:
        choices = [2, 2, 3]
    elif profile.task_id in {"v2_train_medium", "v2_medium"}:
        choices = [2, 2, 3, 3]
    else:
        choices = [2, 3, 3]
    return {spec.warehouse_id: rng.choice(choices) for spec in specs}


def _initial_inventory(specs: list[WarehouseSpec], rng: random.Random, profile: DifficultyProfile, motif: str) -> dict[str, dict[str, int]]:
    if profile.task_id == "v2_train_easy":
        base = 5
    elif profile.task_id == "v2_easy":
        base = 6
    else:
        base = 6 if profile.warehouse_count <= 4 else 5
    inventory: dict[str, dict[str, int]] = {}
    for index, spec in enumerate(specs):
        inventory[spec.warehouse_id] = {
            "fresh_milk": max(1, min(6, base - 1 + rng.randint(-2, 1))),
            "rice_bag_5kg": max(1, base + rng.randint(-2, 3)),
            "insulin_pack": max(1, base - 2 + rng.randint(-1, 3)),
            "usb_c_charger": max(1, base - 3 + rng.randint(-1, 3)),
        }
        if motif == "mild_understock" and index == 0:
            inventory[spec.warehouse_id]["fresh_milk"] = max(1, inventory[spec.warehouse_id]["fresh_milk"] - 3)
            inventory[spec.warehouse_id]["rice_bag_5kg"] = max(1, inventory[spec.warehouse_id]["rice_bag_5kg"] - 2)
        elif motif == "transfer_needed" and index == 0:
            inventory[spec.warehouse_id]["insulin_pack"] = max(1, inventory[spec.warehouse_id]["insulin_pack"] - 3)
        elif motif == "transfer_needed" and index == len(specs) - 1:
            inventory[spec.warehouse_id]["insulin_pack"] += 4
        elif motif == "perishable_pressure" and index % 2 == 0:
            inventory[spec.warehouse_id]["fresh_milk"] = min(7, inventory[spec.warehouse_id]["fresh_milk"] + 1)
    return inventory


def _orders(specs: list[WarehouseSpec], rng: random.Random, profile: DifficultyProfile, motif: str) -> list[OrderTemplate]:
    count_by_task = {
        "v2_train_easy": 28,
        "v2_train_medium": 44,
        "v2_train_hard": 62,
        "v2_easy": 50,
        "v2_medium": 70,
        "v2_hard": 94,
    }
    count = count_by_task[profile.task_id]
    burst_rounds = _burst_rounds(rng, profile, motif)
    orders: list[OrderTemplate] = []
    for index in range(count):
        if burst_rounds and rng.random() < _burst_probability(profile, motif, index, count):
            created = min(profile.total_rounds - 1, max(0, rng.choice(burst_rounds) + rng.choice([-1, 0, 0, 0, 1])))
        elif index < max(3, int(count * 0.16)):
            created = rng.choice([0, 0, 1, 1, 2])
        else:
            created = min(profile.total_rounds - 1, int(index * profile.total_rounds / count) + rng.choice([0, 0, 0, 1]))
        if motif == "regional_shift" and index > count * 0.45:
            warehouse = rng.choice(specs[-max(1, len(specs) // 2):])
        elif motif == "premium_burst" and index % 6 == 0:
            warehouse = rng.choice(specs)
        elif motif == "rice_festival" and index > count * 0.35:
            warehouse = rng.choice(specs[: max(1, len(specs) // 2)])
        else:
            warehouse = rng.choice(specs)
        weights = [0.34, 0.30, 0.22, 0.14]
        if motif == "perishable_pressure":
            weights = [0.46, 0.24, 0.18, 0.12]
        elif motif == "premium_burst" and index % 6 == 0:
            weights = [0.10, 0.15, 0.35, 0.40]
        elif motif == "rice_festival":
            weights = [0.18, 0.55, 0.17, 0.10]
        elif motif == "regional_shift" and index > count * 0.45:
            weights = [0.28, 0.22, 0.26, 0.24]
        sku = rng.choices(["fresh_milk", "rice_bag_5kg", "insulin_pack", "usb_c_charger"], weights=weights, k=1)[0]
        units = rng.choice([1, 1, 2, 2, 3])
        sla = rng.choice([2, 3, 4, 5])
        if motif in {"regional_shift", "rice_festival", "premium_burst"} and index > count * 0.35 and index % 4 == 0:
            units = rng.choice([2, 3, 3, 4])
            sla = rng.choice([2, 3])
        if motif == "tight_sla" and index % 5 == 0:
            sla = rng.choice([1, 2])
        if profile.task_id in {"v2_train_hard", "v2_hard"} and index % 7 == 0:
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


def _burst_rounds(rng: random.Random, profile: DifficultyProfile, motif: str) -> list[int]:
    if profile.total_rounds <= 12:
        anchors = [max(1, profile.total_rounds // 2)]
    elif profile.total_rounds <= 24:
        anchors = [profile.total_rounds // 3, (2 * profile.total_rounds) // 3]
    else:
        anchors = [profile.total_rounds // 4, profile.total_rounds // 2, (3 * profile.total_rounds) // 4]
    if motif in {"premium_burst", "tight_sla", "regional_shift", "rice_festival"}:
        anchors.append(max(1, profile.total_rounds // 2 + rng.choice([-2, -1, 1, 2])))
    return sorted({min(profile.total_rounds - 2, max(1, round_index)) for round_index in anchors})


def _burst_probability(profile: DifficultyProfile, motif: str, index: int, count: int) -> float:
    base = 0.20 if profile.warehouse_count <= 3 else 0.26
    if motif in {"premium_burst", "tight_sla", "regional_shift"}:
        base += 0.08
    if motif == "rice_festival":
        base += 0.10
    if index > count * 0.55:
        base += 0.05
    return min(0.42, base)


def _public_forecasts(specs: list[WarehouseSpec], profile: DifficultyProfile, motif: str, orders: tuple[OrderTemplate, ...]) -> list[dict]:
    windows = _forecast_windows(profile)
    forecasts: list[dict] = []
    for start, end in windows:
        future = [order for order in orders if start <= order.created_round <= end]
        if not future:
            continue
        by_pair: dict[tuple[str, str], int] = {}
        for order in future:
            by_pair[(order.warehouse_id, order.sku)] = by_pair.get((order.warehouse_id, order.sku), 0) + order.units
        top = sorted(by_pair.items(), key=lambda item: item[1], reverse=True)[:3]
        for (warehouse_id, sku), units in top:
            confidence = "medium"
            if motif in {"premium_burst", "rice_festival", "perishable_pressure"} and units >= 5:
                confidence = "high"
            elif units <= 2:
                confidence = "low"
            forecasts.append(
                {
                    "available_round": max(0, start - 3),
                    "window_start": start,
                    "window_end": end,
                    "warehouse_id": warehouse_id,
                    "sku": sku,
                    "expected_pressure": "high" if units >= 5 else "medium" if units >= 3 else "low",
                    "confidence": confidence,
                    "hint": _forecast_hint(motif, warehouse_id, sku, start, end),
                }
            )
    return forecasts


def _forecast_windows(profile: DifficultyProfile) -> list[tuple[int, int]]:
    if profile.total_rounds <= 12:
        return [(3, 6), (7, 10)]
    if profile.total_rounds <= 24:
        return [(4, 8), (9, 14), (15, 20)]
    return [(5, 10), (12, 18), (20, 27)]


def _forecast_hint(motif: str, warehouse_id: str, sku: str, start: int, end: int) -> str:
    names = {
        "regional_shift": "regional demand is shifting",
        "transfer_needed": "some warehouses may need cross-fill",
        "premium_burst": "premium item pressure may spike",
        "tight_sla": "short-deadline demand is likely",
        "perishable_pressure": "perishable demand may be lumpy",
        "rice_festival": "bulk pantry demand may cluster",
        "mild_understock": "early stock gaps may matter",
        "stable": "steady demand expected",
    }
    return f"{names.get(motif, 'demand pattern forming')}: watch {warehouse_id} {sku} around rounds {start}-{end}"
