from __future__ import annotations

import secrets


TASK_IDS: tuple[str, ...] = (
    "v1_cooperative_restock",
    "v1_scarcity_negotiation",
    "v1_crisis_coalition",
)

TRAIN_SEEDS: dict[str, tuple[int, ...]] = {
    "v1_cooperative_restock": (101, 103, 105, 107, 109, 111),
    "v1_scarcity_negotiation": (201, 203, 205, 207, 209, 211),
    "v1_crisis_coalition": (301, 303, 305, 307, 309, 311),
}

EVAL_SEEDS: dict[str, tuple[int, ...]] = {
    "v1_cooperative_restock": (401, 403, 405, 407),
    "v1_scarcity_negotiation": (501, 503, 505, 507),
    "v1_crisis_coalition": (601, 603, 605, 607),
}

TEST_SEEDS: dict[str, tuple[int, ...]] = {
    "v1_cooperative_restock": (701, 703, 705, 707, 709, 711),
    "v1_scarcity_negotiation": (801, 803, 805, 807, 809, 811),
    "v1_crisis_coalition": (901, 903, 905, 907, 909, 911),
}

SEED_POOLS: dict[str, dict[str, tuple[int, ...]]] = {
    "train": TRAIN_SEEDS,
    "eval": EVAL_SEEDS,
    "official": TEST_SEEDS,
    "test": TEST_SEEDS,
}


def resolve_curated_seed(task_id: str, external_seed: int, pool_name: str = "test") -> int:
    pool = SEED_POOLS[pool_name][task_id]
    return pool[((external_seed * 1315423911) + sum(ord(ch) for ch in task_id)) % len(pool)]


def resolve_task_id(external_seed: int) -> str:
    return TASK_IDS[external_seed % len(TASK_IDS)]


def choose_random_task_id() -> str:
    return secrets.choice(TASK_IDS)


def choose_random_curated_seed(task_id: str, pool_name: str = "test") -> int:
    return secrets.choice(SEED_POOLS[pool_name][task_id])
