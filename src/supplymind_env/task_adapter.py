from __future__ import annotations

PUBLIC_TO_INTERNAL_TASK_ID: dict[str, str] = {
    "cooperative_restock": "v1_cooperative_restock",
    "scarcity_negotiation": "v1_scarcity_negotiation",
    "crisis_coalition": "v1_crisis_coalition",
}

INTERNAL_TO_PUBLIC_TASK_ID: dict[str, str] = {
    internal: public for public, internal in PUBLIC_TO_INTERNAL_TASK_ID.items()
}

PUBLIC_TASK_IDS: tuple[str, ...] = tuple(PUBLIC_TO_INTERNAL_TASK_ID)


def to_internal_task_id(task_id: str) -> str:
    return PUBLIC_TO_INTERNAL_TASK_ID.get(task_id, task_id)


def to_public_task_id(task_id: str) -> str:
    return INTERNAL_TO_PUBLIC_TASK_ID.get(task_id, task_id)


def is_public_task_id(task_id: str) -> bool:
    return task_id in PUBLIC_TO_INTERNAL_TASK_ID

