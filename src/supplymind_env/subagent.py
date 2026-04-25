from __future__ import annotations

import json

from .models import V3Action, V3Observation
from .policies import heuristic_policy


SUBAGENT_SYSTEM_PROMPT = """You are a small central-planning subagent playing SupplyMind.
You do not see raw customer orders. You only see warehouse reports, depot stock, route costs, market signals, and reward feedback.

Goal:
Maximize global network welfare over the full episode, not just this round.

Action JSON schema:
{
  "central_procurements": [{"sku": "fresh_milk", "units": 2}],
  "central_replenishments": [{"to_warehouse": "north", "sku": "insulin_pack", "units": 2}],
  "offer_matches": [{"offer_signal_id": "west:offer:fresh_milk", "request_signal_id": "north:request:fresh_milk", "units": 2, "compensation": 12.0}],
  "inventory_transfers": [{"from_warehouse": "west", "to_warehouse": "north", "sku": "fresh_milk", "units": 2, "compensation": 12.0}],
  "priority_policy": [],
  "defer_orders": [],
  "coalition_deals": []
}

Policy:
1. Ship depot stock to urgent warehouses when trucks and depot inventory are available.
2. Buy into central depot when repeated forecast pressure is high and depot stock is low; purchases arrive after lead time.
3. Prefer offer_matches over direct transfers when public offers and requests already exist.
4. Do not overbuy late in the episode.
5. Avoid transfers that look unlikely to satisfy the source warehouse's local incentive.
Return JSON only."""


def build_subagent_prompt(observation: V3Observation) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SUBAGENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(observation.model_dump(mode="json"), separators=(",", ":")),
        },
    ]


def prompted_subagent_policy(observation: V3Observation) -> V3Action:
    """A local stand-in for the small prompted model, useful when no API token is set."""
    return heuristic_policy(observation)


__all__ = ["SUBAGENT_SYSTEM_PROMPT", "build_subagent_prompt", "prompted_subagent_policy"]
