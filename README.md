---
title: SupplyMind OpenEnv
sdk: docker
app_port: 7860
---

# SupplyMind

**SupplyMind is a multi-agent warehouse orchestration environment for training an LLM central planner to coordinate self-interested warehouses.**

The trainable agent is the central orchestrator. The environment contains a finite central depot plus multiple warehouse agents with local inventory, local drivers, private demand pressure, and hidden local incentives. Warehouses handle local customer orders themselves and report compressed demand pressure. The orchestrator sends limited depot replenishments, approves warehouse offer matches, and proposes inventory transfers; it receives one official global reward plus per-agent diagnostics.

## Theme Fit

Primary theme: **Theme #1 - Multi-Agent Interactions**

SupplyMind includes:
- cooperation through inventory sharing and central depot allocation
- competition between local warehouse safety and global service level
- negotiation through compensation-based transfer proposals
- coalition formation when transfers enable later fulfillment
- partial observability through private warehouse forecasts and public messages

## Public Tasks

- `cooperative_restock`: easier coordination, friendlier warehouses, lower penalties
- `scarcity_negotiation`: tighter stock, more rejection risk, higher stockout pressure
- `crisis_coalition`: more warehouses, more orders, sharper scarcity, higher need for coalition planning

## Observation

The state includes:
- warehouse inventory for `fresh_milk`, `rice_bag_5kg`, `insulin_pack`, and `usb_c_charger`
- central depot inventory, available trucks, and replenishment cap
- available and returning drivers
- public warehouse messages and market signals
- compressed warehouse demand reports instead of raw customer orders
- recent negotiation trace
- reward components and per-agent diagnostics

## Action

The central orchestrator returns strict JSON:

```json
{
  "central_replenishments": [
    {"to_warehouse": "north", "sku": "insulin_pack", "units": 2}
  ],
  "inventory_transfers": [
    {
      "from_warehouse": "west",
      "to_warehouse": "north",
      "sku": "fresh_milk",
      "units": 3,
      "compensation": 18.0
    }
  ],
  "offer_matches": [
    {
      "offer_signal_id": "west:offer:fresh_milk",
      "request_signal_id": "north:request:fresh_milk",
      "units": 2,
      "compensation": 12.0
    }
  ],
  "priority_policy": [],
  "defer_orders": [],
  "coalition_deals": []
}
```

## Reward

The official scalar reward is global welfare:

```text
fulfilled order value
- delivery cost
- central replenishment cost
- transfer cost
- stockout penalty
- late delivery penalty
- holding and waste costs
- fairness penalty
- invalid action penalty
+ accepted trade / coalition bonuses
```

The environment also logs per-warehouse reward diagnostics, proposal outcomes, and reward components for debugging and storytelling. Hidden trust/personality variables influence behavior internally but are not handed to the policy.

Delivery is intentionally simple and local: each warehouse agent chooses which local orders to serve using its own inventory and drivers. The center does not see raw orders. The richer coordination happens upstream: central depot replenishments consume depot stock/trucks, and warehouse offer matches are central approvals that move inventory through bounded transfer capacity.

## Pipeline

Fleetmind-style structure:
- `src/supplymind_env/models.py`: typed observation/action/reward models
- `src/supplymind_env/generator.py`: seeded world recipes
- `src/supplymind_env/environment.py`: `reset/state/step`
- `src/supplymind_env/policies.py`: baseline and heuristic policies
- `src/supplymind_env/grading.py`: normalized score against baseline/reference
- `inference.py`: LLM-first rollout with deterministic fallback
- `validate_submission.py`: manifest/API/inference/Docker checks
- `static/index.html`: small inspection UI

## Run

```bash
python inference.py
python validate_submission.py
uvicorn app:app --host 0.0.0.0 --port 7860
```

Then open `http://127.0.0.1:7860`.
