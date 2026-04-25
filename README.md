---
title: SupplyMind OpenEnv
sdk: docker
app_port: 7860
---

# SupplyMind

SupplyMind is a multi-agent warehouse orchestration environment for training an LLM to coordinate a supply network under partial observability.

The central idea is simple: local warehouses see and serve their own customer demand, but the central orchestrator does not see every individual order. Instead, warehouses report compressed demand pressure, surplus, shortage, and forecast signals. The central orchestrator must decide when to hold inventory, when to send stock from the central depot, and when to approve warehouse-to-warehouse transfers.

## Why This Exists

Many real operational decisions are not question-answer tasks. They are repeated allocation problems:

- demand appears over time
- inventory is scarce
- moving stock costs money
- local teams have their own constraints
- acting too early can waste stock
- acting too late causes missed demand

SupplyMind turns that pattern into an OpenEnv-compatible RL environment for LLM agents.

## Hackathon Theme Fit

Primary theme: **Theme #1 - Multi-Agent Interactions**

SupplyMind includes:

- **cooperation** through inventory sharing and central depot allocation
- **competition** between local warehouse safety and global network performance
- **negotiation** through compensation-based transfer proposals
- **coalition formation** when several warehouses help satisfy future demand
- **partial observability** because the central agent sees reports, not raw orders

## Agents

### Central Orchestrator

This is the trainable/evaluated LLM policy.

It observes:

- central depot inventory
- depot truck capacity
- warehouse inventory summaries
- warehouse demand reports
- public market signals from warehouses
- previous reward and event feedback

It does **not** observe raw customer orders.

It can act by:

- buying new inventory into the central depot, with procurement caps and lead time
- sending stock from the central depot to warehouses
- approving warehouse offer/request matches
- proposing direct inventory transfers
- choosing to hold inventory for later

### Local Warehouses

Warehouses are environment agents. They are not full LLM agents yet, but the architecture treats them as local decision makers.

Each warehouse has:

- local inventory
- local drivers
- local customer demand
- private forecasts
- safety stock preferences
- hidden behavior parameters
- local utility diagnostics

Warehouses handle their own local fulfillment. They decide which local orders to serve using their own inventory and drivers. The central agent only sees the aggregated consequences through reports and rewards.

### Central Depot

The central depot has finite inventory, limited trucks, and a limited procurement budget each round. It can buy new stock, but purchases arrive after a lead time. It can also supply warehouses, but every shipment consumes depot stock and truck capacity.

This makes forecasting and holding inventory real strategic choices.

## Items

The environment currently tracks four concrete SKUs:

- `fresh_milk`
- `rice_bag_5kg`
- `insulin_pack`
- `usb_c_charger`

These were chosen to create different operational tradeoffs: perishability, staple demand, high-priority health demand, and higher-value retail demand.

## Public Tasks

SupplyMind exposes three public tasks:

- `cooperative_restock`: easy, 4 warehouses, 25 rounds
- `scarcity_negotiation`: medium, 5 warehouses, 30 rounds
- `crisis_coalition`: hard, 7 warehouses, 40 rounds

Each task is generated deterministically from a seed. The same task and seed recreate the same world.

## Episode Loop

Each round works like this:

1. Depot trucks and local drivers return if their trips are complete.
2. Previously purchased depot inventory arrives if its lead time has elapsed.
3. Warehouses publish compressed demand reports and market signals.
4. The central orchestrator submits an action.
5. Depot procurements buy future central stock.
6. Depot replenishments consume depot stock and trucks.
7. Approved offer matches become proposed warehouse transfers.
8. Warehouses accept or reject transfers based on hidden local incentives.
9. Local warehouses fulfill their own local customer orders.
10. Missed or late demand is penalized.
11. Reward components and diagnostics are returned.

## Observation

The central policy receives structured JSON containing:

- `central_depot`: depot inventory, available trucks, returning trucks, inbound purchases, shipment cap, procurement cap
- `warehouses`: inventory, driver availability, route costs, public message
- `demand_reports`: compressed truthful reports of requested, forecast, missed, and at-risk units
- `market_signals`: inventory offers and requests
- `feedback`: reward components, recent events, negotiation trace, local utility diagnostics
- `scenario_info`: task id, seed, round limits, action brief

Raw customer orders are intentionally hidden from the central agent.

## Action Space

The central orchestrator returns strict JSON:

```json
{
  "central_procurements": [
    {
      "sku": "insulin_pack",
      "units": 3
    }
  ],
  "central_replenishments": [
    {
      "to_warehouse": "north",
      "sku": "insulin_pack",
      "units": 2
    }
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

The most important actions today are `central_procurements`, `central_replenishments`, `inventory_transfers`, and `offer_matches`.

## Reward

The official reward is one scalar global welfare score per step.

```text
step_reward =
  fulfilled_order_value
+ accepted_trade_bonus
+ coalition_bonus
+ central_replenishment_bonus
+ strategic_procurement_bonus
+ priority_alignment_bonus
- delivery_cost
- central_procurement_cost
- central_replenishment_cost
- transfer_cost
- stockout_penalty
- late_delivery_penalty
- holding_cost
- waste_penalty
- fairness_penalty
- rejected_trade_penalty
- invalid_action_penalty
```

The code stores negative terms as negative components, then computes:

```python
step_reward = sum(reward_components.values())
```

The UI shows the component breakdown for each round.

## Grading

Cases are generated procedurally from curated seeds. They are deterministic, but not hand-solved.

The score is normalized against:

- a simple baseline policy
- a negotiation heuristic
- a privileged reference policy with extra internal information

The reference is a strong anchor, not a mathematically proven optimum.

## Current Evaluation Snapshot

Recent local evaluation over the seeded policy suite:

```text
no_op:                mean_score 0.0718
reactive_baseline:    mean_score 0.0500
negotiation_heuristic: mean_score 0.4011
privileged_reference: mean_score 0.9999
```

The long-horizon hard task is intentionally challenging. The current fallback heuristic is not final; it is mainly a reproducible baseline for comparison.

## UI Demo

The local UI is served at:

```text
http://127.0.0.1:7860/
```

The UI shows:

- depot inventory and truck status
- warehouse inventory and route costs
- demand reports
- market signals
- central action JSON
- episode timeline
- reward breakdown per round

Use **Run Episode** to watch the central policy act across the full episode.

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API and UI:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Run a deterministic inference rollout:

```bash
python inference.py
```

Run policy evaluation:

```bash
python scripts/evaluate_policies.py
```

Run validation:

```bash
python validate_submission.py
```

## Project Structure

```text
src/supplymind_env/models.py       typed observation/action/reward models
src/supplymind_env/generator.py    seeded world generation
src/supplymind_env/environment.py  reset/state/step environment logic
src/supplymind_env/dynamics.py     reward helpers and market signals
src/supplymind_env/policies.py     baseline and heuristic policies
src/supplymind_env/solver.py       privileged reference rollout
src/supplymind_env/grading.py      normalized scoring
src/supplymind_env/api.py          FastAPI app
static/index.html                  inspection UI
inference.py                       LLM-first inference with fallback
validate_submission.py             packaging and contract checks
scripts/train_supplymind_grpo.py   minimal TRL/Unsloth training scaffold
```

## Training Path

The intended finale pipeline is:

1. Build and validate the environment.
2. Use the deterministic fallback policies as baselines.
3. Train an LLM policy with TRL/Unsloth against the environment reward.
4. Compare before/after reward curves.
5. Show qualitative episode traces in the UI.

The repo includes `scripts/train_supplymind_grpo.py` as a minimal training scaffold. It should be run in a GPU/Colab/Hugging Face environment with TRL and Unsloth installed.

## Design Notes

Important design choices:

- The central agent does not see raw customer orders.
- Warehouses are local agents, initially simulated but replaceable later.
- Reports are compressed and truthful in v1; noisy or biased reports can be added later.
- Driver loans are left in the schema for future extension, but the current core focuses on inventory coordination.
- Route costs differ by warehouse and region, so warehouses are not equidistant.

## Known Next Steps

- Tune the fallback heuristic for 40-round hard episodes.
- Add more seed families and demand motifs.
- Improve the training script into a Colab-ready notebook.
- Add reward and loss plots after the first real training run.
- Push the environment to Hugging Face Spaces.
