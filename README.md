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

SupplyMind V2 exposes short training tiers and longer benchmark tiers:

- `train_easy`: 3 warehouses, 12 rounds
- `train_medium`: 4 warehouses, 18 rounds
- `train_hard`: 5 warehouses, 24 rounds
- `easy`: 3 warehouses, 18 rounds
- `medium`: 4 warehouses, 26 rounds
- `hard`: 5 warehouses, 34 rounds

Legacy names still work as aliases: `cooperative_market -> easy`, `scarcity_market -> medium`, and `crisis_market -> hard`.

Each task is generated deterministically from a curated seed pool when no seed is provided. Passing an explicit seed recreates the same world. Seeds vary motifs such as stable demand, understock, perishable pressure, regional shifts, transfer-needed cases, premium bursts, and tight-SLA pressure.

## Episode Loop

Each round works like this:

1. Depot trucks and local drivers return if their trips are complete.
2. Previously purchased depot inventory arrives if its lead time has elapsed.
3. Warehouses observe their own orders and choose accept/reject decisions.
4. Warehouses publish inventory offers and requests.
5. The central orchestrator buys depot stock, sells stock to warehouses, and proposes or matches transfers.
6. Warehouses accept or reject transfer proposals.
7. Local warehouses fulfill accepted customer orders with local drivers.
8. Rejections, missed accepted orders, silent expiries, stockouts, waste, cost, and fairness effects are scored.
9. Reward components and public feedback are returned. Full audit metrics are kept for evaluators and the UI, not normal black-box play.

## Observation

The joint V2 policy receives structured JSON containing:

- `center`: depot inventory, depot trucks, inbound procurements, warehouse summaries, price bands, and pending proposals
- `warehouses`: per-warehouse inventory, drivers, local orders, route costs, safety stock, and pending proposals
- `feedback`: reward components, recent events, invalid action details, and role rewards
- `scenario_info`: task id, chosen seed, round limits, caps, and compact public rules

Raw customer orders are visible to warehouse agents, but the center only receives summaries in the center observation and role-training endpoint.

## Action Space

The V2 joint action controls both role surfaces:

```json
{
  "warehouse_actions": {
    "north": {
      "order_decisions": [{"order_id": "o1", "decision": "accept"}],
      "inventory_offers": [{"sku": "fresh_milk", "units": 2, "ask_price": 6.0}],
      "inventory_requests": [{"sku": "insulin_pack", "units": 2, "max_price": 12.0}],
      "transfer_responses": [{"proposal_id": "p1", "decision": "accept"}],
      "local_priority": [{"sku": "insulin_pack", "priority": 3}]
    }
  },
  "central_action": {
    "central_procurements": [{"sku": "fresh_milk", "units": 4, "max_unit_cost": 4.0}],
    "central_replenishments": [{"to_warehouse": "north", "sku": "fresh_milk", "units": 2, "unit_price": 6.0}],
    "inventory_transfer_proposals": [{"from_warehouse": "west", "to_warehouse": "north", "sku": "rice_bag_5kg", "units": 2, "compensation": 10.0}],
    "offer_matches": [{"offer_signal_id": "west:offer:rice_bag_5kg", "request_signal_id": "north:request:rice_bag_5kg", "units": 2, "compensation": 10.0}]
  }
}
```

Role-specific training endpoints are also available: `/v2/center/*` freezes warehouses with a strong heuristic, and `/v2/warehouse/*` freezes the center with a strong heuristic.

## Reward

The official reward is one scalar global welfare score per step.

```text
step_reward =
  fulfilled_order_value
+ order_reject_penalties
+ accepted_missed_penalties
+ silent_expiry_penalties
+ center_wholesale_terms
+ warehouse_local_terms
- procurement_cost
- center_shipment_cost
- transfer_cost
- stockout_penalty
- holding_cost
- spoilage_cost
- terminal_leftover_penalty
- terminal_fairness_penalty
- invalid_action_penalty
```

Internal payments are tracked for center and warehouse rewards, but the official global reward avoids double-counting them. The center broker fee is a role reward only; global welfare gets no direct transfer bonus. The code stores signed components, then computes:

```python
step_reward = sum(reward_components.values())
```

The UI shows the component breakdown for each round.

## Grading

Cases are generated procedurally from curated seeds. They are deterministic, but not hand-solved.

The score is normalized against:

- a simple baseline policy
- a negotiation heuristic
- a privileged bounded lookahead planner with extra internal information

The reference is a strong anchor, not a mathematically proven optimum. The planner simulates a capped set of candidate actions over a short horizon and uses hidden environment state only for grading/reference, not for the public agent observation.

```text
progress = (raw_reward - baseline_reward) / (reference_reward - baseline_reward)

if progress <= 1:
  score = 0.05 + progress * (0.95 - 0.05)
else:
  score = 0.95 + min(progress - 1, 1.0) * (0.9999 - 0.95)
```

The baseline maps to about `0.05`. The privileged planner maps to `0.95`. Agents that beat the planner can score above `0.95`.

## Current Evaluation Snapshot

Recent local evaluation over the seeded policy suite:

```text
no_op:                 mean_score 0.0533
reactive_baseline:    mean_score 0.0500
negotiation_heuristic: mean_score 0.2897
privileged_reference:  mean_score 0.9500
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

### SupplyMind V2 Preview

SupplyMind V2 is implemented side-by-side with the stable V1 environment. It turns the warehouse layer into explicit local agents while keeping one official benchmark score.

V2 adds two trainable roles:

- a **central wholesaler/coordinator** that buys stock, holds depot inventory, sells to warehouses, and brokers transfers
- a **shared warehouse policy** copied across every warehouse, where each warehouse accepts/rejects local orders, publishes offers/requests, and accepts/rejects transfer proposals

The business model is closer to a Blinkit/Zepto-style regional supply network:

```text
supplier -> center -> warehouse -> customer
```

V2 exposes:

```text
POST /v2/reset
GET  /v2/state
POST /v2/step
GET  /v2/heuristic-joint-action
GET  /v2/rules
POST /v2/center/reset
GET  /v2/center/state
POST /v2/center/step
POST /v2/warehouse/reset
GET  /v2/warehouse/state
POST /v2/warehouse/step
GET  /v2/ui
```

The V2 UI is served at:

```text
http://127.0.0.1:7860/v2/ui
```

The public rules contract is served at `/v2/rules` and is also linked from `scenario_info.public_rules` in every V2 observation. It includes role definitions, item prices, order penalties, operating costs, transfer costs, and the action schema. It does not reveal future orders, hidden seed recipes, baseline rewards, target rewards, or the reference planner.

V2 tracks three reward views:

- **global welfare**, used for official grading
- **center reward**, used to train/evaluate the central coordinator
- **per-warehouse rewards**, used to train/evaluate the shared warehouse policy

Global welfare deliberately does not reward transfer activity directly. Transfers affect the official score only through later fulfillment, lower stockouts, lower waste, and their real movement cost. The center still receives a small center-only broker fee for successful transfers, so the coordinator has an incentive to broker useful trades without turning transfer count into a global reward hack.

Terminal V2 summaries include audit metrics such as transfer count, successful transfer units, broker fees, stockouts, holding cost, spoilage, terminal leftovers, invalid penalties, and per-warehouse service rates. These audit metrics are not exposed in normal agent-facing state; agents see current-step reward components, events, and invalid-action details. A small terminal fairness penalty discourages serving one region while starving another.

V2 also exposes role-specific training endpoints:

- `/v2/center/*`: the agent controls only the center action while warehouses are frozen to a deterministic local heuristic.
- `/v2/warehouse/*`: the agent controls warehouse actions while the center is frozen to a deterministic procurement/replenishment/matching heuristic.
- `/v2/step`: final joint evaluation where trained center and trained warehouse policies can act together.

This keeps small-model training realistic: the warehouse model learns local order and inventory behavior first, then the center model learns network allocation against a stable warehouse policy.

The V2 grading formula is the same normalized scheme:

```text
progress = (raw_global_reward - baseline_reward) / (target_reward - baseline_reward)

if progress <= 1:
  score = 0.05 + progress * (0.95 - 0.05)
else:
  score = 0.95 + min(progress - 1, 1.0) * (0.9999 - 0.95)
```

The baseline is a naive joint policy. The target is a bounded privileged planner/reference policy, not a claimed mathematical optimum. This keeps the score fair enough for comparison while allowing a trained policy to beat the reference on some seeds.

Current V2 local smoke evaluation:

```text
no_op:                mean_score 0.0001
naive_joint:          mean_score 0.0500
heuristic_joint:      mean_score 0.6264
privileged_reference: mean_score 0.9500
```

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
python scripts/evaluate_v2_policies.py
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
