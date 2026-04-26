---
title: SupplyMind OpenEnv
sdk: docker
app_port: 7860
---

# SupplyMind

SupplyMind is an OpenEnv environment for training LLM agents to coordinate a regional supply network. A central coordinator allocates depot stock, buys replenishment, and brokers transfers; local warehouses accept customer orders, manage inventory, and expose offers or requests. The task is multi-agent because local and central incentives are related but not identical, and the center acts under partial observability.

HF Space: [rishavutk/supplymind](https://huggingface.co/spaces/rishavutk/supplymind)

## Why It Matters

Real supply-chain operations are not single-turn question answering. They require repeated decisions under scarcity:

- demand arrives over time
- inventory spoils or sits idle
- deliveries and transfers cost money
- local warehouses know more about local orders than the center
- helping one warehouse can hurt another later

SupplyMind turns this into a verifiable RL environment with structured observations, JSON actions, and programmatic rewards.

## Theme Fit

Primary hackathon theme: **Theme #1 - Multi-Agent Interactions**.

The environment includes cooperation, competition, negotiation, partial observability, and coalition-like inventory sharing. The center can improve global outcomes only by reasoning about warehouse needs, costs, and incentives.

## Environment

The world models a quick-commerce supply network:

```text
supplier -> central depot -> regional warehouses -> customers
```

Tracked SKUs:

- `fresh_milk`
- `rice_bag_5kg`
- `insulin_pack`
- `usb_c_charger`

Public tasks:

| Task | Warehouses | Rounds |
|---|---:|---:|
| `easy` | 3 | 18 |
| `medium` | 4 | 26 |
| `hard` | 5 | 34 |

Training aliases are also available as `v2_train_easy`, `v2_train_medium`, and `v2_train_hard`.

## Agent Interfaces

The joint action has two role surfaces:

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
    "central_liquidations": [{"sku": "fresh_milk", "units": 2}],
    "central_replenishments": [{"to_warehouse": "north", "sku": "insulin_pack", "units": 2, "unit_price": 12.0}],
    "inventory_transfer_proposals": [{"from_warehouse": "west", "to_warehouse": "north", "sku": "rice_bag_5kg", "units": 2, "compensation": 10.0}],
    "offer_matches": [{"offer_signal_id": "west:offer:rice_bag_5kg", "request_signal_id": "north:request:rice_bag_5kg", "units": 2, "compensation": 10.0}]
  }
}
```

The repo also exposes role-specific training endpoints:

- `/v2/center/*`: train the center while warehouses are frozen to a heuristic
- `/v2/warehouse/*`: train warehouse behavior while the center is frozen
- `/v2/step`: evaluate both roles together in the same world

## Reward And Grading

The official score uses **global welfare**:

```text
global_welfare =
  fulfilled_customer_value
- procurement_cost
- center_shipment_cost
- transfer_cost
- warehouse_delivery_cost
- holding_cost
- spoilage_cost
- stockout_penalty
- terminal_leftover_penalty
- fairness_penalty
- invalid_action_penalty
```

Role-specific rewards are also tracked:

- center reward: wholesale margin, useful service share, broker fees, depot costs, stockout share
- warehouse reward: customer fulfillment revenue, local costs, missed-order penalties, transfer economics

These role rewards are used for role training evidence. The official benchmark score remains global welfare.

Final score is normalized against a naive baseline and a privileged bounded planner:

```text
progress = (raw_reward - baseline_reward) / (target_reward - baseline_reward)

if progress <= 1:
  score = 0.05 + progress * 0.90
else:
  score = 0.95 + min(progress - 1, 1.0) * 0.0499
```

The target planner is a strong reference policy, not a claimed mathematical optimum.

## Training Evidence

Our final story is:

```text
Base Qwen -> SFT warm start -> GRPO improvement
```

Held-out role-eval results on seeds `131, 149, 163`:

| Role | Variant | Global score | Role score | Raw reward | Invalid payloads | Invalid actions |
|---|---|---:|---:|---:|---:|---:|
| warehouse | Base Qwen 0.5B | 0.0001 | 0.0001 | -864.40 | 36 | 0 |
| warehouse | SFT parent | 0.2343 | 0.2166 | 26.05 | 0 | 69 |
| warehouse | GRPO child | 0.2801 | 0.2881 | 58.73 | 1 | 58 |
| center | Base Qwen 0.5B | 0.5172 | 0.6336 | 176.12 | 36 | 0 |
| center | SFT parent | 0.5327 | 0.5977 | 186.56 | 0 | 22 |
| center | GRPO child | 0.6469 | 0.7626 | 239.21 | 0 | 0 |

Curated evidence lives in [results/submission/summary.md](results/submission/summary.md).

Key plots:

- [Baseline/SFT/GRPO scores](results/submission/baseline_sft_grpo_scores.png)
- [Invalid outputs](results/submission/invalids.png)
- [GRPO reward curve](results/submission/grpo_reward_curve.png)
- [Joint SFT benchmark](results/submission/joint_sft_benchmark.png)

## Run Locally

Install:

```bash
pip install -r requirements.txt
```

Start the Space/API:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Open:

```text
http://127.0.0.1:7860/
```

Useful endpoints:

```text
POST /reset
GET  /state
POST /step
GET  /v2/rules
GET  /v2/ui
GET  /v2/dashboard
```

## Reproduce

Environment validation:

```bash
python validate_submission.py
```

Policy baselines:

```bash
python scripts/evaluate_v2_policies.py
```

Role training and evaluation scripts:

```bash
python scripts/hf_sft_supplymind_roles.py --help
python scripts/hf_train_supplymind_roles.py --help
python scripts/hf_eval_supplymind_adapters.py --help
```

Colab/HF training scaffold:

```text
colab/supplymind_v2_grpo_colab.py
```

## Project Structure

```text
src/supplymind_env_v2/        environment, models, rewards, generator, planner
src/supplymind_env/api.py     FastAPI app mounting V2 routes
static/v2.html                interactive episode UI
inference.py                  deterministic benchmark inference path
scripts/                      training, evaluation, plotting, preflight scripts
configs/                      documented reward configuration
results/submission/           curated judge-facing evidence
```

## Notes

We also experimented with same-rollout multi-agent adapter updates. Those scripts are kept as experimental scaffolding, but the final submission evidence focuses on the stable and reproducible role-training path: SFT for action format, then GRPO for reward improvement.
