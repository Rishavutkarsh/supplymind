# SupplyMind Project Spec

## Goal

SupplyMind trains an LLM central orchestrator to operate in a multi-agent warehouse network. Warehouses are stateful local agents with inventory, drivers, demand reports, and local incentives. The orchestrator does not see individual customer orders; it satisfies global demand indirectly by allocating depot stock and approving warehouse-to-warehouse inventory movement.

## Agents

`central_orchestrator`: the evaluated/trainable LLM.

`central_depot`: a finite emergency supply source with limited stock and trucks.

`warehouse_*`: environment agents with:
- inventory by SKU
- available local delivery drivers
- private demand forecasts
- safety stock preferences
- hidden personality and relationship state
- local utility
- local fulfillment policy

## Step Loop

Each round:
1. drivers due back become available
2. warehouses publish compressed demand reports and public offers/requests based on local state
3. central submits depot replenishments, offer matches, transfers, deferrals, and optional coalition notes
4. depot replenishments consume central stock and trucks
5. approved offer matches become inventory transfers that consume bounded transfer capacity
6. warehouse agents accept/reject proposed transfers and loans
7. accepted transfers and loans update inventory/capacity
8. warehouse local agents fulfill local orders using inventory and local drivers
9. late or unserved orders expire
10. holding, waste, fairness, and service rewards are computed
11. the world advances one round

## Official Reward

OpenEnv receives one scalar reward:

```text
global_welfare =
  fulfilled_value
- delivery_cost
- central_replenishment_cost
- transfer_cost
- stockout_penalty
- late_penalty
- holding_cost
- waste_penalty
- fairness_penalty
- invalid_action_penalty
+ coalition_bonus
```

Per-agent rewards are diagnostics in `info` / feedback, not the official score.

## Grading

The score follows the Fleetmind pattern:

```text
score = normalized(agent_reward, reactive_baseline_reward, privileged_reference_reward)
```

The reference planner is a strong privileged anchor, not a mathematical optimum.

## Why This Is Multi-Agent

The LLM is not merely allocating resources. It must model other agents' incentives:
- risk-averse warehouses reject risky transfers
- selfish warehouses require higher compensation
- cooperative warehouses share more readily
- hidden relationship state affects future acceptance
- unfairly draining one warehouse harms local utility and fairness

This makes cooperation, competition, negotiation, and coalition formation visible in the rollout trace.
