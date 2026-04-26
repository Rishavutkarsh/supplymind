# SupplyMind Training Evidence

This file is generated from local HF job logs. It is intentionally compact for README/blog reuse.

## Training Runs

| run | steps | first loss | last loss | min loss | invalid payloads | invalid actions |
|---|---:|---:|---:|---:|---:|---:|
| warehouse_sft_v1 | 80 | 0.9343 | 0.0661 | 0.0345 | 0 | 0 |
| warehouse_sft_v2 | 57 | 0.8177 | 0.0763 | 0.0314 | 0 | 0 |
| warehouse_grpo_v2 | 24 | 0.0000 | 0.0000 | 0.0000 | 1 | 2 |
| center_sft_v1 | 97 | 1.3980 | 0.0957 | 0.0360 | 0 | 0 |

## Held-out Evaluation

| role | label | global score | role score | raw reward | invalid payloads | invalid actions | action totals |
|---|---|---:|---:|---:|---:|---:|---|
| warehouse | grpo | 0.2801 | 0.2881 | 58.73 | 1 | 58 | inventory_offers:44, inventory_requests:158, local_priority:214, order_decisions:131, warehouses_controlled:107 |
| warehouse | base | 0.0001 | 0.0001 | -864.40 | 36 | 0 | order_decisions:45, transfer_responses:4, warehouses_controlled:24 |
| warehouse | sft | 0.2343 | 0.2166 | 26.05 | 0 | 69 | inventory_offers:161, inventory_requests:105, local_priority:216, order_decisions:142, warehouses_controlled:108 |
| center | base | 0.5172 | 0.6336 | 176.12 | 36 | 0 | - |
| center | sft | 0.5327 | 0.5977 | 186.56 | 0 | 22 | central_replenishments:29 |
| center | grpo | 0.6469 | 0.7626 | 239.21 | 0 | 0 | central_replenishments:4 |

## Current Read
- Best warehouse role score so far: `grpo` at `0.2881`.
- Best center role score so far: `grpo` at `0.7626`.
- Promotion rule: prefer the checkpoint that improves held-out role score without increasing invalid payloads/actions.

## Plots

- `results/plots/submission_training_loss.png`
- `results/plots/submission_warehouse_grpo_reward.png`
- `results/plots/submission_warehouse_scores.png`
- `results/plots/submission_warehouse_invalids.png`
- `results/plots/submission_center_scores.png`
- `results/plots/submission_center_invalids.png`
