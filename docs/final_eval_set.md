# SupplyMind Final Eval Set

Use this small held-out set for final model comparisons:

| tier | task_id | seed |
|---|---|---:|
| easy | `v2_train_easy` | `131` |
| medium | `v2_train_medium` | `211` |
| hard | `v2_train_hard` | `307` |

Compare the same cases across:

1. Base center + base warehouse
2. SFT center + SFT warehouse
3. GRPO center + best warehouse
4. Best center + best warehouse

Primary plots:

- global score by setup
- center role score by setup
- warehouse role score by setup
- invalid payload/action counts by setup

This set is intentionally small so it can be rerun quickly during the hackathon. Do not use these seeds for further training.
