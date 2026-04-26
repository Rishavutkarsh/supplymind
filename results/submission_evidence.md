# SupplyMind Submission Evidence

This compact evidence file mirrors the judge-facing README story: baseline behavior is poor, SFT teaches valid role behavior, and GRPO improves the promoted role policies over their own SFT parents.

## Held-Out Evaluation

| role | policy | global score | role score | raw reward | invalid payloads | invalid actions |
|---|---|---:|---:|---:|---:|---:|
| warehouse | base Qwen | 0.0001 | 0.0001 | -864.40 | 36 | 0 |
| warehouse | SFT parent | 0.2343 | 0.2166 | 26.05 | 0 | 69 |
| warehouse | GRPO child | 0.2801 | 0.2881 | 58.73 | 1 | 58 |
| center | base Qwen | 0.5172 | 0.6336 | 176.12 | 36 | 0 |
| center | SFT parent | 0.5327 | 0.5977 | 186.56 | 0 | 22 |
| center | GRPO child | 0.6469 | 0.7626 | 239.21 | 0 | 0 |

## Submission Evidence

Plot images are intentionally kept out of the Hugging Face Space package. Use this table as the compact judge-facing evidence.

## Read

SFT is doing the format and basic behavior work. GRPO then improves the role-specific held-out score for the promoted center and warehouse stories. The official environment score remains global welfare; role scores are logged for training evidence and diagnostics.
