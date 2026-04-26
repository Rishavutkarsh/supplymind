# SupplyMind Submission Evidence

Final judge-facing training story:

```text
Base Qwen -> SFT warm start -> GRPO improvement
```

Held-out role-eval seeds: `131, 149, 163`.

| Role | Variant | Global score | Role score | Raw reward | Invalid payloads | Invalid actions |
|---|---|---:|---:|---:|---:|---:|
| warehouse | Base Qwen 0.5B | 0.0001 | 0.0001 | -864.40 | 36 | 0 |
| warehouse | SFT parent | 0.2343 | 0.2166 | 26.05 | 0 | 69 |
| warehouse | GRPO child | 0.2801 | 0.2881 | 58.73 | 1 | 58 |
| center | Base Qwen 0.5B | 0.5172 | 0.6336 | 176.12 | 36 | 0 |
| center | SFT parent | 0.5327 | 0.5977 | 186.56 | 0 | 22 |
| center | GRPO child | 0.6469 | 0.7626 | 239.21 | 0 | 0 |

Interpretation:

- SFT makes the model produce usable role actions.
- GRPO improves held-out role scores over the SFT parent.
- Global welfare is logged for audit, while role reward is used for role-specific training.
- Joint-world play is supported through the same environment, but the headline evidence uses stable role training.

Plot images are intentionally kept out of the Hugging Face Space package. The table above is the compact judge-facing evidence.
