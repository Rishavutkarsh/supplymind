# SupplyMind V2 Submission Pipeline

This is the fast path for turning the current V2 environment into a hackathon submission without blocking iteration.

## 1. Local Preflight

Run:

```powershell
python scripts/submission_preflight_v2.py
```

This checks:

- required submission files exist
- `/health`, `/v2/rules`, `/v2/reset`, `/v2/state`, `/v2/step`
- JSON-body reset for `easy`, `medium`, `hard`
- role endpoints for center and warehouse training
- one no-op episode and one heuristic episode
- V2 policy evaluation summary if `results/v2_policy_eval.json` exists

It writes:

```text
results/submission_preflight_v2.json
```

## 2. Training Smoke

Use the Colab scaffold:

```text
colab/supplymind_v2_grpo_colab.py
```

Recommended first run:

```text
model: Qwen/Qwen2.5-0.5B-Instruct
task: train_easy
role: warehouse or joint
episodes: small smoke only
```

The goal is not to train a great model on the first run. The goal is to show:

- the trainer can call the environment
- rewards are logged
- outputs become valid JSON more often
- before/after replay can be shown in the UI

## 3. Hugging Face Space

The Space should run:

```text
uvicorn app:app --host 0.0.0.0 --port 7860
```

Important public links to include in the final README:

- Space URL
- Colab notebook URL
- reward/loss plot
- short demo/video/blog

## 4. Current Training Recommendation

Because Qwen-small is the target, train in stages:

1. **Format/SFT or very small GRPO smoke** on `train_easy`
2. **Warehouse role** with fixed center
3. **Center role** with fixed warehouses
4. Optional final **joint** policy run

For the hackathon, a clean before/after on `train_easy` and `train_medium` is enough if the story is clear.

## 5. Do Not Overfit The Environment Now

Freeze the core environment once preflight and a training smoke pass. Further world-hardening should be small and measured against:

```text
naive ~= 0.05
heuristic ~= 0.60-0.75
reference ~= 0.95
```

