# %% [markdown]
# # SupplyMind GRPO Colab Smoke
#
# This is a Colab-friendly script/notebook scaffold. Start with the smallest model and a short run.

# %%
# In Colab, run:
# !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" trl datasets transformers accelerate peft pydantic fastapi pyyaml requests
# Prefer the neutral entrypoint:
# %run colab/supplymind_grpo_colab.py

# %%
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from supplymind_env_v2.environment import V2SupplyMindEnv
from supplymind_env_v2.models import V2JointAction
from supplymind_env_v2.policies import heuristic_joint_policy, no_op_policy


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "outputs/supplymind-v2-qwen-grpo-smoke"
TASK_IDS = ["train_easy"]
SEEDS = [101, 113]

SYSTEM_PROMPT = """You are playing SupplyMind V2.
Return only strict JSON with top-level keys warehouse_actions and central_action.
Use the public state, public forecasts, inventory, drivers, and route costs.
Optimize global welfare. Avoid invalid actions and accepted-order expiry."""


def extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def build_rows() -> list[dict[str, Any]]:
    rows = []
    for task_id in TASK_IDS:
        for seed in SEEDS:
            env = V2SupplyMindEnv(default_task_id=task_id)
            observation = env.reset_internal(task_id, seed)
            while not env.done:
                state = observation.model_dump(mode="json")
                rows.append(
                    {
                        "prompt": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": json.dumps(state, separators=(",", ":"))},
                        ],
                        "task_id": task_id,
                        "seed": seed,
                        "round_index": observation.round_index,
                    }
                )
                result = env.step(heuristic_joint_policy(observation), grade_terminal=False)
                observation = result.observation
    return rows


def reward_completions(prompts, completions, task_id, seed, round_index, **kwargs) -> list[float]:
    rewards = []
    for completion, current_task, current_seed, current_round in zip(completions, task_id, seed, round_index, strict=True):
        payload = extract_json(completion)
        if payload is None:
            rewards.append(-8.0)
            continue
        try:
            action = V2JointAction.model_validate(payload)
        except Exception:
            rewards.append(-8.0)
            continue

        env = V2SupplyMindEnv(default_task_id=current_task)
        observation = env.reset_internal(current_task, int(current_seed))
        while observation.round_index < int(current_round) and not env.done:
            result = env.step(heuristic_joint_policy(observation), grade_terminal=False)
            observation = result.observation
        result = env.step(action, grade_terminal=False)
        format_bonus = 1.0 if payload.keys() >= {"warehouse_actions", "central_action"} else 0.0
        rewards.append(float(result.reward.step_reward) + format_bonus)
    return rewards


# %%
# Smoke dataset
from datasets import Dataset

dataset = Dataset.from_list(build_rows())
print(dataset)

# %%
# Train with Unsloth + TRL GRPO
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=4096,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    max_steps=20,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_generations=2,
    max_prompt_length=2048,
    max_completion_length=512,
    logging_steps=1,
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_completions,
    args=config,
    train_dataset=dataset,
)
trainer.train()
trainer.save_model(f"{OUTPUT_DIR}-final")
