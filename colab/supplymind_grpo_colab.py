# %% [markdown]
# # SupplyMind GRPO Colab Smoke
#
# Run this first with `Qwen/Qwen2.5-0.5B-Instruct`. The goal is a short proof that
# model outputs become more valid and rewards improve on the environment loop.

# %% [markdown]
# ## 1. Install And Clone
#
# In Colab, run this cell once. Restart the runtime if Unsloth asks for it.

# %%
# !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" trl datasets transformers accelerate peft pydantic fastapi pyyaml requests matplotlib
# !git clone https://github.com/Rishavutkarsh/supplymind.git
# %cd supplymind

# %%
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from supplymind_env_v2.environment import V2SupplyMindEnv
from supplymind_env_v2.models import V2JointAction
from supplymind_env_v2.policies import heuristic_joint_policy, no_op_policy


# %% [markdown]
# ## 2. Config

# %%
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "outputs/supplymind-qwen-grpo-smoke"

TASK_IDS = ["v2_train_easy"]
SEEDS = [101, 113, 127]
MAX_STEPS = 20
REWARD_SCALE = 10.0
REWARD_CLIP = 20.0
PUSH_TO_HUB = False
HUB_MODEL_ID = "rishavutk/supplymind-qwen-0.5b-grpo"

SYSTEM_PROMPT = """You are playing SupplyMind, a multi-agent supply network environment.
Return only strict JSON with top-level keys warehouse_actions and central_action.
Use public forecasts, visible orders, inventory, drivers, truck availability, and route costs.
Accept orders only when they can be fulfilled or supported by immediate replenishment.
Avoid invalid actions, accepted-order expiry, pointless procurement, and excess spoilage."""


def log(message: str, **fields: Any) -> None:
    print(json.dumps({"message": message, **fields}, sort_keys=True), flush=True)


# %% [markdown]
# ## 3. Dataset Builder

# %%
def compact_observation(observation) -> dict[str, Any]:
    data = observation.model_dump(mode="json")
    data["scenario_info"].pop("public_rules", None)
    return data


def build_rows() -> list[dict[str, Any]]:
    log("building_dataset", task_ids=TASK_IDS, seeds=SEEDS)
    rows: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        for seed in SEEDS:
            env = V2SupplyMindEnv(default_task_id=task_id)
            observation = env.reset_internal(task_id, seed)
            while not env.done:
                rows.append(
                    {
                        "prompt": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": json.dumps(compact_observation(observation), separators=(",", ":"))},
                        ],
                        "task_id": task_id,
                        "seed": seed,
                        "round_index": observation.round_index,
                    }
                )
                result = env.step(heuristic_joint_policy(observation), grade_terminal=False)
                observation = result.observation
    log("dataset_built", rows=len(rows))
    return rows


def extract_json(completion: Any) -> dict[str, Any] | None:
    if isinstance(completion, list):
        completion = completion[-1].get("content", "") if completion else ""
    if not isinstance(completion, str):
        completion = str(completion)
    match = re.search(r"\{.*\}", completion, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def replay_to_round(task_id: str, seed: int, round_index: int) -> tuple[V2SupplyMindEnv, Any]:
    env = V2SupplyMindEnv(default_task_id=task_id)
    observation = env.reset_internal(task_id, seed)
    while observation.round_index < round_index and not env.done:
        result = env.step(heuristic_joint_policy(observation), grade_terminal=False)
        observation = result.observation
    return env, observation


def run_policy(task_id: str, seed: int, policy_name: str) -> dict[str, Any]:
    env = V2SupplyMindEnv(default_task_id=task_id)
    observation = env.reset_internal(task_id, seed)
    while not env.done:
        if policy_name == "no_op":
            action = no_op_policy(observation)
        else:
            action = heuristic_joint_policy(observation)
        result = env.step(action)
        observation = result.observation
    return dict(env.last_episode_summary or {})


# %% [markdown]
# ## 4. Reward Function

# %%
reward_trace: list[float] = []
valid_json_trace: list[float] = []
reward_call_count = 0


def reward_completions(prompts, completions, task_id, seed, round_index, **kwargs) -> list[float]:
    global reward_call_count
    reward_call_count += 1
    rewards: list[float] = []
    invalid_payloads = 0
    invalid_actions = 0
    for completion, current_task, current_seed, current_round in zip(completions, task_id, seed, round_index, strict=True):
        payload = extract_json(completion)
        if payload is None:
            rewards.append(-8.0)
            valid_json_trace.append(0.0)
            invalid_payloads += 1
            continue
        try:
            action = V2JointAction.model_validate(payload)
        except Exception:
            rewards.append(-8.0)
            valid_json_trace.append(0.0)
            invalid_payloads += 1
            continue

        env, _ = replay_to_round(current_task, int(current_seed), int(current_round))
        result = env.step(action, grade_terminal=False)
        invalid_count = len(result.observation.feedback.get("invalid_action_details", []))
        invalid_actions += invalid_count
        scaled_step = float(result.reward.step_reward) / REWARD_SCALE
        shaped_reward = max(-REWARD_CLIP, min(REWARD_CLIP, scaled_step))
        shaped_reward += 1.0
        shaped_reward -= 2.0 * invalid_count
        rewards.append(shaped_reward)
        valid_json_trace.append(1.0)

    reward_trace.extend(rewards)
    if reward_call_count == 1 or reward_call_count % 10 == 0:
        log(
            "reward_batch",
            call=reward_call_count,
            count=len(rewards),
            mean_reward=round(mean(rewards), 4) if rewards else None,
            invalid_payloads=invalid_payloads,
            invalid_actions=invalid_actions,
        )
    return rewards


# %% [markdown]
# ## 5. Baseline Sanity Check

# %%
from datasets import Dataset

rows = build_rows()
dataset = Dataset.from_list(rows)
log("dataset_ready", rows=len(rows), tasks=TASK_IDS, seeds=SEEDS, max_steps=MAX_STEPS)

env = V2SupplyMindEnv(default_task_id="v2_train_easy")
for policy_name in ("no_op", "heuristic"):
    summaries = [run_policy("v2_train_easy", seed, policy_name) for seed in SEEDS]
    log(
        "baseline_probe",
        policy=policy_name,
        mean_score=round(mean(float(row["graded_score"]) for row in summaries), 4),
        mean_center_role_score=round(mean(float(row["center_role_score"]) for row in summaries), 4),
        mean_warehouse_role_score=round(mean(float(row["warehouse_role_score"]) for row in summaries), 4),
        mean_reward=round(mean(float(row["raw_reward"]) for row in summaries), 3),
        mean_center_reward=round(mean(float(row["center_reward"]) for row in summaries), 3),
        mean_average_warehouse_reward=round(mean(float(row["average_warehouse_reward"]) for row in summaries), 3),
    )


# %% [markdown]
# ## 6. Train

# %%
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

started = time.time()
log("loading_model", model=MODEL_NAME)
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
log("model_loaded", model=MODEL_NAME)

config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    max_steps=MAX_STEPS,
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
log("training_start", max_steps=MAX_STEPS)
train_result = trainer.train()
log("training_done", elapsed_seconds=round(time.time() - started, 2))
trainer.save_model(f"{OUTPUT_DIR}-final")

if PUSH_TO_HUB:
    log("pushing_model", hub_model_id=HUB_MODEL_ID)
    trainer.push_to_hub(HUB_MODEL_ID)


# %% [markdown]
# ## 7. Plot Training Signals

# %%
import matplotlib.pyplot as plt

results_dir = ROOT / "results"
results_dir.mkdir(exist_ok=True)

if reward_trace:
    plt.figure(figsize=(7, 4))
    plt.plot(reward_trace)
    plt.xlabel("reward call")
    plt.ylabel("environment reward")
    plt.title("SupplyMind GRPO reward trace")
    plt.tight_layout()
    plt.savefig(results_dir / "colab_grpo_reward_trace.png", dpi=160)
    plt.show()

if valid_json_trace:
    window = 10
    rolling = [mean(valid_json_trace[max(0, i - window + 1): i + 1]) for i in range(len(valid_json_trace))]
    plt.figure(figsize=(7, 4))
    plt.plot(rolling)
    plt.xlabel("reward call")
    plt.ylabel("valid JSON rate")
    plt.title("SupplyMind valid action formatting")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(results_dir / "colab_grpo_valid_json_rate.png", dpi=160)
    plt.show()

history = {
    "train_result": getattr(train_result, "metrics", {}),
    "trainer_log_history": trainer.state.log_history,
    "reward_trace": reward_trace,
    "valid_json_trace": valid_json_trace,
    "config": {
        "model_name": MODEL_NAME,
        "task_ids": TASK_IDS,
        "seeds": SEEDS,
        "max_steps": MAX_STEPS,
        "reward_scale": REWARD_SCALE,
        "reward_clip": REWARD_CLIP,
    },
}
(results_dir / "colab_grpo_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
log("saved_training_artifacts", results_dir=str(results_dir))
