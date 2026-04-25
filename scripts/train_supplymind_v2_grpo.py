from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from supplymind_env_v2.environment import V2SupplyMindEnv
from supplymind_env_v2.generator import PROFILES
from supplymind_env_v2.models import V2JointAction
from supplymind_env_v2.policies import naive_joint_policy


SYSTEM_PROMPT = """You are playing SupplyMind V2, a multi-agent supply network.
Return strict JSON with warehouse_actions and central_action.
Warehouses decide local order acceptance, offers, requests, transfer responses, and priorities.
The center decides procurement, wholesale replenishment, transfer proposals, and offer matches.
Optimize global welfare while improving center reward and average warehouse reward."""


def build_training_rows(limit_per_task: int = 2) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id in PROFILES:
        for seed in range(101, 101 + limit_per_task * 2, 2):
            env = V2SupplyMindEnv(default_task_id=task_id)
            observation = env.reset_internal(task_id, seed)
            while not env.done:
                rows.append(
                    {
                        "prompt": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": json.dumps(observation.model_dump(mode="json"), separators=(",", ":"))},
                        ],
                        "task_id": task_id,
                        "seed": seed,
                        "round_index": observation.round_index,
                    }
                )
                result = env.step(naive_joint_policy(observation), grade_terminal=False)
                observation = result.observation
    return rows


def extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def reward_completions(prompts: list[Any], completions: list[str], task_id: list[str], seed: list[int], round_index: list[int], **_: Any) -> list[float]:
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
        observation = env.reset_internal(current_task, current_seed)
        while observation.round_index < current_round and not env.done:
            result = env.step(naive_joint_policy(observation), grade_terminal=False)
            observation = result.observation
        result = env.step(action, grade_terminal=False)
        rewards.append(float(result.reward.step_reward))
    return rewards


def main() -> None:
    try:
        from datasets import Dataset
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit("Install unsloth, trl, datasets, transformers, accelerate in a GPU environment first.") from exc

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=model_name, max_seq_length=4096, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = Dataset.from_list(build_training_rows())
    config = GRPOConfig(
        output_dir=str(ROOT / "outputs" / "supplymind-v2-grpo"),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=2,
        max_prompt_length=2048,
        max_completion_length=512,
        logging_steps=1,
        report_to="none",
    )
    trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=reward_completions, args=config, train_dataset=dataset)
    trainer.train()
    trainer.save_model(str(ROOT / "outputs" / "supplymind-v2-grpo-final"))


if __name__ == "__main__":
    main()
