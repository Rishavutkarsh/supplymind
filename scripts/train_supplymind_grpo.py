from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from supplymind_env.environment import V3SupplyMindEnv
from supplymind_env.models import V3Action
from supplymind_env.policies import baseline_policy
from supplymind_env.seed_catalog import TRAIN_SEEDS, TASK_IDS


SYSTEM_PROMPT = """You are the central orchestrator for SupplyMind.
Return JSON only:
{"central_replenishments":[],"inventory_transfers":[],"offer_matches":[],"priority_policy":[],"defer_orders":[],"coalition_deals":[]}
Warehouses publish local offers and requests as market_signals, but hidden incentives must be inferred from public behavior.
You do not see individual customer orders; local warehouse agents handle local fulfillment. Use central_replenishments for limited depot-to-warehouse restock, offer_matches for compatible stock trades, and inventory_transfers for direct stock sharing with compensation."""


def build_training_rows(limit_per_task: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        for seed in TRAIN_SEEDS[task_id][:limit_per_task]:
            env = V3SupplyMindEnv(default_task_id=task_id)
            observation = env.reset_internal(task_id=task_id, internal_seed=seed, public_seed=seed)
            while not env.done:
                rows.append(
                    {
                        "prompt": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": json.dumps(observation.model_dump(mode="json"), separators=(",", ":")),
                            },
                        ],
                        "task_id": task_id,
                        "seed": seed,
                        "round_index": observation.round_index,
                    }
                )
                result = env.step(V3Action())
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
    rewards: list[float] = []
    for completion, current_task, current_seed, current_round in zip(completions, task_id, seed, round_index, strict=True):
        payload = extract_json(completion)
        if payload is None:
            rewards.append(-8.0)
            continue
        try:
            action = V3Action.model_validate(payload)
        except Exception:
            rewards.append(-8.0)
            continue

        env = V3SupplyMindEnv(default_task_id=current_task)
        observation = env.reset_internal(task_id=current_task, internal_seed=current_seed, public_seed=current_seed)
        while observation.round_index < current_round and not env.done:
            result = env.step(baseline_policy(observation), grade_terminal=False)
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
        raise SystemExit(
            "Install finale training deps in Colab first, for example: "
            "pip install unsloth trl datasets transformers accelerate"
        ) from exc

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = Dataset.from_list(build_training_rows())

    config = GRPOConfig(
        output_dir=str(ROOT / "outputs" / "supplymind-grpo"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_generations=2,
        max_prompt_length=2048,
        max_completion_length=256,
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
    trainer.save_model(str(ROOT / "outputs" / "supplymind-grpo-final"))


if __name__ == "__main__":
    main()
