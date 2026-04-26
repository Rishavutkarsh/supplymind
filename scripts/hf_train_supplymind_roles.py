# /// script
# dependencies = [
#   "torch",
#   "transformers>=4.45.0",
#   "trl>=0.12.0",
#   "peft>=0.13.0",
#   "accelerate",
#   "datasets",
#   "huggingface_hub",
#   "trackio",
#   "pydantic",
#   "pyyaml",
#   "matplotlib",
# ]
# ///
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from datasets import Dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


REPO_ID = "rishavutk/supplymind"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REWARD_SCALE = 10.0
REWARD_CLIP = 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["center", "warehouse", "joint"], default="center")
    parser.add_argument("--task-id", default="train_easy")
    parser.add_argument("--seeds", default="101,113,127")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--hub-model-id", default="")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def prepare_repo() -> Path:
    local_dir = Path(snapshot_download(repo_id=REPO_ID, repo_type="space"))
    sys.path.insert(0, str(local_dir / "src"))
    return local_dir


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)


def extract_json(completion: Any) -> dict[str, Any] | None:
    text = completion_to_text(completion)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def compact_observation(observation: Any, role: str) -> dict[str, Any]:
    data = observation.model_dump(mode="json")
    data["scenario_info"].pop("public_rules", None)
    if role == "center":
        return {
            "role": "center",
            "round_index": data["round_index"],
            "remaining_rounds": data["remaining_rounds"],
            "task_id": data["task_id"],
            "scenario_info": data["scenario_info"],
            "center": data["center"],
            "warehouse_summaries": data["center"].get("warehouse_summaries", []),
            "market_signals": data["center"].get("market_signals", []),
            "feedback": data.get("feedback", {}),
        }
    if role == "warehouse":
        return {
            "role": "warehouse",
            "round_index": data["round_index"],
            "remaining_rounds": data["remaining_rounds"],
            "task_id": data["task_id"],
            "scenario_info": data["scenario_info"],
            "warehouses": data["warehouses"],
            "pending_transfer_proposals": data["center"].get("pending_transfer_proposals", []),
            "feedback": data.get("feedback", {}),
        }
    return data


def system_prompt(role: str) -> str:
    if role == "center":
        return (
            "You are the center policy in SupplyMind. Return only strict JSON matching CenterAction: "
            "central_procurements, central_liquidations, central_replenishments, inventory_transfer_proposals, offer_matches. "
            "Warehouses are controlled by a fixed heuristic. Earn margin and a small share of realized service profit, "
            "but avoid waste, stockouts, overpriced actions, and needless shipments."
        )
    if role == "warehouse":
        return (
            "You are the shared warehouse policy in SupplyMind, copied across all warehouses. "
            "Return only strict JSON with key warehouse_actions mapping warehouse ids to actions. "
            "Use order_decisions, inventory_offers, inventory_requests, transfer_responses, and local_priority. "
            "The center is controlled by a fixed heuristic. Accept orders you can serve, request needed stock, "
            "and reject bad or impossible commitments."
        )
    return (
        "You are playing SupplyMind. Return only strict JSON with top-level keys warehouse_actions and central_action. "
        "Optimize global welfare while avoiding invalid actions, missed accepted orders, stockouts, waste, and pointless transfers."
    )


def build_rows(role: str, task_id: str, seeds: list[int]) -> list[dict[str, Any]]:
    from supplymind_env_v2.environment import V2SupplyMindEnv
    from supplymind_env_v2.policies import heuristic_joint_policy

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        env = V2SupplyMindEnv(default_task_id=task_id)
        observation = env.reset_internal(task_id, seed)
        while not env.done:
            rows.append(
                {
                    "prompt": [
                        {"role": "system", "content": system_prompt(role)},
                        {"role": "user", "content": json.dumps(compact_observation(observation, role), separators=(",", ":"))},
                    ],
                    "task_id": task_id,
                    "seed": seed,
                    "round_index": observation.round_index,
                }
            )
            result = env.step(heuristic_joint_policy(observation), grade_terminal=False)
            observation = result.observation
    return rows


def make_reward_fn(role: str):
    from supplymind_env_v2.environment import V2SupplyMindEnv
    from supplymind_env_v2.models import CenterAction, V2JointAction, V2WarehouseRoleAction
    from supplymind_env_v2.policies import fixed_center_action, fixed_warehouse_actions, heuristic_joint_policy

    def replay_to_round(task_id: str, seed: int, round_index: int):
        env = V2SupplyMindEnv(default_task_id=task_id)
        observation = env.reset_internal(task_id, seed)
        while observation.round_index < round_index and not env.done:
            result = env.step(heuristic_joint_policy(observation), grade_terminal=False)
            observation = result.observation
        return env, observation

    def reward_completions(prompts, completions, task_id, seed, round_index, **kwargs) -> list[float]:
        rewards: list[float] = []
        for completion, current_task, current_seed, current_round in zip(completions, task_id, seed, round_index, strict=True):
            payload = extract_json(completion)
            if payload is None:
                rewards.append(-8.0)
                continue
            env, observation = replay_to_round(current_task, int(current_seed), int(current_round))
            before = dict(env.agent_rewards)
            try:
                if role == "center":
                    center_action = CenterAction.model_validate(payload)
                    action = V2JointAction(warehouse_actions=fixed_warehouse_actions(observation), central_action=center_action)
                elif role == "warehouse":
                    role_action = V2WarehouseRoleAction.model_validate(payload)
                    action = V2JointAction(
                        warehouse_actions=role_action.warehouse_actions,
                        central_action=fixed_center_action(observation, role_action.warehouse_actions),
                    )
                else:
                    action = V2JointAction.model_validate(payload)
            except Exception:
                rewards.append(-8.0)
                continue
            result = env.step(action, grade_terminal=False)
            invalid_count = len(result.observation.feedback.get("invalid_action_details", []))
            if role == "center":
                role_delta = env.agent_rewards["center"] - before.get("center", 0.0)
            elif role == "warehouse":
                warehouse_ids = [key for key in env.agent_rewards if key != "center"]
                role_delta = mean(env.agent_rewards[key] - before.get(key, 0.0) for key in warehouse_ids)
            else:
                role_delta = float(result.reward.step_reward)
            scaled = max(-REWARD_CLIP, min(REWARD_CLIP, role_delta / REWARD_SCALE))
            rewards.append(scaled + 1.0 - 2.0 * invalid_count)
        return rewards

    return reward_completions


def main() -> None:
    args = parse_args()
    prepare_repo()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    output_dir = args.output_dir or f"outputs/supplymind-{args.role}-qwen-grpo"
    hub_model_id = args.hub_model_id or f"rishavutk/supplymind-{args.role}-qwen-0.5b-grpo"

    rows = build_rows(args.role, args.task_id, seeds)
    dataset = Dataset.from_list(rows)
    print(json.dumps({"role": args.role, "rows": len(rows), "task_id": args.task_id, "seeds": seeds}, indent=2), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=make_reward_fn(args.role),
        train_dataset=dataset,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
        args=GRPOConfig(
            output_dir=output_dir,
            max_steps=args.max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            num_generations=2,
            max_prompt_length=2048,
            max_completion_length=512,
            logging_steps=1,
            report_to=["trackio"],
            project="supplymind",
            run_name=f"{args.role}-grpo-smoke",
            push_to_hub=True,
            hub_model_id=hub_model_id,
        ),
    )
    trainer.train()
    trainer.push_to_hub()
    print(json.dumps({"status": "done", "hub_model_id": hub_model_id}, indent=2), flush=True)


if __name__ == "__main__":
    main()
