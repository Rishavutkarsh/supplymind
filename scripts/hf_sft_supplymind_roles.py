# /// script
# dependencies = [
#   "torch",
#   "transformers>=4.45.0",
#   "peft>=0.13.0",
#   "accelerate",
#   "datasets",
#   "huggingface_hub",
#   "pydantic",
#   "pyyaml",
# ]
# ///
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from datasets import Dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments


REPO_ID = "rishavutk/supplymind"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def log(message: str, **fields: Any) -> None:
    print(json.dumps({"message": message, **fields}, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["center", "warehouse"], required=True)
    parser.add_argument("--task-id", default="v2_train_easy")
    parser.add_argument("--seeds", default="101,113,127")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--center-non-empty-weight", type=int, default=3)
    parser.add_argument("--hub-model-id", default="")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def prepare_repo() -> Path:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    log("downloading_supplymind_space", repo_id=REPO_ID)
    local_dir = Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="space",
            allow_patterns=["src/**", "configs/**"],
            local_dir=Path("supplymind_snapshot").resolve(),
        )
    )
    config_path = local_dir / "configs" / "supplymind_v2_rewards.yaml"
    os.environ["SUPPLYMIND_REWARD_CONFIG"] = str(config_path)
    sys.path.insert(0, str(local_dir / "src"))
    log("repo_ready", path=str(local_dir), config_exists=config_path.exists())
    return local_dir


def compact_observation(observation: Any, role: str, warehouse_id: str | None = None) -> dict[str, Any]:
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
    if warehouse_id:
        warehouse = data["warehouses"][warehouse_id]
        return {
            "role": "warehouse",
            "warehouse_id": warehouse_id,
            "round_index": data["round_index"],
            "remaining_rounds": data["remaining_rounds"],
            "task_id": data["task_id"],
            "scenario_info": data["scenario_info"],
            "warehouse": warehouse,
            "pending_transfer_proposals": warehouse.get("pending_transfer_proposals", []),
            "feedback": data.get("feedback", {}),
        }
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


def system_prompt(role: str) -> str:
    if role == "center":
        return (
            "You are the center policy in SupplyMind. Return only strict JSON matching CenterAction: "
            "central_procurements, central_liquidations, central_replenishments, inventory_transfer_proposals, offer_matches. "
            "Warehouses are controlled by a fixed heuristic. Earn margin and a small share of realized service profit, "
            "but avoid waste, stockouts, overpriced actions, and needless shipments. Empty lists are only appropriate "
            "when no useful procurement, liquidation, replenishment, transfer proposal, or offer match exists."
        )
    return (
        "You are the shared warehouse policy in SupplyMind, copied across all warehouses. "
        "You control exactly one warehouse from the user observation. Return only strict JSON matching WarehouseAction: "
        "order_decisions, inventory_offers, inventory_requests, transfer_responses, and local_priority. "
        "The center is controlled by a fixed heuristic. Accept orders you can serve, request needed stock, "
        "and reject bad or impossible commitments. Only use order_id and proposal_id values visible in this observation."
    )


def action_completion(role: str, joint_action: Any) -> str:
    if role == "center":
        payload = joint_action.central_action.model_dump(mode="json")
    else:
        raise ValueError("warehouse completion needs action_completion_for_warehouse")
    return json.dumps(payload, separators=(",", ":"))


def action_completion_for_warehouse(joint_action: Any, warehouse_id: str) -> str:
    payload = joint_action.warehouse_actions[warehouse_id].model_dump(mode="json")
    return json.dumps(payload, separators=(",", ":"))


def center_action_is_empty(center_action: Any) -> bool:
    data = center_action.model_dump(mode="json")
    return not any(data.get(key) for key in data)


def _chat_text(tokenizer: Any, system: str, user: dict[str, Any], assistant: str | None) -> tuple[str, str]:
    prompt_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, separators=(",", ":"))},
    ]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    if assistant is None:
        return prompt_text, prompt_text
    full_messages = [*prompt_messages, {"role": "assistant", "content": assistant}]
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    return prompt_text, full_text


def build_rows(role: str, task_ids: list[str], seeds: list[int], tokenizer: Any, center_non_empty_weight: int) -> list[dict[str, str]]:
    from supplymind_env_v2.environment import V2SupplyMindEnv
    from supplymind_env_v2.policies import heuristic_joint_policy

    rows: list[dict[str, str]] = []
    center_empty = 0
    center_non_empty = 0
    for task_id in task_ids:
        for seed in seeds:
            env = V2SupplyMindEnv(default_task_id=task_id)
            observation = env.reset_internal(task_id, seed)
            while not env.done:
                joint_action = heuristic_joint_policy(observation)
                if role == "center":
                    user = compact_observation(observation, role)
                    completion = action_completion(role, joint_action)
                    prompt_text, text = _chat_text(tokenizer, system_prompt(role), user, completion)
                    row = {"text": text, "prompt_text": prompt_text, "task_id": task_id, "seed": seed, "round_index": observation.round_index}
                    is_empty = center_action_is_empty(joint_action.central_action)
                    center_empty += int(is_empty)
                    center_non_empty += int(not is_empty)
                    weight = 1 if is_empty else max(1, center_non_empty_weight)
                    rows.extend([row] * weight)
                else:
                    for warehouse_id in observation.warehouses:
                        user = compact_observation(observation, role, warehouse_id)
                        completion = action_completion_for_warehouse(joint_action, warehouse_id)
                        prompt_text, text = _chat_text(tokenizer, system_prompt(role), user, completion)
                        rows.append(
                            {
                                "text": text,
                                "prompt_text": prompt_text,
                                "task_id": task_id,
                                "seed": seed,
                                "round_index": observation.round_index,
                                "warehouse_id": warehouse_id,
                            }
                        )
                result = env.step(joint_action, grade_terminal=False)
                observation = result.observation
    if role == "center":
        log(
            "center_sft_mix",
            task_ids=task_ids,
            seeds=seeds,
            empty_teacher_steps=center_empty,
            non_empty_teacher_steps=center_non_empty,
            non_empty_weight=center_non_empty_weight,
            total_rows=len(rows),
        )
    return rows


def baseline_probe(role: str, task_id: str, seeds: list[int]) -> None:
    from supplymind_env_v2.environment import V2SupplyMindEnv
    from supplymind_env_v2.policies import heuristic_joint_policy, no_op_policy

    for policy_name, policy_fn in (("no_op", no_op_policy), ("heuristic", heuristic_joint_policy)):
        summaries: list[dict[str, Any]] = []
        for seed in seeds:
            env = V2SupplyMindEnv(default_task_id=task_id)
            observation = env.reset_internal(task_id, seed)
            while not env.done:
                result = env.step(policy_fn(observation))
                observation = result.observation
            summaries.append(env.last_episode_summary or {})
        log(
            "baseline_probe",
            role=role,
            policy=policy_name,
            mean_global_score=round(mean(float(row.get("graded_score", 0.0)) for row in summaries), 4),
            mean_center_role_score=round(mean(float(row.get("center_role_score", 0.0)) for row in summaries), 4),
            mean_warehouse_role_score=round(mean(float(row.get("warehouse_role_score", 0.0)) for row in summaries), 4),
            mean_raw_reward=round(mean(float(row.get("raw_reward", 0.0)) for row in summaries), 3),
        )


def tokenize_dataset(dataset: Dataset, tokenizer: Any, max_length: int) -> Dataset:
    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        encoded = tokenizer(batch["text"], truncation=True, max_length=max_length)
        labels = []
        for input_ids, prompt_text in zip(encoded["input_ids"], batch["prompt_text"], strict=True):
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            prompt_len = min(len(prompt_ids), len(input_ids))
            labels.append([-100] * prompt_len + input_ids[prompt_len:])
        encoded["labels"] = labels
        return encoded

    return dataset.map(tokenize, batched=True, remove_columns=list(dataset.column_names))


def make_training_args(output_dir: str, max_steps: int, hub_model_id: str, role: str) -> TrainingArguments:
    requested = {
        "output_dir": output_dir,
        "max_steps": max_steps,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.03,
        "logging_steps": 1,
        "report_to": [],
        "save_strategy": "no",
        "push_to_hub": False,
        "hub_model_id": hub_model_id,
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "run_name": f"{role}-sft-smoke",
    }
    signature = inspect.signature(TrainingArguments)
    supported = {key: value for key, value in requested.items() if key in signature.parameters}
    dropped = sorted(set(requested) - set(supported))
    if dropped:
        log("training_args_dropped_unsupported_keys", dropped=dropped)
    return TrainingArguments(**supported)


def main() -> None:
    started = time.time()
    args = parse_args()
    prepare_repo()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    task_ids = [value.strip() for value in args.task_id.split(",") if value.strip()]
    output_dir = args.output_dir or f"outputs/supplymind-{args.role}-qwen-sft"
    hub_model_id = args.hub_model_id or f"rishavutk/supplymind-{args.role}-qwen-0.5b-sft"

    log("loading_tokenizer", model_id=MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = build_rows(args.role, task_ids, seeds, tokenizer, args.center_non_empty_weight)
    dataset = Dataset.from_list(rows)
    log("dataset_ready", role=args.role, rows=len(rows), task_ids=task_ids, seeds=seeds, hub_model_id=hub_model_id)
    baseline_probe(args.role, task_ids[0], seeds)

    tokenized = tokenize_dataset(dataset, tokenizer, args.max_length)
    log("tokenized_dataset_ready", rows=len(tokenized), max_length=args.max_length)

    log("loading_model", model_id=MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=make_training_args(output_dir, args.max_steps, hub_model_id, args.role),
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100),
    )
    log("training_start", role=args.role, max_steps=args.max_steps)
    trainer.train()
    log("training_done", elapsed_seconds=round(time.time() - started, 2))
    log("pushing_model", hub_model_id=hub_model_id)
    model.push_to_hub(hub_model_id)
    tokenizer.push_to_hub(hub_model_id)
    log("job_done", hub_model_id=hub_model_id, elapsed_seconds=round(time.time() - started, 2))


if __name__ == "__main__":
    main()
