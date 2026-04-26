from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from supplymind_env.api import app
from supplymind_env_v2.environment import V2SupplyMindEnv
from supplymind_env_v2.generator import BENCHMARK_PROFILE_IDS, PUBLIC_BY_INTERNAL
from supplymind_env_v2.models import V2JointAction, V2Observation, V2StepResult


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_openenv_yaml() -> dict:
    path = Path("openenv.yaml")
    check(path.exists(), "openenv.yaml is missing")
    data = yaml.safe_load(path.read_text())
    check(data["name"] == "supplymind-openenv", "openenv.yaml name mismatch")
    task_ids = [task["id"] for task in data["tasks"]]
    expected = tuple(PUBLIC_BY_INTERNAL[task_id] for task_id in BENCHMARK_PROFILE_IDS)
    check(tuple(task_ids) == expected, "openenv.yaml tasks do not match benchmark tasks")
    return data


def validate_environment_contract() -> None:
    env = V2SupplyMindEnv("medium")
    observation = env.reset(task_id="easy", seed=17031)
    check(isinstance(observation, V2Observation), "reset() must return V2Observation")
    check(observation.task_id == "easy", "reset() should expose public task id")

    state = env.state()
    check(isinstance(state, V2Observation), "state() must return V2Observation")

    step_result = env.step(V2JointAction())
    check(isinstance(step_result, V2StepResult), "step() must return V2StepResult")
    check(step_result.reward.cumulative_reward == step_result.reward.cumulative_reward, "reward object should be accessible")


def validate_inference() -> dict:
    import inference

    result = inference.score_tasks("baseline")
    check("tasks" in result and "overall_score" in result, "inference output missing keys")
    check(len(result["tasks"]) >= 3, "inference must score at least three tasks")
    for task in result["tasks"]:
        check(0.0 < float(task["score"]) < 1.0, f"task score must be strictly between 0 and 1 for {task['task_id']}")
    return result


def validate_inference_cli_output() -> None:
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    env.pop("OPENAI_API_KEY", None)
    completed = subprocess.run(
        [sys.executable, "inference.py"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    stdout = completed.stdout
    check("[START]" in stdout, "inference.py stdout is missing [START] block")
    check("[STEP]" in stdout, "inference.py stdout is missing [STEP] block")
    check("[END]" in stdout, "inference.py stdout is missing [END] block")


def validate_inference_cli_output_with_configured_llm_if_present() -> None:
    env = os.environ.copy()
    token = env.get("HF_TOKEN") or env.get("OPENAI_API_KEY")
    if not token:
        return
    completed = subprocess.run(
        [sys.executable, "inference.py"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    stdout = completed.stdout
    check("[START]" in stdout, "configured inference.py stdout is missing [START] block")
    check("[STEP]" in stdout, "configured inference.py stdout is missing [STEP] block")
    check("[END]" in stdout, "configured inference.py stdout is missing [END] block")


def validate_http_api() -> None:
    client = TestClient(app)
    health = client.get("/health")
    check(health.status_code == 200, "/health must return 200")

    for task_id in ("easy", "medium", "hard"):
        reset = client.post("/reset", params={"task_id": task_id, "seed": 12345})
        check(reset.status_code == 200, f"/reset must return 200 for {task_id}")
        reset_body = reset.json()
        check(reset_body["task_id"] == task_id, f"/reset should expose requested public task {task_id}")

    invalid_reset = client.post("/reset", params={"task_id": "unknown_network"})
    check(invalid_reset.status_code == 400, "/reset must reject unknown task_id with 400")

    reset = client.post("/reset")
    check(reset.status_code == 200, "/reset without task_id must return 200")
    check(reset.json()["task_id"] in ("easy", "medium", "hard", "train_easy", "train_medium", "train_hard", "cooperative_market", "scarcity_market", "crisis_market"), "/reset without task_id should choose a public task")

    state = client.get("/state")
    check(state.status_code == 200, "/state must return 200")

    step = client.post("/step", json={"warehouse_actions": {}, "central_action": {}})
    check(step.status_code == 200, "/step must return 200")
    step_body = step.json()
    check("observation" in step_body and "reward" in step_body, "/step response shape is invalid")


def validate_docker_build() -> None:
    completed = subprocess.run(
        ["docker", "build", "-t", "supplymind-openenv", "."],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
    )
    check(completed.returncode == 0, f"Docker build failed:\n{completed.stderr}")


def main() -> None:
    yaml_data = validate_openenv_yaml()
    validate_environment_contract()
    inference_result = validate_inference()
    validate_inference_cli_output()
    validate_inference_cli_output_with_configured_llm_if_present()
    validate_http_api()
    validate_docker_build()

    summary = {
        "status": "ok",
        "openenv_name": yaml_data["name"],
        "tasks": [task["id"] for task in yaml_data["tasks"]],
        "inference": inference_result,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

