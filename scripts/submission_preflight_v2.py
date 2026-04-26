from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from supplymind_env.api import app
from supplymind_env_v2.models import V2Observation
from supplymind_env_v2.policies import heuristic_joint_policy, no_op_policy


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_episode(client: TestClient, task_id: str, policy_name: str) -> dict[str, Any]:
    reset = client.post("/v2/reset", json={"task_id": task_id})
    check(reset.status_code == 200, f"/v2/reset failed for {task_id}: {reset.text}")
    observation = reset.json()
    total_invalid = 0
    rounds = 0
    done = False
    result_body: dict[str, Any] = {}
    while not done:
        if policy_name == "no_op":
            action = no_op_policy(None).model_dump(mode="json")
        else:
            action = heuristic_joint_policy(V2Observation.model_validate(observation)).model_dump(mode="json")
        step = client.post("/v2/step", json=action)
        check(step.status_code == 200, f"/v2/step failed for {task_id}/{policy_name}: {step.text}")
        result_body = step.json()
        observation = result_body["observation"]
        done = bool(result_body["done"])
        rounds += 1
        total_invalid += len(observation.get("feedback", {}).get("invalid_action_details", []))
        check(rounds <= 80, f"episode did not terminate for {task_id}/{policy_name}")
    summary = result_body["info"].get("episode_summary", {})
    return {
        "task_id": task_id,
        "policy": policy_name,
        "rounds": rounds,
        "raw_reward": summary.get("raw_reward"),
        "graded_score": summary.get("graded_score"),
        "invalid_actions": total_invalid,
    }


def validate_v2_api(client: TestClient) -> dict[str, Any]:
    health = client.get("/health")
    check(health.status_code == 200, "/health must return 200")

    rules = client.get("/v2/rules")
    check(rules.status_code == 200, "/v2/rules must return 200")
    check("action_space" in rules.json(), "/v2/rules should expose action_space")

    task_summaries = []
    for task_id in ("easy", "medium", "hard"):
        reset = client.post("/v2/reset", json={"task_id": task_id})
        check(reset.status_code == 200, f"/v2/reset JSON body failed for {task_id}")
        body = reset.json()
        check(body["task_id"] == task_id, f"/v2/reset returned wrong task for {task_id}: {body['task_id']}")

        state = client.get("/v2/state")
        check(state.status_code == 200, "/v2/state must return 200")

        step = client.post("/v2/step", json={})
        check(step.status_code == 200, "/v2/step must accept empty joint action")
        step_body = step.json()
        check("observation" in step_body and "reward" in step_body and "done" in step_body, "/v2/step shape invalid")
        task_summaries.append({"task_id": task_id, "reset_ok": True, "empty_step_ok": True})

    for role in ("center", "warehouse"):
        reset = client.post(f"/v2/{role}/reset", json={"task_id": "train_easy"})
        check(reset.status_code == 200, f"/v2/{role}/reset must return 200")
        body = reset.json()
        check(body.get("role") == role, f"/v2/{role}/reset should expose role")
        check("controlled_action_schema" in body, f"/v2/{role}/reset should expose controlled_action_schema")

    return {"tasks": task_summaries, "roles": ["center", "warehouse"]}


def load_eval_summary() -> dict[str, Any] | None:
    path = ROOT / "results" / "v2_policy_eval.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("summary")


def validate_repo_files() -> list[str]:
    required = ["app.py", "Dockerfile", "openenv.yaml", "inference.py", "requirements.txt"]
    missing = [name for name in required if not (ROOT / name).exists()]
    check(not missing, f"missing required files: {missing}")
    return required


def main() -> None:
    client = TestClient(app)
    files = validate_repo_files()
    api = validate_v2_api(client)
    episodes = [
        run_episode(client, "train_easy", "no_op"),
        run_episode(client, "train_easy", "heuristic"),
    ]
    scores = [float(row["graded_score"]) for row in episodes if row["graded_score"] is not None]
    report = {
        "status": "ok",
        "checked_files": files,
        "api": api,
        "episode_smoke": episodes,
        "mean_smoke_score": round(mean(scores), 4) if scores else None,
        "latest_eval_summary": load_eval_summary(),
    }
    out = ROOT / "results" / "submission_preflight_v2.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
