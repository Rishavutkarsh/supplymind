from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from .environment import V2SupplyMindEnv
from .generator import PUBLIC_TASK_IDS
from .models import V2JointAction
from .policies import heuristic_joint_policy
from .rules import public_rules


def create_v2_router() -> APIRouter:
    router = APIRouter(prefix="/v2", tags=["SupplyMind V2"])
    env = V2SupplyMindEnv()

    @router.post("/reset")
    def reset(task_id: str | None = None, seed: int | None = None) -> dict:
        if task_id is not None and task_id not in PUBLIC_TASK_IDS and not task_id.startswith("v2_"):
            raise HTTPException(status_code=400, detail=f"Unknown v2 task_id '{task_id}'. Expected one of: {', '.join(PUBLIC_TASK_IDS)}")
        return env.reset(task_id=task_id, seed=seed).model_dump(mode="json")

    @router.get("/state")
    def state() -> dict:
        return env.state().model_dump(mode="json")

    @router.post("/step")
    def step(action: V2JointAction) -> dict:
        return env.step(action).model_dump(mode="json")

    @router.get("/heuristic-joint-action")
    def heuristic_joint_action() -> dict:
        return heuristic_joint_policy(env.state()).model_dump(mode="json")

    @router.get("/rules")
    def rules() -> dict:
        return public_rules()

    @router.get("/ui", response_class=HTMLResponse)
    def ui() -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parents[2] / "static" / "v2.html").read_text(encoding="utf-8")

    return router
