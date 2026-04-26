from __future__ import annotations

import json
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from supplymind_env_v2.api import create_v2_router
from supplymind_env_v2.generator import PUBLIC_TASK_IDS
from supplymind_env_v2.models import V2JointAction
from supplymind_env_v2.environment import V2SupplyMindEnv
from supplymind_env_v2.policies import heuristic_joint_policy


class ResetRequest(BaseModel):
    task_id: str | None = None
    seed: int | None = None


def create_app() -> FastAPI:
    app = FastAPI(title="SupplyMind Benchmark")
    env = V2SupplyMindEnv(default_task_id="easy")
    app.include_router(create_v2_router())

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parents[2] / "static" / "v2.html").read_text(encoding="utf-8")

    @app.post("/reset")
    def reset(
        payload: ResetRequest | None = Body(default=None),
        task_id: str | None = None,
        seed: int | None = None,
        pool_name: str = "test",
    ) -> dict:
        del pool_name
        resolved_task_id = task_id if task_id is not None else None if payload is None else payload.task_id
        resolved_seed = seed if seed is not None else None if payload is None else payload.seed
        if resolved_task_id is not None and resolved_task_id not in PUBLIC_TASK_IDS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown task_id '{resolved_task_id}'. Expected one of: {', '.join(PUBLIC_TASK_IDS)}",
            )
        return env.reset(task_id=resolved_task_id, seed=resolved_seed).model_dump(mode="json")

    @app.get("/state")
    def state() -> dict:
        return env.state().model_dump(mode="json")

    @app.post("/step")
    def step(action: V2JointAction) -> dict:
        return env.step(action).model_dump(mode="json")

    @app.get("/heuristic-action")
    def heuristic_action() -> dict:
        return heuristic_joint_policy(env.state()).model_dump(mode="json")

    @app.get("/blackbox-trace")
    def blackbox_trace() -> dict:
        from pathlib import Path

        trace_path = Path(__file__).resolve().parents[2] / "results" / "blackbox_codex_subagent_episode.json"
        if not trace_path.exists():
            raise HTTPException(status_code=404, detail="No black-box subagent trace has been saved yet.")
        return json.loads(trace_path.read_text(encoding="utf-8-sig"))

    return app


app = create_app()

