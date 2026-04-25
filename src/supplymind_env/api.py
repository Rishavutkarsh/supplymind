from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .environment import V3SupplyMindEnv
from .models import V3Action
from .policies import heuristic_policy
from .task_adapter import PUBLIC_TASK_IDS, is_public_task_id


def create_app() -> FastAPI:
    app = FastAPI(title="SupplyMind Benchmark")
    env = V3SupplyMindEnv()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parents[2] / "static" / "index.html").read_text(encoding="utf-8")

    @app.post("/reset")
    def reset(task_id: str | None = None, seed: int | None = None, pool_name: str = "test") -> dict:
        if task_id is not None and not is_public_task_id(task_id):
            raise HTTPException(
                status_code=400,
                detail=f"Unknown task_id '{task_id}'. Expected one of: {', '.join(PUBLIC_TASK_IDS)}",
            )
        return env.reset(task_id=task_id, seed=seed, pool_name=pool_name).model_dump(mode="json")

    @app.get("/state")
    def state() -> dict:
        return env.state().model_dump(mode="json")

    @app.post("/step")
    def step(action: V3Action) -> dict:
        return env.step(action).model_dump(mode="json")

    @app.get("/heuristic-action")
    def heuristic_action() -> dict:
        return heuristic_policy(env.state()).model_dump(mode="json")

    return app


app = create_app()

