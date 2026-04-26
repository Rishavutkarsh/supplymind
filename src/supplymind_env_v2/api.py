from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .environment import V2SupplyMindEnv
from .generator import PUBLIC_TASK_IDS
from .models import CenterAction, V2JointAction, V2WarehouseRoleAction
from .policies import fixed_center_action, fixed_warehouse_actions, heuristic_joint_policy
from .rules import public_rules


class ResetRequest(BaseModel):
    task_id: str | None = None
    seed: int | None = None


def create_v2_router() -> APIRouter:
    router = APIRouter(prefix="/v2", tags=["SupplyMind V2"])
    env = V2SupplyMindEnv()
    center_env = V2SupplyMindEnv()
    warehouse_env = V2SupplyMindEnv()

    @router.post("/reset")
    def reset(payload: ResetRequest | None = Body(default=None), task_id: str | None = None, seed: int | None = None) -> dict:
        resolved_task_id = task_id if task_id is not None else None if payload is None else payload.task_id
        resolved_seed = seed if seed is not None else None if payload is None else payload.seed
        _validate_task_id(resolved_task_id)
        return env.reset(task_id=resolved_task_id, seed=resolved_seed).model_dump(mode="json")

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

    @router.post("/center/reset")
    def center_reset(payload: ResetRequest | None = Body(default=None), task_id: str | None = None, seed: int | None = None) -> dict:
        resolved_task_id = task_id if task_id is not None else None if payload is None else payload.task_id
        resolved_seed = seed if seed is not None else None if payload is None else payload.seed
        _validate_task_id(resolved_task_id)
        observation = center_env.reset(task_id=resolved_task_id, seed=resolved_seed)
        _apply_center_training_pressure(center_env)
        observation = center_env.state()
        return _center_role_payload(observation)

    @router.get("/center/state")
    def center_state() -> dict:
        return _center_role_payload(center_env.state())

    @router.post("/center/step")
    def center_step(action: CenterAction) -> dict:
        observation = center_env.state()
        frozen_warehouses = fixed_warehouse_actions(observation)
        result = center_env.step(V2JointAction(warehouse_actions=frozen_warehouses, central_action=action))
        payload = result.model_dump(mode="json")
        payload["role"] = "center"
        payload["frozen_warehouse_actions"] = {key: value.model_dump(mode="json") for key, value in frozen_warehouses.items()}
        return payload

    @router.post("/warehouse/reset")
    def warehouse_reset(payload: ResetRequest | None = Body(default=None), task_id: str | None = None, seed: int | None = None) -> dict:
        resolved_task_id = task_id if task_id is not None else None if payload is None else payload.task_id
        resolved_seed = seed if seed is not None else None if payload is None else payload.seed
        _validate_task_id(resolved_task_id)
        observation = warehouse_env.reset(task_id=resolved_task_id, seed=resolved_seed)
        return _warehouse_role_payload(observation)

    @router.get("/warehouse/state")
    def warehouse_state() -> dict:
        return _warehouse_role_payload(warehouse_env.state())

    @router.post("/warehouse/step")
    def warehouse_step(action: V2WarehouseRoleAction) -> dict:
        observation = warehouse_env.state()
        center_action = fixed_center_action(observation, action.warehouse_actions)
        result = warehouse_env.step(V2JointAction(warehouse_actions=action.warehouse_actions, central_action=center_action))
        payload = result.model_dump(mode="json")
        payload["role"] = "warehouse"
        payload["frozen_center_action"] = center_action.model_dump(mode="json")
        return payload

    @router.get("/ui", response_class=HTMLResponse)
    def ui() -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parents[2] / "static" / "v2.html").read_text(encoding="utf-8")

    @router.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parents[2] / "static" / "dashboard.html").read_text(encoding="utf-8")

    @router.get("/training-results")
    def training_results() -> dict:
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "results" / "training_dashboard.json"
        if not path.exists():
            return {
                "status": "pending",
                "message": "No training dashboard results have been saved yet.",
                "runs": [],
                "comparisons": [],
            }
        return json.loads(path.read_text(encoding="utf-8"))

    return router


def _validate_task_id(task_id: str | None) -> None:
    if task_id is not None and task_id not in PUBLIC_TASK_IDS and not task_id.startswith("v2_"):
        raise HTTPException(status_code=400, detail=f"Unknown v2 task_id '{task_id}'. Expected one of: {', '.join(PUBLIC_TASK_IDS)}")


def _center_role_payload(observation) -> dict:
    frozen_warehouses = fixed_warehouse_actions(observation)
    payload = observation.model_dump(mode="json")
    payload["role"] = "center"
    payload["controlled_action_schema"] = {
        "central_procurements": [{"sku": "fresh_milk", "units": 4, "max_unit_cost": 4.0}],
        "central_liquidations": [{"sku": "fresh_milk", "units": 2}],
        "central_replenishments": [{"to_warehouse": "north", "sku": "insulin_pack", "units": 2, "unit_price": 12.0}],
        "inventory_transfer_proposals": [{"from_warehouse": "west", "to_warehouse": "east", "sku": "rice_bag_5kg", "units": 2, "compensation": 10.0}],
        "offer_matches": [{"offer_signal_id": "west:offer:rice_bag_5kg", "request_signal_id": "east:request:rice_bag_5kg", "units": 2, "compensation": 10.0}],
    }
    payload["frozen_warehouse_action_preview"] = {key: value.model_dump(mode="json") for key, value in frozen_warehouses.items()}
    payload["fixed_counterparty"] = "warehouses use deterministic heuristic actions generated from the same public observation"
    payload["role_training_variant"] = "center_pressure"
    return payload


def _warehouse_role_payload(observation) -> dict:
    fixed_center = fixed_center_action(observation, fixed_warehouse_actions(observation))
    payload = observation.model_dump(mode="json")
    payload["role"] = "warehouse"
    payload["controlled_action_schema"] = {
        "warehouse_actions": {
            "<warehouse_id>": {
                "order_decisions": [{"order_id": "o17", "decision": "accept"}],
                "inventory_offers": [{"sku": "fresh_milk", "units": 2, "ask_price": 6.0}],
                "inventory_requests": [{"sku": "insulin_pack", "units": 3, "max_price": 12.0}],
                "transfer_responses": [{"proposal_id": "p1", "decision": "accept"}],
                "local_priority": [{"sku": "insulin_pack", "priority": 3}],
            }
        }
    }
    payload["frozen_center_action_preview"] = fixed_center.model_dump(mode="json")
    payload["fixed_counterparty"] = "center uses deterministic heuristic procurement, replenishment, and offer matching"
    return payload


def _apply_center_training_pressure(env: V2SupplyMindEnv) -> None:
    """Make center-role episodes require real center action without changing joint eval."""
    recipe = env._require_recipe()
    priority_skus = ("insulin_pack", "usb_c_charger", "fresh_milk")
    for index, spec in enumerate(recipe.warehouse_specs):
        sku = priority_skus[index % len(priority_skus)]
        env.inventory[spec.warehouse_id][sku] = min(env.inventory[spec.warehouse_id].get(sku, 0), 1 if sku != "fresh_milk" else 2)
        env.central_depot_inventory[sku] = max(env.central_depot_inventory.get(sku, 0), 8)
    env.last_events = ["center training pressure applied: selected warehouse stocks were tightened and depot stock was preserved"]
