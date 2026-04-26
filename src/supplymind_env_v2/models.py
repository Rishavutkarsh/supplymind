from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SKU = Literal["fresh_milk", "rice_bag_5kg", "insulin_pack", "usb_c_charger"]
OrderStatus = Literal["pending", "accepted", "rejected", "fulfilled", "expired"]
Decision = Literal["accept", "reject"]


class DifficultyProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    warehouse_count: int
    total_rounds: int
    depot_trucks: int
    depot_replenishment_cap: int
    depot_procurement_cap: int
    depot_procurement_lead_time: int
    transfer_cap: int


class WarehouseSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    warehouse_id: str
    label: str
    region: str
    safety_stock: dict[SKU, int]
    route_costs: dict[str, float]
    route_times: dict[str, int]


class OrderTemplate(BaseModel):
    model_config = ConfigDict(frozen=True)

    order_id: str
    created_round: int
    warehouse_id: str
    sku: SKU
    units: int
    customer_value_per_unit: float
    deadline_round: int


class HiddenRecipe(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    seed: int
    profile: DifficultyProfile
    warehouse_specs: tuple[WarehouseSpec, ...]
    initial_inventory: dict[str, dict[SKU, int]]
    initial_drivers: dict[str, int]
    central_depot_inventory: dict[SKU, int]
    orders: tuple[OrderTemplate, ...]
    public_forecasts: tuple[dict[str, Any], ...] = Field(default_factory=tuple)


class LocalOrderSnapshot(BaseModel):
    order_id: str
    sku: SKU
    units: int
    customer_value_per_unit: float
    created_round: int
    deadline_round: int
    status: OrderStatus


class PendingTransferProposal(BaseModel):
    proposal_id: str
    from_warehouse: str
    to_warehouse: str
    sku: SKU
    units: int
    compensation: float


class WarehouseObservation(BaseModel):
    warehouse_id: str
    label: str
    region: str
    inventory: dict[SKU, int]
    inventory_age: dict[SKU, float] = Field(default_factory=dict)
    drivers_available: int
    local_orders: list[LocalOrderSnapshot]
    order_action_context: list[dict[str, Any]] = Field(default_factory=list)
    pending_transfer_proposals: list[PendingTransferProposal] = Field(default_factory=list)
    safety_stock: dict[SKU, int]
    route_costs: dict[str, float]
    last_reward: float = 0.0


class CenterObservation(BaseModel):
    round_index: int
    remaining_rounds: int
    depot_inventory: dict[SKU, int]
    depot_inventory_age: dict[SKU, float] = Field(default_factory=dict)
    depot_trucks_available: int
    inbound_procurements: list[dict[str, int | str]] = Field(default_factory=list)
    warehouse_summaries: list[dict[str, Any]] = Field(default_factory=list)
    market_signals: list[dict[str, Any]] = Field(default_factory=list)
    pending_transfer_proposals: list[PendingTransferProposal] = Field(default_factory=list)
    price_bands: dict[str, dict[str, float]] = Field(default_factory=dict)


class OrderDecision(BaseModel):
    order_id: str
    decision: Decision


class InventoryOffer(BaseModel):
    sku: SKU
    units: int
    ask_price: float


class InventoryRequest(BaseModel):
    sku: SKU
    units: int
    max_price: float


class TransferResponse(BaseModel):
    proposal_id: str
    decision: Decision


class LocalPriority(BaseModel):
    sku: SKU
    priority: int = 1


class WarehouseAction(BaseModel):
    order_decisions: list[OrderDecision] = Field(default_factory=list)
    inventory_offers: list[InventoryOffer] = Field(default_factory=list)
    inventory_requests: list[InventoryRequest] = Field(default_factory=list)
    transfer_responses: list[TransferResponse] = Field(default_factory=list)
    local_priority: list[LocalPriority] = Field(default_factory=list)


class CentralProcurement(BaseModel):
    sku: SKU
    units: int
    max_unit_cost: float


class CentralLiquidation(BaseModel):
    sku: SKU
    units: int


class CentralReplenishment(BaseModel):
    to_warehouse: str
    sku: SKU
    units: int
    unit_price: float


class InventoryTransferProposal(BaseModel):
    from_warehouse: str
    to_warehouse: str
    sku: SKU
    units: int
    compensation: float


class OfferMatch(BaseModel):
    offer_signal_id: str
    request_signal_id: str
    units: int
    compensation: float


class CenterAction(BaseModel):
    central_procurements: list[CentralProcurement] = Field(default_factory=list)
    central_liquidations: list[CentralLiquidation] = Field(default_factory=list)
    central_replenishments: list[CentralReplenishment] = Field(default_factory=list)
    inventory_transfer_proposals: list[InventoryTransferProposal] = Field(default_factory=list)
    offer_matches: list[OfferMatch] = Field(default_factory=list)


class V2JointAction(BaseModel):
    warehouse_actions: dict[str, WarehouseAction] = Field(default_factory=dict)
    central_action: CenterAction = Field(default_factory=CenterAction)


class V2WarehouseRoleAction(BaseModel):
    warehouse_actions: dict[str, WarehouseAction] = Field(default_factory=dict)


class V2Reward(BaseModel):
    step_reward: float
    cumulative_reward: float
    components: dict[str, float] = Field(default_factory=dict)


class V2Observation(BaseModel):
    round_index: int
    remaining_rounds: int
    task_id: str
    scenario_info: dict[str, Any] = Field(default_factory=dict)
    center: CenterObservation
    warehouses: dict[str, WarehouseObservation]
    feedback: dict[str, Any] = Field(default_factory=dict)


class V2StepResult(BaseModel):
    observation: V2Observation
    reward: V2Reward
    done: bool
    info: dict[str, Any]


class V2TaskResult(BaseModel):
    task_id: str
    raw_reward: float
    baseline_reward: float
    target_reward: float
    score: float
    center_reward: float
    average_warehouse_reward: float
