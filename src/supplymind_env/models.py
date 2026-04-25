from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SKU = Literal["fresh_milk", "rice_bag_5kg", "insulin_pack", "usb_c_charger"]
Personality = Literal["cooperative", "risk_averse", "selfish", "opportunistic"]
OrderStatus = Literal["open", "fulfilled", "expired"]
ProposalStatus = Literal["accepted", "rejected", "countered", "invalid"]


class DifficultyProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    warehouse_count: int
    total_rounds: int
    transfer_cap: int
    truck_capacity: int
    depot_replenishment_cap: int = 0
    depot_trucks: int = 1
    depot_procurement_cap: int = 0
    depot_procurement_lead_time: int = 2
    starting_trust: float
    stockout_penalty_multiplier: float
    holding_cost_per_unit: float
    waste_cost_per_unit: float
    fairness_penalty_weight: float
    coalition_bonus: float


class WarehouseSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    warehouse_id: str
    label: str
    region: str
    personality: Personality
    safety_stock: dict[SKU, int]
    delivery_cost_by_region: dict[str, float]
    delivery_time_by_region: dict[str, int]


class OrderTemplate(BaseModel):
    model_config = ConfigDict(frozen=True)

    order_id: str
    created_round: int
    region: str
    sku: SKU
    units: int
    value_per_unit: float
    deadline_round: int
    priority: int = 1


class HiddenRecipe(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    seed: int
    profile: DifficultyProfile
    warehouse_specs: tuple[WarehouseSpec, ...]
    initial_inventory: dict[str, dict[SKU, int]]
    central_depot_inventory: dict[SKU, int] = Field(default_factory=dict)
    initial_drivers: dict[str, int]
    private_forecasts: dict[str, dict[SKU, int]]
    orders: tuple[OrderTemplate, ...]


class WarehouseSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    warehouse_id: str
    label: str
    region: str
    inventory: dict[SKU, int]
    drivers_available: int
    drivers_returning: list[int] = Field(default_factory=list)
    route_costs: dict[str, float] = Field(default_factory=dict)
    public_message: str


class CentralDepotSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    inventory: dict[SKU, int]
    trucks_available: int
    trucks_returning: list[int] = Field(default_factory=list)
    replenishment_cap: int
    procurement_cap: int = 0
    procurement_lead_time: int = 0
    inbound_procurements: list[dict[str, int | str]] = Field(default_factory=list)
    message: str = ""


class OrderSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    order_id: str
    region: str
    sku: SKU
    units: int
    value_per_unit: float
    deadline_round: int
    priority: int
    status: OrderStatus = "open"


class DemandReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    warehouse_id: str
    region: str
    sku: SKU
    requested_units: int = 0
    fulfilled_units_last_round: int = 0
    missed_units_last_round: int = 0
    at_risk_units: int = 0
    forecast_units: int = 0
    urgency: int = 1
    message: str = ""


class FulfillmentDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    warehouse_id: str
    order_id: str


class InventoryTransferProposal(BaseModel):
    model_config = ConfigDict(frozen=True)

    from_warehouse: str
    to_warehouse: str
    sku: SKU
    units: int
    compensation: float = 0.0


class DriverLoanProposal(BaseModel):
    model_config = ConfigDict(frozen=True)

    from_warehouse: str
    to_warehouse: str
    driver_count: int
    compensation: float = 0.0


class CentralReplenishmentProposal(BaseModel):
    model_config = ConfigDict(frozen=True)

    to_warehouse: str
    sku: SKU
    units: int


class CentralProcurementProposal(BaseModel):
    model_config = ConfigDict(frozen=True)

    sku: SKU
    units: int


class MarketSignal(BaseModel):
    model_config = ConfigDict(frozen=True)

    signal_id: str
    warehouse_id: str
    signal_type: Literal["inventory_offer", "inventory_request", "driver_offer", "driver_request", "demand_warning"]
    sku: SKU | None = None
    units: int = 0
    driver_count: int = 0
    ask_price: float = 0.0
    urgency: int = 1
    message: str = ""


class OfferMatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    offer_signal_id: str
    request_signal_id: str
    units: int
    compensation: float = 0.0


class PriorityRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    sku: SKU | None = None
    region: str | None = None
    priority: int = 1


class CoalitionDeal(BaseModel):
    model_config = ConfigDict(frozen=True)

    deal_id: str
    order_id: str
    supporting_warehouses: list[str] = Field(default_factory=list)
    compensation_pool: float = 0.0


class V3Action(BaseModel):
    fulfillments: list[FulfillmentDecision] = Field(default_factory=list)
    central_procurements: list[CentralProcurementProposal] = Field(default_factory=list)
    central_replenishments: list[CentralReplenishmentProposal] = Field(default_factory=list)
    inventory_transfers: list[InventoryTransferProposal] = Field(default_factory=list)
    driver_loans: list[DriverLoanProposal] = Field(default_factory=list)
    offer_matches: list[OfferMatch] = Field(default_factory=list)
    priority_policy: list[PriorityRule] = Field(default_factory=list)
    defer_orders: list[str] = Field(default_factory=list)
    coalition_deals: list[CoalitionDeal] = Field(default_factory=list)


class NegotiationEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    actor: str
    proposal_type: str
    status: ProposalStatus
    summary: str
    local_utility_delta: float = 0.0


class V3Reward(BaseModel):
    step_reward: float
    cumulative_reward: float
    components: dict[str, float] = Field(default_factory=dict)


class V3Feedback(BaseModel):
    last_step_reward: float = 0.0
    cumulative_reward: float = 0.0
    recent_events: list[str] = Field(default_factory=list)
    negotiation_trace: list[NegotiationEvent] = Field(default_factory=list)
    reward_components: dict[str, float] = Field(default_factory=dict)
    agent_rewards: dict[str, float] = Field(default_factory=dict)
    current_pressure: str = ""


class V3ScenarioInfo(BaseModel):
    task_id: str
    used_seed: int
    total_rounds: int
    transfer_cap: int
    truck_capacity: int
    objective_brief: str = ""
    action_brief: str = ""
    episode_brief: str = ""


class V3Observation(BaseModel):
    round_index: int
    remaining_rounds: int
    task_id: str
    central_depot: CentralDepotSnapshot
    warehouses: list[WarehouseSnapshot]
    market_signals: list[MarketSignal] = Field(default_factory=list)
    demand_reports: list[DemandReport] = Field(default_factory=list)
    open_orders: list[OrderSnapshot] = Field(default_factory=list)
    feedback: V3Feedback
    scenario_info: V3ScenarioInfo


class V3StepResult(BaseModel):
    observation: V3Observation
    reward: V3Reward
    done: bool
    info: dict[str, Any]


class V3TaskResult(BaseModel):
    task_id: str
    raw_reward: float
    baseline_reward: float
    target_reward: float
    score: float
    heuristic_reward: float | None = None
