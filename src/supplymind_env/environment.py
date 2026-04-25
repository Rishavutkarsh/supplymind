from __future__ import annotations

import secrets
from copy import deepcopy

from .dynamics import (
    acceptance_decision,
    add_component,
    clamp,
    fairness_penalty,
    generate_market_signals,
    holding_and_waste_cost,
    transfer_cost,
    visible_orders,
    warehouse_message,
)
from .generator import generate_recipe
from .models import (
    CentralDepotSnapshot,
    DemandReport,
    HiddenRecipe,
    InventoryTransferProposal,
    NegotiationEvent,
    OrderSnapshot,
    OrderTemplate,
    V3Action,
    V3Feedback,
    V3Observation,
    V3Reward,
    V3ScenarioInfo,
    V3StepResult,
    WarehouseSnapshot,
)
from .task_adapter import to_internal_task_id, to_public_task_id


class V3SupplyMindEnv:
    def __init__(self, default_task_id: str = "scarcity_negotiation") -> None:
        self.default_task_id = default_task_id
        self.recipe: HiddenRecipe | None = None
        self.task_id = default_task_id
        self.internal_task_id = to_internal_task_id(default_task_id)
        self.public_seed = 0
        self.internal_seed = 0
        self.pool_name = "test"
        self.round_index = 0
        self.inventory: dict[str, dict[str, int]] = {}
        self.central_depot_inventory: dict[str, int] = {}
        self.depot_trucks_available = 0
        self.depot_truck_returns: list[int] = []
        self.drivers_available: dict[str, int] = {}
        self.driver_returns: dict[str, list[int]] = {}
        self.trust: dict[str, float] = {}
        self.local_utility: dict[str, float] = {}
        self.last_fulfilled_units: dict[tuple[str, str], int] = {}
        self.last_missed_units: dict[tuple[str, str], int] = {}
        self.completed_orders: set[str] = set()
        self.expired_orders: set[str] = set()
        self.cumulative_reward = 0.0
        self.last_step_reward = 0.0
        self.last_components: dict[str, float] = {}
        self.recent_events: list[str] = []
        self.negotiation_trace: list[NegotiationEvent] = []
        self.done = False
        self.last_episode_summary: dict[str, object] | None = None

    def reset(self, task_id: str | None = None, seed: int | None = None, pool_name: str = "test") -> V3Observation:
        self.pool_name = pool_name
        self.public_seed = seed if seed is not None else secrets.randbelow(1_000_000_000)
        from .seed_catalog import choose_random_curated_seed, choose_random_task_id, resolve_curated_seed, resolve_task_id

        if task_id is None:
            self.internal_task_id = resolve_task_id(self.public_seed) if seed is not None else choose_random_task_id()
        else:
            self.internal_task_id = to_internal_task_id(task_id)
        self.task_id = to_public_task_id(self.internal_task_id)
        self.internal_seed = (
            choose_random_curated_seed(self.internal_task_id, pool_name=self.pool_name)
            if seed is None
            else resolve_curated_seed(self.internal_task_id, self.public_seed, pool_name=self.pool_name)
        )
        self._load_recipe(generate_recipe(self.internal_task_id, self.internal_seed))
        return self.state()

    def reset_internal(
        self,
        task_id: str,
        internal_seed: int,
        public_seed: int | None = None,
        pool_name: str = "test",
    ) -> V3Observation:
        self.internal_task_id = to_internal_task_id(task_id)
        self.task_id = to_public_task_id(self.internal_task_id)
        self.public_seed = internal_seed if public_seed is None else public_seed
        self.internal_seed = internal_seed
        self.pool_name = pool_name
        self._load_recipe(generate_recipe(self.internal_task_id, self.internal_seed))
        return self.state()

    def _load_recipe(self, recipe: HiddenRecipe) -> None:
        self.recipe = recipe
        self.round_index = 0
        self.inventory = deepcopy(recipe.initial_inventory)
        self.central_depot_inventory = dict(recipe.central_depot_inventory)
        self.depot_trucks_available = recipe.profile.depot_trucks
        self.depot_truck_returns = []
        self.drivers_available = dict(recipe.initial_drivers)
        self.driver_returns = {warehouse_id: [] for warehouse_id in self.inventory}
        self.trust = {spec.warehouse_id: recipe.profile.starting_trust for spec in recipe.warehouse_specs}
        self.local_utility = {spec.warehouse_id: 0.0 for spec in recipe.warehouse_specs}
        self.last_fulfilled_units = {}
        self.last_missed_units = {}
        self.completed_orders = set()
        self.expired_orders = set()
        self.cumulative_reward = 0.0
        self.last_step_reward = 0.0
        self.last_components = {}
        self.recent_events = ["environment reset"]
        self.negotiation_trace = []
        self.done = False
        self.last_episode_summary = None

    def state(self) -> V3Observation:
        recipe = self._require_recipe()
        specs_by_id = {spec.warehouse_id: spec for spec in recipe.warehouse_specs}
        open_orders = visible_orders(recipe, self.round_index, self.completed_orders, self.expired_orders)
        demand_reports = _demand_reports(
            recipe,
            open_orders,
            self.inventory,
            self.last_fulfilled_units,
            self.last_missed_units,
            self.round_index,
        )
        return V3Observation(
            round_index=self.round_index,
            remaining_rounds=max(0, recipe.profile.total_rounds - self.round_index),
            task_id=self.task_id,
            central_depot=CentralDepotSnapshot(
                inventory=dict(self.central_depot_inventory),
                trucks_available=self.depot_trucks_available,
                trucks_returning=sorted(self.depot_truck_returns),
                replenishment_cap=recipe.profile.depot_replenishment_cap,
                message=_depot_message(self.central_depot_inventory, self.depot_trucks_available),
            ),
            warehouses=[
                WarehouseSnapshot(
                    warehouse_id=spec.warehouse_id,
                    label=spec.label,
                    region=spec.region,
                    inventory=dict(self.inventory[spec.warehouse_id]),
                    drivers_available=self.drivers_available[spec.warehouse_id],
                    drivers_returning=sorted(self.driver_returns[spec.warehouse_id]),
                    route_costs=dict(spec.delivery_cost_by_region),
                    public_message=warehouse_message(
                        spec,
                        self.inventory[spec.warehouse_id],
                        self.drivers_available[spec.warehouse_id],
                        recipe.private_forecasts[spec.warehouse_id],
                        self.trust[spec.warehouse_id],
                    ),
                )
                for spec in recipe.warehouse_specs
            ],
            market_signals=generate_market_signals(recipe, self.inventory, self.drivers_available, self.trust),
            demand_reports=demand_reports,
            open_orders=[],
            feedback=V3Feedback(
                last_step_reward=round(self.last_step_reward, 3),
                cumulative_reward=round(self.cumulative_reward, 3),
                recent_events=list(self.recent_events),
                negotiation_trace=list(self.negotiation_trace),
                reward_components=dict(self.last_components),
                agent_rewards={key: round(value, 3) for key, value in self.local_utility.items()},
                current_pressure=_pressure_summary_from_reports(demand_reports),
            ),
            scenario_info=V3ScenarioInfo(
                task_id=self.task_id,
                used_seed=self.public_seed,
                total_rounds=recipe.profile.total_rounds,
                transfer_cap=recipe.profile.transfer_cap,
                truck_capacity=recipe.profile.truck_capacity,
                objective_brief="Maximize global welfare from compressed warehouse demand reports.",
                action_brief="Submit depot replenishments, approved offer matches, and transfer proposals. Warehouses handle local orders.",
                episode_brief="Each step resolves central allocation, local warehouse fulfillment, deliveries, and expiries.",
            ),
        )

    def step(self, action: V3Action, grade_terminal: bool = True) -> V3StepResult:
        if self.done:
            return V3StepResult(
                observation=self.state(),
                reward=V3Reward(step_reward=0.0, cumulative_reward=self.cumulative_reward, components={}),
                done=True,
                info={"episode_summary": self.last_episode_summary or {}},
            )

        recipe = self._require_recipe()
        self._return_drivers()
        self._return_depot_trucks()
        components: dict[str, float] = {}
        events: list[str] = []
        trace: list[NegotiationEvent] = []
        specs_by_id = {spec.warehouse_id: spec for spec in recipe.warehouse_specs}
        open_orders = {order.order_id: order for order in visible_orders(recipe, self.round_index, self.completed_orders, self.expired_orders)}
        market_signals = {
            signal.signal_id: signal
            for signal in generate_market_signals(recipe, self.inventory, self.drivers_available, self.trust)
        }

        moved_units = 0
        depot_units = 0
        for replenishment in action.central_replenishments:
            if replenishment.to_warehouse not in specs_by_id:
                add_component(components, "invalid_action_penalty", -5.0)
                continue
            if replenishment.units <= 0 or replenishment.sku not in self.central_depot_inventory:
                add_component(components, "invalid_action_penalty", -5.0)
                continue
            if depot_units + replenishment.units > recipe.profile.depot_replenishment_cap:
                add_component(components, "invalid_action_penalty", -4.0)
                events.append("central depot replenishment cap exceeded")
                continue
            if replenishment.units > self.central_depot_inventory[replenishment.sku]:
                add_component(components, "invalid_action_penalty", -4.0)
                events.append(f"central depot lacked {replenishment.sku}")
                continue
            if self.depot_trucks_available <= 0:
                add_component(components, "invalid_action_penalty", -4.0)
                events.append("central depot had no truck available")
                continue
            target_spec = specs_by_id[replenishment.to_warehouse]
            trip_time = max(1, target_spec.delivery_time_by_region.get(target_spec.region, 1))
            depot_cost = (1.2 + target_spec.delivery_cost_by_region.get(target_spec.region, 1.5)) * replenishment.units
            self.central_depot_inventory[replenishment.sku] -= replenishment.units
            self.inventory[replenishment.to_warehouse][replenishment.sku] += replenishment.units
            self.depot_trucks_available -= 1
            self.depot_truck_returns.append(self.round_index + trip_time)
            depot_units += replenishment.units
            add_component(components, "central_replenishment_cost", -depot_cost)
            add_component(components, "central_replenishment_bonus", 0.25 * replenishment.units)
            events.append(
                f"central depot sent {replenishment.units} {replenishment.sku} to {replenishment.to_warehouse}"
            )

        expanded_transfers = list(action.inventory_transfers)
        for match in action.offer_matches:
            offer = market_signals.get(match.offer_signal_id)
            request = market_signals.get(match.request_signal_id)
            if (
                offer is None
                or request is None
                or offer.signal_type != "inventory_offer"
                or request.signal_type != "inventory_request"
                or offer.sku != request.sku
                or match.units <= 0
            ):
                add_component(components, "invalid_action_penalty", -4.0)
                continue
            expanded_transfers.append(
                InventoryTransferProposal(
                    from_warehouse=offer.warehouse_id,
                    to_warehouse=request.warehouse_id,
                    sku=offer.sku,
                    units=min(match.units, offer.units, request.units),
                    compensation=max(match.compensation, offer.ask_price * max(1, min(match.units, offer.units, request.units))),
                )
            )
            events.append(
                f"central approved {min(match.units, offer.units, request.units)} {offer.sku} from "
                f"{offer.warehouse_id} to {request.warehouse_id} via transfer truck capacity"
            )

        for proposal in expanded_transfers:
            if not hasattr(proposal, "from_warehouse"):
                continue
            if proposal.from_warehouse not in specs_by_id or proposal.to_warehouse not in specs_by_id:
                add_component(components, "invalid_action_penalty", -6.0)
                continue
            if proposal.units <= 0 or proposal.sku not in self.inventory[proposal.from_warehouse]:
                add_component(components, "invalid_action_penalty", -6.0)
                continue
            if moved_units + proposal.units > recipe.profile.transfer_cap:
                add_component(components, "invalid_action_penalty", -4.0)
                trace.append(_trace(proposal.from_warehouse, "inventory_transfer", "invalid", "transfer cap exceeded", -0.4))
                continue
            source_spec = specs_by_id[proposal.from_warehouse]
            accepted, utility_delta, reason = acceptance_decision(
                source_spec,
                proposal,
                self.inventory[proposal.from_warehouse],
                recipe.private_forecasts[proposal.from_warehouse],
                self.trust[proposal.from_warehouse],
            )
            status = "accepted" if accepted else "rejected"
            trace.append(_trace(proposal.from_warehouse, "inventory_transfer", status, reason, utility_delta))
            self.local_utility[proposal.from_warehouse] += utility_delta
            if accepted:
                cost = transfer_cost(specs_by_id, proposal)
                self.inventory[proposal.from_warehouse][proposal.sku] -= proposal.units
                self.inventory[proposal.to_warehouse][proposal.sku] += proposal.units
                moved_units += proposal.units
                add_component(components, "transfer_cost", -cost)
                add_component(components, "accepted_trade_bonus", 0.4)
                self.trust[proposal.from_warehouse] = clamp(self.trust[proposal.from_warehouse] + 0.03, 0.0, 1.0)
                events.append(
                    f"{proposal.from_warehouse} transferred {proposal.units} {proposal.sku} to {proposal.to_warehouse}"
                )
            else:
                add_component(components, "rejected_trade_penalty", -0.5)
                self.trust[proposal.from_warehouse] = clamp(self.trust[proposal.from_warehouse] - 0.02, 0.0, 1.0)

        for loan in action.driver_loans:
            if loan.from_warehouse not in specs_by_id or loan.to_warehouse not in specs_by_id or loan.driver_count <= 0:
                add_component(components, "invalid_action_penalty", -5.0)
                continue
            available = self.drivers_available[loan.from_warehouse]
            if loan.driver_count > available:
                add_component(components, "invalid_action_penalty", -4.0)
                trace.append(_trace(loan.from_warehouse, "driver_loan", "invalid", "not enough drivers", -0.3))
                continue
            threshold = 2.0 if specs_by_id[loan.from_warehouse].personality in {"cooperative", "opportunistic"} else 4.0
            if loan.compensation < threshold * loan.driver_count:
                trace.append(_trace(loan.from_warehouse, "driver_loan", "rejected", "compensation too low", -0.4))
                add_component(components, "rejected_trade_penalty", -0.4)
                continue
            self.drivers_available[loan.from_warehouse] -= loan.driver_count
            self.drivers_available[loan.to_warehouse] += loan.driver_count
            self.local_utility[loan.from_warehouse] += loan.compensation - loan.driver_count
            trace.append(_trace(loan.from_warehouse, "driver_loan", "accepted", "driver loan accepted", loan.compensation - loan.driver_count))
            add_component(components, "accepted_trade_bonus", 0.4 * loan.driver_count)
            events.append(f"{loan.from_warehouse} loaned {loan.driver_count} driver(s) to {loan.to_warehouse}")

        fulfilled_orders = self._resolve_local_warehouse_fulfillment(open_orders, specs_by_id, components, events)
        used_orders = set(fulfilled_orders)

        expired_now = [
            order for order in open_orders.values()
            if order.order_id not in self.completed_orders and order.deadline_round < self.round_index
        ]
        for order in expired_now:
            self.expired_orders.add(order.order_id)
            penalty = order.units * order.value_per_unit * recipe.profile.stockout_penalty_multiplier
            add_component(components, "stockout_penalty", -penalty)
            events.append(f"{order.order_id} expired")

        holding, waste = holding_and_waste_cost(self.inventory, recipe.profile)
        add_component(components, "holding_cost", holding)
        add_component(components, "waste_penalty", waste)
        fairness = fairness_penalty(self.local_utility, recipe.profile.fairness_penalty_weight)
        add_component(components, "fairness_penalty", fairness)
        if moved_units > 0 and used_orders:
            add_component(components, "coalition_bonus", recipe.profile.coalition_bonus)
        if action.coalition_deals and moved_units > 0 and used_orders:
            add_component(components, "coalition_bonus", min(6.0, len(action.coalition_deals) * recipe.profile.coalition_bonus * 0.5))

        step_reward = sum(components.values())
        self.cumulative_reward += step_reward
        self.last_step_reward = step_reward
        self.last_components = {key: round(value, 3) for key, value in components.items() if abs(value) > 1e-9}
        self.recent_events = events or ["no accepted commitments this round"]
        self.negotiation_trace = trace
        self.round_index += 1
        self.done = self.round_index >= recipe.profile.total_rounds or len(self.completed_orders | self.expired_orders) >= len(recipe.orders)

        info: dict[str, object] = {
            "global_reward": round(step_reward, 3),
            "agent_rewards": {key: round(value, 3) for key, value in self.local_utility.items()},
            "reward_components": dict(self.last_components),
            "negotiation_trace": [event.model_dump(mode="json") for event in trace],
        }
        if self.done and grade_terminal:
            from .grading import grade_episode

            task_result = grade_episode(self.internal_task_id, self.internal_seed, self.cumulative_reward)
            self.last_episode_summary = {
                "raw_reward": round(task_result.raw_reward, 3),
                "baseline_reward": round(task_result.baseline_reward, 3),
                "target_reward": round(task_result.target_reward, 3),
                "heuristic_reward": None if task_result.heuristic_reward is None else round(task_result.heuristic_reward, 3),
                "graded_score": round(task_result.score, 4),
            }
            info["episode_summary"] = dict(self.last_episode_summary)

        return V3StepResult(
            observation=self.state(),
            reward=V3Reward(
                step_reward=round(step_reward, 3),
                cumulative_reward=round(self.cumulative_reward, 3),
                components=dict(self.last_components),
            ),
            done=self.done,
            info=info,
        )

    def _return_drivers(self) -> None:
        for warehouse_id, return_rounds in self.driver_returns.items():
            returned = [round_index for round_index in return_rounds if round_index <= self.round_index]
            if returned:
                self.drivers_available[warehouse_id] += len(returned)
                self.driver_returns[warehouse_id] = [round_index for round_index in return_rounds if round_index > self.round_index]

    def _return_depot_trucks(self) -> None:
        returned = [round_index for round_index in self.depot_truck_returns if round_index <= self.round_index]
        if returned:
            self.depot_trucks_available += len(returned)
            self.depot_truck_returns = [round_index for round_index in self.depot_truck_returns if round_index > self.round_index]

    def _resolve_local_warehouse_fulfillment(
        self,
        open_orders: dict[str, OrderTemplate],
        specs_by_id: dict[str, object],
        components: dict[str, float],
        events: list[str],
    ) -> set[str]:
        self.last_fulfilled_units = {}
        self.last_missed_units = {}
        fulfilled_orders: set[str] = set()
        for warehouse_id, spec in specs_by_id.items():
            local_orders = [
                order for order in open_orders.values()
                if order.region == spec.region and order.order_id not in self.completed_orders
            ]
            for order in sorted(local_orders, key=lambda item: (-item.priority, item.deadline_round, -item.units * item.value_per_unit)):
                key = (warehouse_id, order.sku)
                if self.inventory[warehouse_id].get(order.sku, 0) >= order.units and self.drivers_available[warehouse_id] > 0:
                    delivery_time = spec.delivery_time_by_region[order.region]
                    delivery_cost = spec.delivery_cost_by_region[order.region] * order.units
                    self.inventory[warehouse_id][order.sku] -= order.units
                    self.drivers_available[warehouse_id] -= 1
                    self.driver_returns[warehouse_id].append(self.round_index + delivery_time)
                    self.completed_orders.add(order.order_id)
                    fulfilled_orders.add(order.order_id)
                    value = order.units * order.value_per_unit
                    lateness = max(0, self.round_index + delivery_time - order.deadline_round)
                    late_penalty = lateness * order.value_per_unit * 0.45
                    add_component(components, "fulfilled_order_value", value)
                    add_component(components, "delivery_cost", -delivery_cost)
                    if late_penalty:
                        add_component(components, "late_delivery_penalty", -late_penalty)
                    self.local_utility[warehouse_id] += value - delivery_cost - late_penalty
                    self.last_fulfilled_units[key] = self.last_fulfilled_units.get(key, 0) + order.units
                    events.append(f"{warehouse_id} local agent fulfilled {order.units} {order.sku}")
                else:
                    self.last_missed_units[key] = self.last_missed_units.get(key, 0) + order.units
        return fulfilled_orders

    def _require_recipe(self) -> HiddenRecipe:
        if self.recipe is None:
            self.reset(self.default_task_id, 101)
        assert self.recipe is not None
        return self.recipe


def _trace(actor: str, proposal_type: str, status: str, summary: str, delta: float) -> NegotiationEvent:
    return NegotiationEvent(
        actor=actor,
        proposal_type=proposal_type,
        status=status,  # type: ignore[arg-type]
        summary=summary,
        local_utility_delta=round(delta, 3),
    )


def _pressure_summary_from_reports(reports: list[DemandReport]) -> str:
    if not reports:
        return "No aggregated warehouse demand pressure is visible."
    urgent = [report for report in reports if report.urgency >= 3]
    if urgent:
        return f"{len(urgent)} urgent warehouse demand report(s) need allocation."
    highest = max(reports, key=lambda report: (report.requested_units + report.at_risk_units, report.urgency))
    return f"Highest visible pressure: {highest.warehouse_id} needs {highest.requested_units} {highest.sku}."


def _depot_message(inventory: dict[str, int], trucks_available: int) -> str:
    low = [sku for sku, units in inventory.items() if units <= 2]
    if trucks_available <= 0:
        return "Central depot trucks are currently committed."
    if low:
        return f"Central depot can replenish, but {', '.join(low)} stock is tight."
    return "Central depot has limited emergency replenishment capacity."


def _matches_priority_policy(order: OrderTemplate, priority_policy: list) -> bool:
    for rule in priority_policy:
        sku_match = rule.sku is None or rule.sku == order.sku
        region_match = rule.region is None or rule.region == order.region
        if sku_match and region_match and rule.priority <= order.priority:
            return True
    return False


def _demand_reports(
    recipe: HiddenRecipe,
    open_orders: list[OrderTemplate],
    inventory: dict[str, dict[str, int]],
    last_fulfilled_units: dict[tuple[str, str], int],
    last_missed_units: dict[tuple[str, str], int],
    round_index: int,
) -> list[DemandReport]:
    specs_by_region = {spec.region: spec for spec in recipe.warehouse_specs}
    units_by_key: dict[tuple[str, str], int] = {}
    at_risk_by_key: dict[tuple[str, str], int] = {}
    value_by_key: dict[tuple[str, str], float] = {}
    for order in open_orders:
        spec = specs_by_region.get(order.region)
        if spec is None:
            continue
        key = (spec.warehouse_id, order.sku)
        units_by_key[key] = units_by_key.get(key, 0) + order.units
        if order.deadline_round <= round_index + 1:
            at_risk_by_key[key] = at_risk_by_key.get(key, 0) + order.units
        value_by_key[key] = value_by_key.get(key, 0.0) + order.units * order.value_per_unit

    reports: list[DemandReport] = []
    for spec in recipe.warehouse_specs:
        forecast = recipe.private_forecasts[spec.warehouse_id]
        for sku, forecast_units in forecast.items():
            key = (spec.warehouse_id, sku)
            requested = units_by_key.get(key, 0)
            available = inventory[spec.warehouse_id].get(sku, 0)
            missed = last_missed_units.get(key, 0)
            at_risk = at_risk_by_key.get(key, 0)
            pressure = max(0, requested + forecast_units + missed - available)
            if pressure <= 0 and missed <= 0 and at_risk <= 0:
                continue
            urgency = 1
            if at_risk > 0 or missed > 0:
                urgency = 3
            elif pressure >= 3:
                urgency = 2
            reports.append(
                DemandReport(
                    warehouse_id=spec.warehouse_id,
                    region=spec.region,
                    sku=sku,
                    requested_units=requested,
                    fulfilled_units_last_round=last_fulfilled_units.get(key, 0),
                    missed_units_last_round=missed,
                    at_risk_units=at_risk,
                    forecast_units=forecast_units,
                    urgency=urgency,
                    message=(
                        f"{spec.label} reports {requested} visible units, {forecast_units} forecast units, "
                        f"{missed} missed last round for {sku}."
                    ),
                )
            )
    return reports
