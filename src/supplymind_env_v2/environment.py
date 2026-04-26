from __future__ import annotations

import secrets
from copy import deepcopy
from math import ceil
from statistics import mean
from typing import Any

from .config import cfg, load_reward_config, price_band
from .generator import generate_recipe, sample_seed, to_internal_task_id, to_public_task_id
from .models import (
    CenterObservation,
    HiddenRecipe,
    LocalOrderSnapshot,
    OrderTemplate,
    PendingTransferProposal,
    V2JointAction,
    V2Observation,
    V2Reward,
    V2StepResult,
    WarehouseObservation,
)
from .rules import compact_public_rules


class V2SupplyMindEnv:
    def __init__(self, default_task_id: str = "scarcity_market") -> None:
        self.default_task_id = default_task_id
        self.recipe: HiddenRecipe | None = None
        self.task_id = default_task_id
        self.internal_task_id = to_internal_task_id(default_task_id)
        self.public_seed = 0
        self.internal_seed = 0
        self.seed_source = "provided"
        self.round_index = 0
        self.inventory: dict[str, dict[str, int]] = {}
        self.inventory_age: dict[str, dict[str, float]] = {}
        self.center_sourced_inventory: dict[str, dict[str, int]] = {}
        self.center_margin_per_unit: dict[str, dict[str, float]] = {}
        self.central_depot_inventory: dict[str, int] = {}
        self.central_depot_inventory_age: dict[str, float] = {}
        self.depot_procurements: list[tuple[int, str, int]] = []
        self.depot_trucks_available = 0
        self.depot_truck_returns: list[int] = []
        self.drivers_available: dict[str, int] = {}
        self.driver_returns: dict[str, list[int]] = {}
        self.order_status: dict[str, str] = {}
        self.pending_transfers: dict[str, PendingTransferProposal] = {}
        self.next_proposal_id = 1
        self.trust: dict[str, float] = {}
        self.agent_rewards: dict[str, float] = {}
        self.cumulative_reward = 0.0
        self.audit_metrics: dict[str, Any] = {}
        self.last_step_reward = 0.0
        self.last_components: dict[str, float] = {}
        self.last_events: list[str] = []
        self.last_invalid_action_details: list[dict[str, Any]] = []
        self.done = False
        self.last_episode_summary: dict[str, Any] | None = None

    def reset(self, task_id: str | None = None, seed: int | None = None) -> V2Observation:
        self.internal_task_id = to_internal_task_id(task_id or self.default_task_id)
        self.seed_source = "provided" if seed is not None else "curated_pool"
        self.public_seed = seed if seed is not None else sample_seed(self.internal_task_id, secrets.randbelow(1_000_000))
        self.task_id = to_public_task_id(self.internal_task_id)
        self.internal_seed = self.public_seed
        self._load_recipe(generate_recipe(self.internal_task_id, self.internal_seed))
        return self.state()

    def reset_internal(self, task_id: str, seed: int) -> V2Observation:
        self.public_seed = seed
        self.internal_seed = seed
        self.seed_source = "provided"
        self.internal_task_id = to_internal_task_id(task_id)
        self.task_id = to_public_task_id(self.internal_task_id)
        self._load_recipe(generate_recipe(self.internal_task_id, self.internal_seed))
        return self.state()

    def _load_recipe(self, recipe: HiddenRecipe) -> None:
        self.recipe = recipe
        self.round_index = 0
        self.inventory = deepcopy(recipe.initial_inventory)
        self.inventory_age = {
            warehouse_id: {sku: 0.0 for sku in inventory}
            for warehouse_id, inventory in self.inventory.items()
        }
        self.center_sourced_inventory = {
            warehouse_id: {sku: 0 for sku in inventory}
            for warehouse_id, inventory in self.inventory.items()
        }
        self.center_margin_per_unit = {
            warehouse_id: {sku: 0.0 for sku in inventory}
            for warehouse_id, inventory in self.inventory.items()
        }
        self.central_depot_inventory = dict(recipe.central_depot_inventory)
        self.central_depot_inventory_age = {sku: 0.0 for sku in self.central_depot_inventory}
        self.depot_procurements = []
        self.depot_trucks_available = recipe.profile.depot_trucks
        self.depot_truck_returns = []
        self.drivers_available = dict(recipe.initial_drivers)
        self.driver_returns = {spec.warehouse_id: [] for spec in recipe.warehouse_specs}
        self.order_status = {order.order_id: "pending" for order in recipe.orders}
        self.pending_transfers = {}
        self.next_proposal_id = 1
        self.trust = {spec.warehouse_id: 0.75 for spec in recipe.warehouse_specs}
        self.agent_rewards = {"center": 0.0, **{spec.warehouse_id: 0.0 for spec in recipe.warehouse_specs}}
        self.cumulative_reward = 0.0
        self.audit_metrics = self._empty_audit_metrics()
        self.last_step_reward = 0.0
        self.last_components = {}
        self.last_events = ["v2 environment reset"]
        self.last_invalid_action_details = []
        self.done = False
        self.last_episode_summary = None

    def state(self) -> V2Observation:
        recipe = self._require_recipe()
        specs_by_id = {spec.warehouse_id: spec for spec in recipe.warehouse_specs}
        visible = self._visible_orders()
        warehouses = {
            spec.warehouse_id: WarehouseObservation(
                warehouse_id=spec.warehouse_id,
                label=spec.label,
                region=spec.region,
                inventory=dict(self.inventory[spec.warehouse_id]),
                inventory_age={key: round(value, 2) for key, value in self.inventory_age[spec.warehouse_id].items()},
                drivers_available=self.drivers_available[spec.warehouse_id],
                local_orders=[
                    LocalOrderSnapshot(
                        order_id=order.order_id,
                        sku=order.sku,
                        units=order.units,
                        customer_value_per_unit=order.customer_value_per_unit,
                        created_round=order.created_round,
                        deadline_round=order.deadline_round,
                        status=self.order_status[order.order_id],  # type: ignore[arg-type]
                    )
                    for order in visible
                    if order.warehouse_id == spec.warehouse_id and self.order_status[order.order_id] in {"pending", "accepted"}
                ],
                order_action_context=[
                    self._order_action_context(order, spec.warehouse_id)
                    for order in visible
                    if order.warehouse_id == spec.warehouse_id and self.order_status[order.order_id] in {"pending", "accepted"}
                ],
                pending_transfer_proposals=[
                    proposal for proposal in self.pending_transfers.values()
                    if proposal.from_warehouse == spec.warehouse_id
                ],
                safety_stock=dict(spec.safety_stock),
                route_costs=dict(spec.route_costs),
                last_reward=round(self.agent_rewards[spec.warehouse_id], 3),
            )
            for spec in recipe.warehouse_specs
        }
        return V2Observation(
            round_index=self.round_index,
            remaining_rounds=max(0, recipe.profile.total_rounds - self.round_index),
            task_id=self.task_id,
            scenario_info={
                "task_id": self.task_id,
                "seed": self.public_seed,
                "seed_source": self.seed_source,
                "warehouse_count": recipe.profile.warehouse_count,
                "total_rounds": recipe.profile.total_rounds,
                "depot_replenishment_cap": recipe.profile.depot_replenishment_cap,
                "depot_procurement_cap": recipe.profile.depot_procurement_cap,
                "depot_procurement_lead_time": recipe.profile.depot_procurement_lead_time,
                "transfer_cap": recipe.profile.transfer_cap,
                "public_forecasts": [
                    forecast for forecast in recipe.public_forecasts
                    if forecast["available_round"] <= self.round_index
                ],
                "public_rules": compact_public_rules(),
            },
            center=CenterObservation(
                round_index=self.round_index,
                remaining_rounds=max(0, recipe.profile.total_rounds - self.round_index),
                depot_inventory=dict(self.central_depot_inventory),
                depot_trucks_available=self.depot_trucks_available,
                inbound_procurements=[
                    {"arrival_round": round_index, "sku": sku, "units": units}
                    for round_index, sku, units in sorted(self.depot_procurements)
                ],
                warehouse_summaries=[
                    {
                        "warehouse_id": spec.warehouse_id,
                        "region": spec.region,
                        "inventory": dict(self.inventory[spec.warehouse_id]),
                        "drivers_available": self.drivers_available[spec.warehouse_id],
                        "trust": round(self.trust[spec.warehouse_id], 3),
                        "pending_orders": sum(1 for order in visible if order.warehouse_id == spec.warehouse_id and self.order_status[order.order_id] == "pending"),
                        "accepted_orders": sum(1 for order in visible if order.warehouse_id == spec.warehouse_id and self.order_status[order.order_id] == "accepted"),
                    }
                    for spec in recipe.warehouse_specs
                ],
                market_signals=[],
                pending_transfer_proposals=list(self.pending_transfers.values()),
                price_bands=load_reward_config()["price_bands"],
                depot_inventory_age={key: round(value, 2) for key, value in self.central_depot_inventory_age.items()},
            ),
            warehouses=warehouses,
            feedback={
                "last_step_reward": round(self.last_step_reward, 3),
                "cumulative_reward": round(self.cumulative_reward, 3),
                "invalid_action_details": list(self.last_invalid_action_details),
                "events": list(self.last_events),
                "episode_summary": None if self.last_episode_summary is None else dict(self.last_episode_summary),
            },
        )

    def step(self, action: V2JointAction, grade_terminal: bool = True) -> V2StepResult:
        if self.done:
            return V2StepResult(
                observation=self.state(),
                reward=V2Reward(step_reward=0.0, cumulative_reward=self.cumulative_reward, components={}),
                done=True,
                info={"episode_summary": self.last_episode_summary or {}, "agent_rewards": dict(self.agent_rewards)},
            )

        recipe = self._require_recipe()
        self._receive_returns()
        components: dict[str, float] = {}
        agent_delta = {key: 0.0 for key in self.agent_rewards}
        events: list[str] = []
        self._current_invalid_action_details: list[dict[str, Any]] = []

        self._resolve_transfer_responses(action, components, agent_delta, events)
        market_signals = self._collect_warehouse_market_actions(action, components, agent_delta, events)
        self._resolve_order_decisions(action, components, agent_delta, events)
        self._resolve_center_procurements(action, components, agent_delta, events)
        self._resolve_center_liquidations(action, components, agent_delta, events)
        self._resolve_center_replenishments(action, components, agent_delta, events)
        self._resolve_offer_matches(action, market_signals, components, agent_delta, events)
        self._queue_transfer_proposals(action, components, agent_delta, events)
        self._fulfill_orders(action, components, agent_delta, events)
        self._expire_orders(components, agent_delta, events)
        self._apply_holding_and_spoilage(components, agent_delta, events)

        will_done = self.round_index + 1 >= recipe.profile.total_rounds or all(
            status in {"fulfilled", "rejected", "expired"} for status in self.order_status.values()
        )
        if will_done:
            self._apply_terminal_order_cleanup(components, agent_delta, events)
            self._apply_terminal_inventory_penalty(components, agent_delta, events)
            self._apply_terminal_fairness_penalty(components, events)

        step_reward = sum(value for key, value in components.items() if key.startswith("global_"))
        self._update_audit_metrics(components)
        self.cumulative_reward += step_reward
        self.last_step_reward = step_reward
        self.last_components = {key: round(value, 3) for key, value in components.items() if abs(value) > 1e-9}
        self.last_events = events or ["no material event this round"]
        self.last_invalid_action_details = list(getattr(self, "_current_invalid_action_details", []))
        for key, value in agent_delta.items():
            self.agent_rewards[key] += value
        self.round_index += 1
        self.done = will_done

        info: dict[str, Any] = {
            "agent_rewards": {key: round(value, 3) for key, value in self.agent_rewards.items()},
            "market_signals": market_signals,
        }
        if self.done and grade_terminal:
            from .grading import grade_episode

            avg_warehouse_reward = mean(value for key, value in self.agent_rewards.items() if key != "center")
            task_result = grade_episode(
                self.internal_task_id,
                self.internal_seed,
                self.cumulative_reward,
                self.agent_rewards["center"],
                avg_warehouse_reward,
            )
            self.last_episode_summary = {
                "raw_reward": round(task_result.raw_reward, 3),
                "baseline_reward": round(task_result.baseline_reward, 3),
                "target_reward": round(task_result.target_reward, 3),
                "graded_score": round(task_result.score, 4),
                "center_reward": round(self.agent_rewards["center"], 3),
                "average_warehouse_reward": round(avg_warehouse_reward, 3),
                "center_role_score": round(task_result.center_role_score, 4),
                "warehouse_role_score": round(task_result.warehouse_role_score, 4),
                "baseline_center_reward": round(task_result.baseline_center_reward, 3),
                "target_center_reward": round(task_result.target_center_reward, 3),
                "baseline_warehouse_reward": round(task_result.baseline_warehouse_reward, 3),
                "target_warehouse_reward": round(task_result.target_warehouse_reward, 3),
                "audit_metrics": dict(self.audit_metrics),
            }
            info["episode_summary"] = dict(self.last_episode_summary)

        return V2StepResult(
            observation=self.state(),
            reward=V2Reward(step_reward=round(step_reward, 3), cumulative_reward=round(self.cumulative_reward, 3), components={}),
            done=self.done,
            info=info,
        )

    def _resolve_transfer_responses(self, action: V2JointAction, components: dict[str, float], agent_delta: dict[str, float], events: list[str]) -> None:
        recipe = self._require_recipe()
        specs_by_id = {spec.warehouse_id: spec for spec in recipe.warehouse_specs}
        for warehouse_id, warehouse_action in action.warehouse_actions.items():
            for response in warehouse_action.transfer_responses:
                proposal = self.pending_transfers.pop(response.proposal_id, None)
                if proposal is None or proposal.from_warehouse != warehouse_id:
                    self._invalid(components, "transfer_response", "proposal_missing_or_wrong_source", -3.0, {"warehouse_id": warehouse_id, "proposal_id": response.proposal_id})
                    continue
                if response.decision == "reject":
                    _add(components, "global_rejected_transfer_penalty", -0.5)
                    _add(components, "audit_rejected_transfer_count", 1.0)
                    agent_delta["center"] -= cfg("center_rewards", "rejected_transfer_penalty")
                    self.trust[warehouse_id] = max(0.0, self.trust[warehouse_id] - 0.04)
                    events.append(f"{warehouse_id} rejected transfer {proposal.proposal_id}")
                    continue
                if self.inventory[proposal.from_warehouse].get(proposal.sku, 0) < proposal.units:
                    self._invalid(components, "transfer_response", "source_insufficient_stock", -4.0, {"warehouse_id": warehouse_id, "proposal_id": response.proposal_id, "sku": proposal.sku})
                    continue
                source = specs_by_id[proposal.from_warehouse]
                target = specs_by_id[proposal.to_warehouse]
                transfer_cost = source.route_costs[target.region] * proposal.units * 0.8
                moved_age = self.inventory_age[proposal.from_warehouse].get(proposal.sku, 0.0)
                self._remove_warehouse_stock(proposal.from_warehouse, proposal.sku, proposal.units)
                self._add_warehouse_stock(proposal.to_warehouse, proposal.sku, proposal.units, moved_age)
                broker_fee = _broker_fee(proposal.units, proposal.compensation)
                _add(components, "global_transfer_cost", -transfer_cost)
                _add(components, "center_transfer_broker_fee", broker_fee)
                _add(components, "audit_transfer_count", 1.0)
                _add(components, "audit_successful_transfer_units", float(proposal.units))
                agent_delta["center"] += broker_fee
                agent_delta[proposal.from_warehouse] += proposal.compensation - proposal.units * cfg("warehouse_rewards", "transfer_sacrifice_cost_per_unit")
                agent_delta[proposal.to_warehouse] -= proposal.compensation
                events.append(f"{proposal.from_warehouse} accepted transfer {proposal.units} {proposal.sku} to {proposal.to_warehouse}")

    def _collect_warehouse_market_actions(self, action: V2JointAction, components: dict[str, float], agent_delta: dict[str, float], events: list[str]) -> list[dict[str, Any]]:
        signals: list[dict[str, Any]] = []
        for warehouse_id, warehouse_action in action.warehouse_actions.items():
            if warehouse_id not in self.inventory:
                self._invalid(components, "warehouse_action", "unknown_warehouse", -3.0, {"warehouse_id": warehouse_id})
                continue
            for offer in warehouse_action.inventory_offers:
                units = min(max(0, offer.units), self.inventory[warehouse_id].get(offer.sku, 0))
                if units > 0:
                    signals.append({"signal_id": f"{warehouse_id}:offer:{offer.sku}", "warehouse_id": warehouse_id, "type": "offer", "sku": offer.sku, "units": units, "price": offer.ask_price})
            for request in warehouse_action.inventory_requests:
                if request.units > 0:
                    signals.append({"signal_id": f"{warehouse_id}:request:{request.sku}", "warehouse_id": warehouse_id, "type": "request", "sku": request.sku, "units": request.units, "price": request.max_price})
        return signals

    def _resolve_order_decisions(self, action: V2JointAction, components: dict[str, float], agent_delta: dict[str, float], events: list[str]) -> None:
        orders = {order.order_id: order for order in self._visible_orders()}
        for warehouse_id, warehouse_action in action.warehouse_actions.items():
            for decision in warehouse_action.order_decisions:
                order = orders.get(decision.order_id)
                if order is None or order.warehouse_id != warehouse_id or self.order_status.get(order.order_id) != "pending":
                    self._invalid(components, "order_decision", "order_missing_wrong_warehouse_or_not_pending", -2.0, {"warehouse_id": warehouse_id, "order_id": decision.order_id})
                    continue
                if decision.decision == "reject":
                    self.order_status[order.order_id] = "rejected"
                    penalty = order.units * order.customer_value_per_unit * cfg("order_rewards", "reject_penalty_multiplier")
                    _add(components, "global_reject_penalty", -penalty)
                    agent_delta[warehouse_id] -= penalty
                    events.append(f"{warehouse_id} rejected {order.order_id}")
                else:
                    self.order_status[order.order_id] = "accepted"
                    events.append(f"{warehouse_id} accepted {order.order_id}")

    def _resolve_center_procurements(self, action: V2JointAction, components: dict[str, float], agent_delta: dict[str, float], events: list[str]) -> None:
        recipe = self._require_recipe()
        used_units = 0
        for procurement in action.central_action.central_procurements:
            band = price_band(procurement.sku)
            if procurement.units <= 0 or procurement.max_unit_cost < band["procurement_cost"] or used_units + procurement.units > recipe.profile.depot_procurement_cap:
                reason = "invalid_units" if procurement.units <= 0 else "max_unit_cost_below_procurement" if procurement.max_unit_cost < band["procurement_cost"] else "procurement_cap_exceeded"
                self._invalid(components, "central_procurement", reason, -4.0, {"sku": procurement.sku, "units": procurement.units})
                agent_delta["center"] -= 2.0
                continue
            cost = procurement.units * band["procurement_cost"]
            arrival = self.round_index + recipe.profile.depot_procurement_lead_time
            self.depot_procurements.append((arrival, procurement.sku, procurement.units))
            used_units += procurement.units
            _add(components, "global_procurement_cost", -cost)
            agent_delta["center"] -= cost
            events.append(f"center procured {procurement.units} {procurement.sku}, arrival round {arrival}")

    def _resolve_center_liquidations(self, action: V2JointAction, components: dict[str, float], agent_delta: dict[str, float], events: list[str]) -> None:
        used_units = 0
        cap = int(cfg("center_rewards", "liquidation_cap_per_round"))
        recovery_multiplier = cfg("center_rewards", "liquidation_recovery_multiplier")
        for liquidation in action.central_action.central_liquidations:
            if liquidation.units <= 0 or used_units + liquidation.units > cap:
                reason = "invalid_units" if liquidation.units <= 0 else "liquidation_cap_exceeded"
                self._invalid(components, "central_liquidation", reason, -3.0, {"sku": liquidation.sku, "units": liquidation.units})
                agent_delta["center"] -= 1.0
                continue
            available = self.central_depot_inventory.get(liquidation.sku, 0)
            if liquidation.units > available:
                self._invalid(components, "central_liquidation", "insufficient_depot_stock", -3.0, {"sku": liquidation.sku, "units": liquidation.units, "available": available})
                agent_delta["center"] -= 1.0
                continue
            recovery = liquidation.units * price_band(liquidation.sku)["procurement_cost"] * recovery_multiplier
            self._remove_depot_stock(liquidation.sku, liquidation.units)
            used_units += liquidation.units
            _add(components, "global_liquidation_recovery", recovery)
            _add(components, "center_liquidation_recovery", recovery)
            agent_delta["center"] += recovery
            events.append(f"center liquidated {liquidation.units} {liquidation.sku} at markdown recovery")

    def _resolve_center_replenishments(self, action: V2JointAction, components: dict[str, float], agent_delta: dict[str, float], events: list[str]) -> None:
        recipe = self._require_recipe()
        specs_by_id = {spec.warehouse_id: spec for spec in recipe.warehouse_specs}
        used_units = 0
        for shipment in action.central_action.central_replenishments:
            band = price_band(shipment.sku)
            if (
                shipment.to_warehouse not in self.inventory
                or shipment.units <= 0
                or shipment.unit_price > band["max_wholesale_price"]
                or shipment.units > self.central_depot_inventory.get(shipment.sku, 0)
                or used_units + shipment.units > recipe.profile.depot_replenishment_cap
                or self.depot_trucks_available <= 0
            ):
                reason = self._shipment_invalid_reason(shipment, used_units, recipe.profile.depot_replenishment_cap, band)
                self._invalid(components, "central_replenishment", reason, -4.0, {"to_warehouse": shipment.to_warehouse, "sku": shipment.sku, "units": shipment.units})
                agent_delta["center"] -= 2.0
                continue
            spec = specs_by_id[shipment.to_warehouse]
            shipment_cost = (1.2 + spec.route_costs[spec.region]) * shipment.units
            moved_age = self.central_depot_inventory_age.get(shipment.sku, 0.0)
            self._remove_depot_stock(shipment.sku, shipment.units)
            self._add_warehouse_stock(shipment.to_warehouse, shipment.sku, shipment.units, moved_age)
            self.center_sourced_inventory[shipment.to_warehouse][shipment.sku] += shipment.units
            self.center_margin_per_unit[shipment.to_warehouse][shipment.sku] = shipment.unit_price - band["procurement_cost"]
            self.depot_trucks_available -= 1
            self.depot_truck_returns.append(self.round_index + spec.route_times[spec.region])
            used_units += shipment.units
            wholesale_cost = shipment.unit_price * shipment.units
            _add(components, "global_center_shipment_cost", -shipment_cost)
            agent_delta["center"] -= shipment_cost
            agent_delta[shipment.to_warehouse] -= wholesale_cost
            events.append(f"center sold {shipment.units} {shipment.sku} to {shipment.to_warehouse} at {shipment.unit_price:.1f}")

    def _resolve_offer_matches(self, action: V2JointAction, market_signals: list[dict[str, Any]], components: dict[str, float], agent_delta: dict[str, float], events: list[str]) -> None:
        signals = {signal["signal_id"]: signal for signal in market_signals}
        moved = 0
        for match in action.central_action.offer_matches:
            offer = signals.get(match.offer_signal_id)
            request = signals.get(match.request_signal_id)
            if not offer or not request or offer["type"] != "offer" or request["type"] != "request" or offer["sku"] != request["sku"]:
                self._invalid(components, "offer_match", "missing_or_incompatible_offer_request", -3.0, {"offer_signal_id": match.offer_signal_id, "request_signal_id": match.request_signal_id})
                continue
            units = min(match.units, offer["units"], request["units"])
            if units <= 0 or self.inventory[offer["warehouse_id"]].get(offer["sku"], 0) < units:
                self._invalid(components, "offer_match", "invalid_units_or_offer_stock", -3.0, {"offer_signal_id": match.offer_signal_id, "request_signal_id": match.request_signal_id, "units": match.units})
                continue
            moved_age = self.inventory_age[offer["warehouse_id"]].get(offer["sku"], 0.0)
            self._remove_warehouse_stock(offer["warehouse_id"], offer["sku"], units)
            self._add_warehouse_stock(request["warehouse_id"], request["sku"], units, moved_age)
            compensation = max(match.compensation, offer["price"] * units)
            broker_fee = _broker_fee(units, compensation)
            _add(components, "global_transfer_cost", -0.5 * units)
            _add(components, "center_transfer_broker_fee", broker_fee)
            _add(components, "audit_transfer_count", 1.0)
            _add(components, "audit_successful_transfer_units", float(units))
            agent_delta["center"] += broker_fee
            agent_delta[offer["warehouse_id"]] += compensation
            agent_delta[request["warehouse_id"]] -= compensation
            moved += units
            events.append(f"center matched {units} {offer['sku']} from {offer['warehouse_id']} to {request['warehouse_id']}")
        if moved > self._require_recipe().profile.transfer_cap:
            self._invalid(components, "offer_match", "transfer_cap_exceeded", -4.0, {"moved_units": moved})

    def _queue_transfer_proposals(self, action: V2JointAction, components: dict[str, float], agent_delta: dict[str, float], events: list[str]) -> None:
        for proposal in action.central_action.inventory_transfer_proposals:
            if proposal.from_warehouse not in self.inventory or proposal.to_warehouse not in self.inventory or proposal.units <= 0:
                self._invalid(components, "transfer_proposal", "unknown_warehouse_or_invalid_units", -3.0, {"from_warehouse": proposal.from_warehouse, "to_warehouse": proposal.to_warehouse, "units": proposal.units})
                continue
            proposal_id = f"p{self.next_proposal_id}"
            self.next_proposal_id += 1
            self.pending_transfers[proposal_id] = PendingTransferProposal(proposal_id=proposal_id, **proposal.model_dump())
            events.append(f"center proposed transfer {proposal_id}: {proposal.from_warehouse}->{proposal.to_warehouse} {proposal.units} {proposal.sku}")

    def _fulfill_orders(self, action: V2JointAction, components: dict[str, float], agent_delta: dict[str, float], events: list[str]) -> None:
        recipe = self._require_recipe()
        specs_by_id = {spec.warehouse_id: spec for spec in recipe.warehouse_specs}
        priority_by_wh: dict[str, dict[str, int]] = {
            warehouse_id: {rule.sku: rule.priority for rule in warehouse_action.local_priority}
            for warehouse_id, warehouse_action in action.warehouse_actions.items()
        }
        accepted = [
            order for order in self._visible_orders()
            if self.order_status[order.order_id] == "accepted"
        ]
        for order in sorted(accepted, key=lambda item: (-priority_by_wh.get(item.warehouse_id, {}).get(item.sku, 1), item.deadline_round, -item.units * item.customer_value_per_unit)):
            spec = specs_by_id[order.warehouse_id]
            if self.inventory[order.warehouse_id].get(order.sku, 0) < order.units or self.drivers_available[order.warehouse_id] <= 0:
                continue
            delivery_cost = spec.route_costs[spec.region] * order.units
            self._remove_warehouse_stock(order.warehouse_id, order.sku, order.units)
            self.drivers_available[order.warehouse_id] -= 1
            self.driver_returns[order.warehouse_id].append(self.round_index + spec.route_times[spec.region])
            self.order_status[order.order_id] = "fulfilled"
            value = order.units * order.customer_value_per_unit
            _add(components, "global_fulfilled_customer_value", value)
            _add(components, "global_warehouse_delivery_cost", -delivery_cost)
            fulfillment_profit = max(0.0, value - delivery_cost)
            center_service_bonus = fulfillment_profit * cfg("center_rewards", "network_service_profit_share")
            agent_delta[order.warehouse_id] += value - delivery_cost - center_service_bonus
            agent_delta["center"] += center_service_bonus
            _add(components, "center_network_service_bonus", center_service_bonus)
            center_units = min(order.units, self.center_sourced_inventory[order.warehouse_id].get(order.sku, 0))
            if center_units:
                margin = center_units * self.center_margin_per_unit[order.warehouse_id].get(order.sku, 0.0)
                self.center_sourced_inventory[order.warehouse_id][order.sku] -= center_units
                agent_delta["center"] += margin
                _add(components, "center_realized_wholesale_margin", margin)
            events.append(f"{order.warehouse_id} fulfilled {order.order_id}")

    def _expire_orders(self, components: dict[str, float], agent_delta: dict[str, float], events: list[str]) -> None:
        for order in self._visible_orders():
            if order.deadline_round >= self.round_index:
                continue
            status = self.order_status[order.order_id]
            if status not in {"pending", "accepted"}:
                continue
            multiplier = cfg("order_rewards", "accepted_missed_penalty_multiplier") if status == "accepted" else cfg("order_rewards", "silent_expiry_penalty_multiplier")
            penalty = order.units * order.customer_value_per_unit * multiplier
            self.order_status[order.order_id] = "expired"
            _add(components, "global_stockout_penalty", -penalty)
            agent_delta[order.warehouse_id] -= penalty * cfg("warehouse_rewards", "local_stockout_share")
            agent_delta["center"] -= penalty * cfg("center_rewards", "network_stockout_share")
            events.append(f"{order.order_id} expired after {status}")

    def _apply_terminal_order_cleanup(self, components: dict[str, float], agent_delta: dict[str, float], events: list[str]) -> None:
        accepted_count = 0
        pending_count = 0
        total_penalty = 0.0
        for order in self._visible_orders():
            status = self.order_status[order.order_id]
            if status not in {"pending", "accepted"}:
                continue
            multiplier = (
                cfg("order_rewards", "accepted_missed_penalty_multiplier")
                if status == "accepted"
                else cfg("global_rewards", "terminal_pending_expiry_multiplier")
            )
            penalty = order.units * order.customer_value_per_unit * multiplier
            self.order_status[order.order_id] = "expired"
            _add(components, "global_terminal_order_penalty", -penalty)
            agent_delta[order.warehouse_id] -= penalty * cfg("warehouse_rewards", "local_stockout_share")
            agent_delta["center"] -= penalty * cfg("center_rewards", "network_stockout_share")
            total_penalty += penalty
            if status == "accepted":
                accepted_count += 1
            else:
                pending_count += 1
        if total_penalty:
            events.append(f"terminal closed {accepted_count} accepted and {pending_count} pending visible orders")

    def _apply_holding_and_spoilage(self, components: dict[str, float], agent_delta: dict[str, float], events: list[str]) -> None:
        center_holding = sum(self.central_depot_inventory.values()) * cfg("center_rewards", "depot_holding_cost_per_unit")
        _add(components, "global_holding_cost", -center_holding)
        agent_delta["center"] -= center_holding
        shelf_life = int(cfg("global_rewards", "fresh_milk_shelf_life_rounds"))
        expiry_fraction = cfg("global_rewards", "fresh_milk_expiry_fraction_per_round")
        disposal_multiplier = cfg("global_rewards", "fresh_milk_disposal_cost_multiplier")
        self._age_fresh_milk()
        for sku, units in list(self.central_depot_inventory.items()):
            if sku == "fresh_milk" and units > 0 and self.central_depot_inventory_age.get(sku, 0.0) > shelf_life:
                spoiled = min(units, max(1, ceil(units * expiry_fraction)))
                penalty = spoiled * price_band(sku)["procurement_cost"] * disposal_multiplier
                self._remove_depot_stock(sku, spoiled)
                _add(components, "global_spoilage_cost", -penalty)
                agent_delta["center"] -= penalty * cfg("center_rewards", "depot_spoilage_share")
                events.append(f"depot spoiled {spoiled} {sku}")
        for warehouse_id, inventory in self.inventory.items():
            holding = sum(inventory.values()) * cfg("warehouse_rewards", "holding_cost_per_unit")
            _add(components, "global_holding_cost", -holding)
            agent_delta[warehouse_id] -= holding
            fresh = inventory.get("fresh_milk", 0)
            if fresh > 0 and self.inventory_age[warehouse_id].get("fresh_milk", 0.0) > shelf_life:
                spoiled = min(fresh, max(1, ceil(fresh * expiry_fraction)))
                penalty = spoiled * price_band("fresh_milk")["procurement_cost"] * disposal_multiplier
                self._remove_warehouse_stock(warehouse_id, "fresh_milk", spoiled)
                _add(components, "global_spoilage_cost", -penalty)
                agent_delta[warehouse_id] -= penalty * cfg("warehouse_rewards", "warehouse_spoilage_share")
                agent_delta["center"] -= penalty * cfg("center_rewards", "warehouse_spoilage_share")
                events.append(f"{warehouse_id} spoiled {spoiled} fresh_milk")

    def _apply_terminal_inventory_penalty(self, components: dict[str, float], agent_delta: dict[str, float], events: list[str]) -> None:
        multiplier = cfg("global_rewards", "terminal_inventory_penalty_multiplier")
        depot_value = sum(units * price_band(sku)["procurement_cost"] for sku, units in self.central_depot_inventory.items())
        depot_penalty = depot_value * multiplier
        if depot_penalty:
            _add(components, "global_terminal_inventory_penalty", -depot_penalty)
            agent_delta["center"] -= depot_penalty * cfg("center_rewards", "terminal_depot_inventory_share")

        warehouse_penalty_total = 0.0
        leftover_units = 0
        for warehouse_id, inventory in self.inventory.items():
            value = sum(units * price_band(sku)["procurement_cost"] for sku, units in inventory.items())
            penalty = value * multiplier
            if not penalty:
                continue
            warehouse_penalty_total += penalty
            leftover_units += sum(inventory.values())
            _add(components, "global_terminal_inventory_penalty", -penalty)
            agent_delta[warehouse_id] -= penalty * cfg("warehouse_rewards", "terminal_inventory_share")
            agent_delta["center"] -= penalty * cfg("center_rewards", "terminal_warehouse_inventory_share")

        if depot_penalty or warehouse_penalty_total:
            depot_units = sum(self.central_depot_inventory.values())
            events.append(f"terminal leftover inventory penalty on {depot_units} depot units and {leftover_units} warehouse units")

    def _apply_terminal_fairness_penalty(self, components: dict[str, float], events: list[str]) -> None:
        service_rates = self._service_rates()
        if not service_rates:
            return
        avg_rate = mean(service_rates.values())
        mad = mean(abs(rate - avg_rate) for rate in service_rates.values())
        penalty = min(cfg("global_rewards", "terminal_fairness_weight") * mad, cfg("global_rewards", "terminal_fairness_cap"))
        if penalty:
            _add(components, "global_fairness_penalty", -penalty)
            _add(components, "audit_fairness_penalty", penalty)
            events.append(f"terminal fairness penalty {penalty:.2f} from service-rate spread")
        self.audit_metrics["per_warehouse_service_rate"] = {key: round(value, 4) for key, value in service_rates.items()}

    def _service_rates(self) -> dict[str, float]:
        recipe = self._require_recipe()
        fulfilled: dict[str, float] = {spec.warehouse_id: 0.0 for spec in recipe.warehouse_specs}
        total: dict[str, float] = {spec.warehouse_id: 0.0 for spec in recipe.warehouse_specs}
        for order in recipe.orders:
            value = order.units * order.customer_value_per_unit
            total[order.warehouse_id] += value
            if self.order_status.get(order.order_id) == "fulfilled":
                fulfilled[order.warehouse_id] += value
        return {
            warehouse_id: (fulfilled[warehouse_id] / value if value else 1.0)
            for warehouse_id, value in total.items()
        }

    def _update_audit_metrics(self, components: dict[str, float]) -> None:
        mapping = {
            "audit_transfer_count": "transfer_count",
            "audit_successful_transfer_units": "successful_transfer_units",
            "global_transfer_cost": "transfer_cost",
            "center_transfer_broker_fee": "center_broker_fee",
            "global_liquidation_recovery": "liquidation_recovery",
            "global_fulfilled_customer_value": "fulfilled_customer_value",
            "global_stockout_penalty": "stockout_penalty",
            "global_terminal_order_penalty": "terminal_order_penalty",
            "global_holding_cost": "holding_cost",
            "global_spoilage_cost": "spoilage_cost",
            "global_terminal_inventory_penalty": "terminal_leftover_penalty",
            "audit_fairness_penalty": "fairness_penalty",
            "global_invalid_action_penalty": "invalid_action_penalty",
            "global_rejected_transfer_penalty": "rejected_transfer_penalty",
            "audit_rejected_transfer_count": "rejected_transfer_count",
        }
        for component, metric in mapping.items():
            value = components.get(component, 0.0)
            if value:
                self.audit_metrics[metric] = round(float(self.audit_metrics.get(metric, 0.0)) + float(value), 3)

    @staticmethod
    def _empty_audit_metrics() -> dict[str, Any]:
        return {
            "transfer_count": 0.0,
            "successful_transfer_units": 0.0,
            "rejected_transfer_count": 0,
            "transfer_cost": 0.0,
            "center_broker_fee": 0.0,
            "liquidation_recovery": 0.0,
            "fulfilled_customer_value": 0.0,
            "stockout_penalty": 0.0,
            "terminal_order_penalty": 0.0,
            "holding_cost": 0.0,
            "spoilage_cost": 0.0,
            "terminal_leftover_penalty": 0.0,
            "fairness_penalty": 0.0,
            "invalid_action_penalty": 0.0,
            "per_warehouse_service_rate": {},
        }

    def _order_action_context(self, order: OrderTemplate, warehouse_id: str) -> dict[str, Any]:
        stock = self.inventory[warehouse_id].get(order.sku, 0)
        value = order.units * order.customer_value_per_unit
        return {
            "order_id": order.order_id,
            "status": self.order_status[order.order_id],
            "sku": order.sku,
            "units": order.units,
            "order_value": value,
            "stock_available": stock,
            "stock_gap": max(0, order.units - stock),
            "driver_available": self.drivers_available[warehouse_id] > 0,
            "rounds_until_deadline": order.deadline_round - self.round_index,
            "can_fulfill_now": stock >= order.units and self.drivers_available[warehouse_id] > 0,
            "reject_penalty": round(value * cfg("order_rewards", "reject_penalty_multiplier"), 3),
            "accepted_missed_penalty": round(value * cfg("order_rewards", "accepted_missed_penalty_multiplier"), 3),
            "silent_expiry_penalty": round(value * cfg("order_rewards", "silent_expiry_penalty_multiplier"), 3),
        }

    def _invalid(self, components: dict[str, float], action_type: str, reason: str, penalty: float, context: dict[str, Any]) -> None:
        _add(components, "global_invalid_action_penalty", penalty)
        if not hasattr(self, "_current_invalid_action_details"):
            self._current_invalid_action_details = []
        self._current_invalid_action_details.append(
            {
                "action_type": action_type,
                "reason": reason,
                "penalty": penalty,
                "context": context,
            }
        )

    def _add_depot_stock(self, sku: str, units: int, age: float = 0.0) -> None:
        current_units = self.central_depot_inventory.get(sku, 0)
        if sku == "fresh_milk" and units > 0:
            current_age = self.central_depot_inventory_age.get(sku, 0.0)
            total_units = current_units + units
            self.central_depot_inventory_age[sku] = ((current_units * current_age) + (units * age)) / total_units
        self.central_depot_inventory[sku] = current_units + units

    def _remove_depot_stock(self, sku: str, units: int) -> None:
        self.central_depot_inventory[sku] = max(0, self.central_depot_inventory.get(sku, 0) - units)
        if self.central_depot_inventory[sku] == 0:
            self.central_depot_inventory_age[sku] = 0.0

    def _add_warehouse_stock(self, warehouse_id: str, sku: str, units: int, age: float = 0.0) -> None:
        current_units = self.inventory[warehouse_id].get(sku, 0)
        if sku == "fresh_milk" and units > 0:
            current_age = self.inventory_age[warehouse_id].get(sku, 0.0)
            total_units = current_units + units
            self.inventory_age[warehouse_id][sku] = ((current_units * current_age) + (units * age)) / total_units
        self.inventory[warehouse_id][sku] = current_units + units

    def _remove_warehouse_stock(self, warehouse_id: str, sku: str, units: int) -> None:
        self.inventory[warehouse_id][sku] = max(0, self.inventory[warehouse_id].get(sku, 0) - units)
        if self.inventory[warehouse_id][sku] == 0:
            self.inventory_age[warehouse_id][sku] = 0.0

    def _age_fresh_milk(self) -> None:
        if self.central_depot_inventory.get("fresh_milk", 0) > 0:
            self.central_depot_inventory_age["fresh_milk"] = self.central_depot_inventory_age.get("fresh_milk", 0.0) + 1.0
        for warehouse_id, inventory in self.inventory.items():
            if inventory.get("fresh_milk", 0) > 0:
                self.inventory_age[warehouse_id]["fresh_milk"] = self.inventory_age[warehouse_id].get("fresh_milk", 0.0) + 1.0

    def _shipment_invalid_reason(self, shipment: Any, used_units: int, cap: int, band: dict[str, float]) -> str:
        if shipment.to_warehouse not in self.inventory:
            return "unknown_destination_warehouse"
        if shipment.units <= 0:
            return "invalid_units"
        if shipment.unit_price > band["max_wholesale_price"]:
            return "unit_price_above_max_wholesale"
        if shipment.units > self.central_depot_inventory.get(shipment.sku, 0):
            return "insufficient_depot_stock"
        if used_units + shipment.units > cap:
            return "replenishment_cap_exceeded"
        if self.depot_trucks_available <= 0:
            return "depot_truck_unavailable"
        return "invalid_replenishment"

    def _receive_returns(self) -> None:
        for arrival, sku, units in [item for item in self.depot_procurements if item[0] <= self.round_index]:
            self._add_depot_stock(sku, units, 0.0)
        self.depot_procurements = [item for item in self.depot_procurements if item[0] > self.round_index]
        returned = [round_index for round_index in self.depot_truck_returns if round_index <= self.round_index]
        if returned:
            self.depot_trucks_available += len(returned)
            self.depot_truck_returns = [round_index for round_index in self.depot_truck_returns if round_index > self.round_index]
        for warehouse_id, return_rounds in self.driver_returns.items():
            done = [round_index for round_index in return_rounds if round_index <= self.round_index]
            if done:
                self.drivers_available[warehouse_id] += len(done)
                self.driver_returns[warehouse_id] = [round_index for round_index in return_rounds if round_index > self.round_index]

    def _visible_orders(self) -> list[OrderTemplate]:
        recipe = self._require_recipe()
        return [order for order in recipe.orders if order.created_round <= self.round_index and self.order_status[order.order_id] not in {"fulfilled", "rejected", "expired"}]

    def _require_recipe(self) -> HiddenRecipe:
        if self.recipe is None:
            self.reset(self.default_task_id, 101)
        assert self.recipe is not None
        return self.recipe


def _add(components: dict[str, float], key: str, value: float) -> None:
    components[key] = components.get(key, 0.0) + value


def _broker_fee(units: int, compensation: float) -> float:
    per_unit_fee = cfg("center_rewards", "transfer_broker_fee_per_unit") * units
    compensation_fee = cfg("center_rewards", "transfer_broker_fee_compensation_share") * compensation
    return min(per_unit_fee, compensation_fee)
