from __future__ import annotations

from .models import V3Action, V3Observation


def no_op_policy(observation: V3Observation) -> V3Action:
    return V3Action()


def baseline_policy(observation: V3Observation) -> V3Action:
    replenishments = []
    used_trucks = 0
    for report in sorted(observation.demand_reports, key=lambda item: (-item.urgency, -item.requested_units - item.at_risk_units)):
        if used_trucks >= observation.central_depot.trucks_available:
            break
        shortage = max(0, report.requested_units + report.at_risk_units + report.forecast_units - _warehouse_inventory(observation, report.warehouse_id, report.sku))
        units = min(shortage, observation.central_depot.inventory.get(report.sku, 0), 2)
        if units > 0:
            replenishments.append({"to_warehouse": report.warehouse_id, "sku": report.sku, "units": units})
            used_trucks += 1
    return V3Action(central_replenishments=replenishments)


def heuristic_policy(observation: V3Observation) -> V3Action:
    central_replenishments = list(baseline_policy(observation).central_replenishments)
    central_procurements = []
    offer_matches = []

    offers = [signal for signal in observation.market_signals if signal.signal_type == "inventory_offer"]
    requests = [signal for signal in observation.market_signals if signal.signal_type == "inventory_request"]
    demand_by_key = {
        (report.warehouse_id, report.sku): report
        for report in observation.demand_reports
    }
    inbound_by_sku: dict[str, int] = {}
    for item in observation.central_depot.inbound_procurements:
        sku = str(item.get("sku", ""))
        inbound_by_sku[sku] = inbound_by_sku.get(sku, 0) + int(item.get("units", 0))
    used_offers: set[str] = set()
    used_requests: set[str] = set()
    for request in sorted(requests, key=lambda signal: (-signal.urgency, -signal.units)):
        for offer in sorted(offers, key=lambda signal: signal.ask_price):
            if offer.signal_id in used_offers or request.signal_id in used_requests:
                continue
            if offer.sku != request.sku or offer.warehouse_id == request.warehouse_id:
                continue
            report = demand_by_key.get((request.warehouse_id, request.sku))
            if report is None or (report.urgency < 2 and report.missed_units_last_round <= 0):
                continue
            units = min(offer.units, request.units)
            if report.missed_units_last_round <= 0 or units < 2:
                continue
            offer_matches.append(
                {
                    "offer_signal_id": offer.signal_id,
                    "request_signal_id": request.signal_id,
                    "units": units,
                    "compensation": max(offer.ask_price * units, 1.0),
                }
            )
            used_offers.add(offer.signal_id)
            used_requests.add(request.signal_id)
            break

    used_trucks = 0
    committed_depot: dict[str, int] = {}
    procurement_by_sku: dict[str, int] = {}
    matched_requests = {match["request_signal_id"] for match in offer_matches}
    for report in sorted(observation.demand_reports, key=lambda item: (-item.urgency, -item.missed_units_last_round, -item.at_risk_units, -item.forecast_units)):
        available = _warehouse_inventory(observation, report.warehouse_id, report.sku)
        if used_trucks < observation.central_depot.trucks_available and f"{report.warehouse_id}:request:{report.sku}" not in matched_requests:
            pressure = max(0, report.requested_units + report.at_risk_units + report.missed_units_last_round - available)
            depot_left = observation.central_depot.inventory.get(report.sku, 0) - committed_depot.get(report.sku, 0)
            units = min(pressure, depot_left, 2)
            if units > 0 and (report.urgency >= 3 or report.missed_units_last_round > 0):
                central_replenishments.append({"to_warehouse": report.warehouse_id, "sku": report.sku, "units": units})
                committed_depot[report.sku] = committed_depot.get(report.sku, 0) + units
                used_trucks += 1

        projected_pressure = report.requested_units + report.at_risk_units + report.forecast_units + report.missed_units_last_round
        depot_left_after_shipments = (
            observation.central_depot.inventory.get(report.sku, 0)
            + inbound_by_sku.get(report.sku, 0)
            + procurement_by_sku.get(report.sku, 0)
            - committed_depot.get(report.sku, 0)
        )
        enough_time_to_use_purchase = observation.remaining_rounds > observation.central_depot.procurement_lead_time + 2
        if enough_time_to_use_purchase and report.urgency >= 2 and projected_pressure >= 5 and depot_left_after_shipments <= 1:
            already = procurement_by_sku.get(report.sku, 0)
            units_to_buy = min(2, projected_pressure - depot_left_after_shipments, observation.central_depot.procurement_cap - sum(procurement_by_sku.values()))
            if units_to_buy > 0:
                procurement_by_sku[report.sku] = already + units_to_buy

    for sku, units in procurement_by_sku.items():
        central_procurements.append({"sku": sku, "units": units})

    return V3Action(
        central_procurements=central_procurements,
        central_replenishments=central_replenishments[: observation.central_depot.trucks_available],
        offer_matches=offer_matches[:3],
    )


def _warehouse_inventory(observation: V3Observation, warehouse_id: str, sku: str) -> int:
    for warehouse in observation.warehouses:
        if warehouse.warehouse_id == warehouse_id:
            return warehouse.inventory.get(sku, 0)
    return 0


__all__ = ["baseline_policy", "heuristic_policy", "no_op_policy"]
