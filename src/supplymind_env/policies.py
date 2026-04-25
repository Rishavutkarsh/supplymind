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
    offer_matches = []

    offers = [signal for signal in observation.market_signals if signal.signal_type == "inventory_offer"]
    requests = [signal for signal in observation.market_signals if signal.signal_type == "inventory_request"]
    demand_by_key = {
        (report.warehouse_id, report.sku): report
        for report in observation.demand_reports
    }
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
    matched_requests = {match["request_signal_id"] for match in offer_matches}
    for report in sorted(observation.demand_reports, key=lambda item: (-item.urgency, -item.missed_units_last_round, -item.at_risk_units, -item.forecast_units)):
        if used_trucks >= observation.central_depot.trucks_available:
            break
        if f"{report.warehouse_id}:request:{report.sku}" in matched_requests:
            continue
        available = _warehouse_inventory(observation, report.warehouse_id, report.sku)
        pressure = max(0, report.requested_units + report.at_risk_units + report.missed_units_last_round - available)
        depot_left = observation.central_depot.inventory.get(report.sku, 0) - committed_depot.get(report.sku, 0)
        units = min(pressure, depot_left, 2)
        if units > 0 and (report.urgency >= 3 or report.missed_units_last_round > 0):
            central_replenishments.append({"to_warehouse": report.warehouse_id, "sku": report.sku, "units": units})
            committed_depot[report.sku] = committed_depot.get(report.sku, 0) + units
            used_trucks += 1

    return V3Action(
        central_replenishments=central_replenishments[: observation.central_depot.trucks_available],
        offer_matches=offer_matches[:3],
    )


def _warehouse_inventory(observation: V3Observation, warehouse_id: str, sku: str) -> int:
    for warehouse in observation.warehouses:
        if warehouse.warehouse_id == warehouse_id:
            return warehouse.inventory.get(sku, 0)
    return 0


__all__ = ["baseline_policy", "heuristic_policy", "no_op_policy"]
