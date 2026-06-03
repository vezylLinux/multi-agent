"""
VRP-based itinerary optimizer using TrackAsia VRP API.

Given a hotel, a ranked list of attractions and restaurants, and the number of days,
returns a per-day assignment of places that minimizes total travel distance.
The result has the same shape as _llm_plan_itinerary output (indices into the input lists)
so it can be dropped into build_trip_plan_payload as a direct replacement.
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import Any

_LOGGER = logging.getLogger(__name__)
_VRP_URL = "https://maps.track-asia.com/api/v1/vrp"

# Job ID encoding to distinguish attractions from restaurants
_ATTRACTION_ID_BASE = 10000
_RESTAURANT_ID_BASE = 20000

# Service time per stop (seconds)
_SERVICE_ATTRACTION = 7200   # 2 hours per attraction
_SERVICE_RESTAURANT = 3600   # 1 hour per meal

# Day time window: 8 AM to 9 PM = 13 hours
_DAY_START_S = 28800   # 8 * 3600
_DAY_END_S = 75600     # 21 * 3600
_DAY_OFFSET_S = 86400  # 24 hours between days

# Max stops per day vehicle (2 attractions + 3 restaurants)
_MAX_TASKS_PER_DAY = 5


def optimize_itinerary(
    hotel: dict[str, Any],
    attractions: list[dict[str, Any]],
    restaurants: list[dict[str, Any]],
    total_days: int,
) -> list[dict[str, Any]] | None:
    """Optimize place-to-day assignment using TrackAsia VRP API.

    Returns a list of per-day dicts:
        [{"day": 1, "morning_idx": int, "afternoon_idx": int,
          "breakfast_idx": int, "lunch_idx": int, "dinner_idx": int}, ...]

    Indices are into the input `attractions` and `restaurants` lists respectively.
    Returns None when VRP is unavailable or fails; caller falls back to LLM planning.
    """
    if not attractions or total_days < 1:
        return None

    h_lon, h_lat = _get_lonlat(hotel)
    if h_lon is None:
        return None

    try:
        from app.core.settings import get_settings
        settings = get_settings()
        api_key = str(settings.trackasia_api_key or "").strip()
        if not api_key or not bool(settings.trackasia_enabled):
            return None
    except Exception:
        return None

    # Limit input size to keep request small
    attractions = attractions[:total_days * 4]
    restaurants = restaurants[:total_days * 6]

    jobs = _build_jobs(attractions, restaurants)
    vehicles = _build_vehicles(h_lon, h_lat, total_days)

    payload = json.dumps({"jobs": jobs, "vehicles": vehicles, "options": {"g": False}}, ensure_ascii=False)

    try:
        req = urllib.request.Request(
            f"{_VRP_URL}?key={urllib.parse.quote(api_key)}",
            data=payload.encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "multi-agent-travel/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        _LOGGER.warning("VRP API call failed: %s", exc)
        return None

    if data.get("code") != 0:
        _LOGGER.warning("VRP returned non-zero code: %s", data.get("code"))
        return None

    return _parse_routes(data.get("routes") or [], total_days)


def _build_jobs(attractions: list[dict], restaurants: list[dict]) -> list[dict]:
    jobs = []
    for i, p in enumerate(attractions):
        lon, lat = _get_lonlat(p)
        if lon is None:
            continue
        jobs.append({
            "id": _ATTRACTION_ID_BASE + i,
            "location": [lon, lat],
            "service": _SERVICE_ATTRACTION,
            "description": str(p.get("name") or "")[:60],
        })
    for i, p in enumerate(restaurants):
        lon, lat = _get_lonlat(p)
        if lon is None:
            continue
        jobs.append({
            "id": _RESTAURANT_ID_BASE + i,
            "location": [lon, lat],
            "service": _SERVICE_RESTAURANT,
            "description": str(p.get("name") or "")[:60],
        })
    return jobs


def _build_vehicles(h_lon: float, h_lat: float, total_days: int) -> list[dict]:
    vehicles = []
    for day in range(total_days):
        offset = day * _DAY_OFFSET_S
        vehicles.append({
            "id": day + 1,
            "start": [h_lon, h_lat],
            "end": [h_lon, h_lat],
            "time_window": [_DAY_START_S + offset, _DAY_END_S + offset],
            "max_tasks": _MAX_TASKS_PER_DAY,
        })
    return vehicles


def _parse_routes(routes: list[dict], total_days: int) -> list[dict[str, Any]] | None:
    day_assignments: dict[int, dict[str, Any]] = {}
    used_attraction_ids: set[int] = set()
    used_restaurant_ids: set[int] = set()

    for route in routes:
        vehicle_id = route.get("vehicle")
        if not isinstance(vehicle_id, int) or vehicle_id < 1 or vehicle_id > total_days:
            continue

        attraction_indices: list[int] = []
        restaurant_indices: list[int] = []

        for step in route.get("steps") or []:
            if step.get("type") != "job":
                continue
            job_id = step.get("id")
            if not isinstance(job_id, int):
                continue
            if job_id >= _RESTAURANT_ID_BASE:
                idx = job_id - _RESTAURANT_ID_BASE
                if idx not in used_restaurant_ids:
                    restaurant_indices.append(idx)
                    used_restaurant_ids.add(idx)
            elif job_id >= _ATTRACTION_ID_BASE:
                idx = job_id - _ATTRACTION_ID_BASE
                if idx not in used_attraction_ids:
                    attraction_indices.append(idx)
                    used_attraction_ids.add(idx)

        day_assignments[vehicle_id] = {
            "day": vehicle_id,
            "morning_idx": attraction_indices[0] if len(attraction_indices) > 0 else None,
            "afternoon_idx": attraction_indices[1] if len(attraction_indices) > 1 else None,
            "breakfast_idx": restaurant_indices[0] if len(restaurant_indices) > 0 else None,
            "lunch_idx": restaurant_indices[1] if len(restaurant_indices) > 1 else None,
            "dinner_idx": restaurant_indices[2] if len(restaurant_indices) > 2 else None,
        }

    if not day_assignments:
        return None

    result = []
    for day in range(1, total_days + 1):
        result.append(day_assignments.get(day) or {"day": day})
    return result


def _get_lonlat(place: dict[str, Any]) -> tuple[float | None, float | None]:
    lat = place.get("lat")
    lon = place.get("lon")
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        return float(lon), float(lat)
    return None, None
