from __future__ import annotations

import json
import re
from typing import List

from app.itinerary.routing import (
    haversine_km as _shared_haversine_km,
    place_map_url as _shared_place_map_url,
    resolve_point_for_map as _shared_resolve_point_for_map,
    resolve_segment_points as _shared_resolve_segment_points,
    segment_map_url as _shared_segment_map_url,
)
from functools import lru_cache
from app.places.metadata import fold_text as _fold
from app.places.metadata import extract_trip_days
from app.places.search import search_processed_places as search_places
from app.tools.trackasia import GeoPoint, configured_route_modes, estimate_route


@lru_cache(maxsize=1)
def _db_coord_index() -> dict[str, tuple[float, float]]:
    """Folded-name → (lat, lon) index from DB for coord enrichment."""
    try:
        from app.places.repository import list_places
        return {
            _fold(str(p.get("name") or "")): (float(p["lat"]), float(p["lon"]))
            for p in list_places()
            if isinstance(p.get("lat"), (int, float)) and isinstance(p.get("lon"), (int, float))
        }
    except Exception:
        return {}


def _enrich_coords(places: List[dict]) -> List[dict]:
    """Fill missing lat/lon from DB index (in-place mutation, returns same list)."""
    idx = _db_coord_index()
    for p in places:
        if isinstance(p.get("lat"), (int, float)) and isinstance(p.get("lon"), (int, float)):
            continue
        key = _fold(str(p.get("name") or ""))
        coords = idx.get(key)
        if coords:
            p["lat"], p["lon"] = coords
    return places


@lru_cache(maxsize=8)
def _load_db_attractions(city_key: str) -> List[dict]:
    try:
        from app.places.repository import list_places
        from app.places.metadata import city_key_from_text as _meta_city_key
        all_places = list_places()
    except Exception:
        return []
    result = []
    for p in all_places:
        cat = str(p.get("category") or "").lower()
        if cat in ("restaurant", "accommodation"):
            continue
        name = str(p.get("name") or "").strip()
        if not name:
            continue
        if city_key:
            pk = str(p.get("city_key") or "").strip().lower()
            if pk and pk != city_key:
                continue
            addr = str(p.get("address") or "")
            addr_city = _meta_city_key(addr)
            if addr_city and addr_city != city_key:
                continue
        result.append(p)
    return result


@lru_cache(maxsize=8)
def _load_db_restaurants(city_key: str) -> List[dict]:
    try:
        from app.places.repository import list_places
        all_places = list_places()
    except Exception:
        return []
    result = []
    for p in all_places:
        if str(p.get("category") or "").lower() != "restaurant":
            continue
        if not isinstance(p.get("lat"), (int, float)) or not isinstance(p.get("lon"), (int, float)):
            continue
        if city_key:
            pk = str(p.get("city_key") or "").strip().lower()
            if pk and pk != city_key:
                continue
        result.append(p)
    return result


_AREA_SEARCH_LABELS: dict[str, str] = {
    "hai chau": "Hai Chau, Da Nang",
    "son tra": "Son Tra, Da Nang",
    "ngu hanh son": "Ngu Hanh Son, Da Nang",
    "thanh khe": "Thanh Khe, Da Nang",
    "cam le": "Cam Le, Da Nang",
    "lien chieu": "Lien Chieu, Da Nang",
    "hoa vang": "Hoa Vang, Da Nang",
}

_AREA_TO_CITY_KEY: dict[str, str] = {
    "hai chau": "da_nang",
    "son tra": "da_nang",
    "ngu hanh son": "da_nang",
    "thanh khe": "da_nang",
    "cam le": "da_nang",
    "lien chieu": "da_nang",
    "hoa vang": "da_nang",
}

_CITY_NAME_PATTERNS: dict[str, tuple[str, ...]] = {
    "da_nang": ("da nang", "danang"),
}
_HOTEL_RELOCATION_THRESHOLD_KM = 30.0


def _safe_idx(items: List[dict], idx) -> dict | None:
    if not isinstance(idx, int) or idx < 0 or idx >= len(items):
        return None
    return items[idx]


def _next_unused(items: List[dict], used: set[str]) -> dict | None:
    for p in items:
        k = _place_key(p)
        if k and k not in used:
            return p
    return None


def _nearest_unused_restaurant(
    food_pool: List[dict],
    anchor: dict | None,
    used_keys: set[str],
) -> dict | None:
    anchor_pt = _resolve_point_for_map(anchor) if anchor else None
    if not anchor_pt:
        return _next_unused(food_pool, used_keys)
    best: dict | None = None
    best_dist = float("inf")
    fallback: dict | None = None
    for p in food_pool:
        k = _place_key(p)
        if not k or k in used_keys:
            continue
        pt = _resolve_point_for_map(p)
        if not pt:
            if fallback is None:
                fallback = p
            continue
        dist = _haversine_km(anchor_pt[0], anchor_pt[1], pt[0], pt[1])
        if dist < best_dist:
            best_dist = dist
            best = p
    return best or fallback


def _llm_plan_itinerary(
    destinations: List[dict],
    restaurants: List[dict],
    hotels: List[dict],
    query: str,
    total_days: int,
    city: str,
) -> dict | None:
    try:
        from openai import OpenAI
        from app.core.settings import get_settings
        settings = get_settings()
        api_key = settings.openrouter_api_key
        if not api_key:
            return None
        client = OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=api_key,
            timeout=max(1, int(settings.openrouter_request_timeout_s or 30)),
        )
    except Exception:
        return None

    def _compact(p: dict, idx: int) -> str:
        name = str(p.get("name") or "").strip()
        addr = str(p.get("address") or str(p.get("city") or "")).strip()
        district = _primary_district_key(p)
        area_tag = f"[{district}]" if district else ""
        label = f"{addr} {area_tag}".strip() if addr else area_tag
        return f"{idx}: {name} | {label}" if label else f"{idx}: {name}"

    # Pre-cluster attractions by district so the LLM can assign whole clusters per day.
    district_clusters: dict[str, list[int]] = {}
    for i, p in enumerate(destinations):
        dk = _primary_district_key(p) or "other"
        district_clusters.setdefault(dk, []).append(i)

    cluster_lines = "\n".join(
        f"  cluster [{dk}]: indices {idxs}"
        for dk, idxs in sorted(district_clusters.items())
    )

    dest_lines = "\n".join(_compact(p, i) for i, p in enumerate(destinations))
    rest_lines = "\n".join(_compact(p, i) for i, p in enumerate(restaurants))
    hotel_lines = (
        "\n".join(_compact(p, i) for i, p in enumerate(hotels))
        if hotels else "(none available)"
    )

    system_prompt = (
        "You are a travel itinerary planner for Vietnam. "
        "Create a logical, preference-matched multi-day itinerary from the given lists. "
        "Return only valid JSON, no explanation, no markdown."
    )
    user_prompt = (
        f"Plan a {total_days}-day trip to {city}.\n"
        f'User request: "{query}"\n\n'
        f"Attractions (index: name | district):\n{dest_lines}\n\n"
        f"Geographic clusters (assign entire clusters to the same day to minimise travel):\n{cluster_lines}\n\n"
        f"Restaurants (index: name | area):\n{rest_lines}\n\n"
        f"Hotels (index: name | area):\n{hotel_lines}\n\n"
        "Rules:\n"
        "- Use each attraction index at most once across all days\n"
        "- Use each restaurant index at most once across all days\n"
        "- CRITICAL: morning_idx and afternoon_idx on the same day MUST be from the same district cluster.\n"
        "  Never mix attractions from son_tra and ngu_hanh_son on the same day — they are 20+ km apart.\n"
        "- Match user preferences (e.g. hai san=seafood restaurants, tham chua=temple attractions, di bien=beach spots)\n"
        "- Choose meals near the day's attractions\n"
        "- Viết day_theme, morning_action, afternoon_action, evening_action bằng tiếng Việt\n"
        "- day_theme: nhãn ngắn tóm tắt chủ đề ngày (ví dụ: 'Văn hoá & Tâm linh', 'Biển & Thư giãn')\n"
        "- morning_action / afternoon_action: một cụm từ ngắn mô tả hoạt động tại địa điểm đó\n"
        "- evening_action: một cụm từ ngắn về trải nghiệm bữa tối\n\n"
        "Return JSON in exactly this format:\n"
        '{"hotel_index": <int>, "days": [{"day": <int>, "morning_idx": <int>, "morning_action": <str>, '
        '"afternoon_idx": <int>, "afternoon_action": <str>, "breakfast_idx": <int>, "lunch_idx": <int>, '
        '"dinner_idx": <int>, "evening_action": <str>, "day_theme": <str>}]}'
    )

    try:
        resp = client.chat.completions.create(
            model=settings.openrouter_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)
        plan = json.loads(raw.strip())
        if not isinstance(plan.get("days"), list):
            return None
        return plan
    except Exception:
        return None


def build_trip_plan_payload(query: str, places: List[dict], strict_mode: bool = False, budget_tier: str = "") -> dict:
    if not places:
        return {
            "plan": "Not enough data to generate an itinerary.",
            "stay_plan": {"segments": [], "change_hotel": False},
            "recommended_hotel": None,
        }

    total_days = extract_trip_days(query) or 1
    destinations = _filter_by_categories(places, ["destination", "museum", "viewpoint", "entertainment"])
    food_places = _filter_by_categories(places, ["restaurant"])
    hotels = _filter_by_categories(places, ["accommodation"])
    city = _extract_city(query, places)
    target_city_keys = _target_city_keys(query, city)
    strict_target_city_key = target_city_keys[0] if len(target_city_keys) == 1 else ""

    if strict_target_city_key:
        destinations_by_city = [p for p in destinations if _place_city_key(p) == strict_target_city_key]
        food_by_city = [p for p in food_places if _place_city_key(p) == strict_target_city_key]
        hotels_by_city = [p for p in hotels if _place_city_key(p) == strict_target_city_key]
        if destinations_by_city:
            destinations = destinations_by_city
        if food_by_city:
            food_places = food_by_city
        if hotels_by_city:
            hotels = hotels_by_city

    if not destinations:
        destinations = [p for p in places if str(p.get("category") or "").lower() != "restaurant"]

    destination_name = _extract_city(query, places).upper()
    lines: List[str] = [
        f"LỊCH TRÌNH {total_days} NGÀY TẠI {destination_name}",
        "",
    ]
    route_plan: List[dict] = []

    attraction_pool = _rank_attractions(_unique_by_name(destinations))

    db_dest = _load_db_attractions(strict_target_city_key or "")
    existing_keys = {_place_key(p) for p in attraction_pool}
    extra = [p for p in db_dest if _place_key(p) not in existing_keys]
    attraction_pool = attraction_pool + _rank_attractions(extra)

    food_pool = _unique_by_name(food_places)
    db_restaurants = _load_db_restaurants(strict_target_city_key or "")
    existing_rest_keys = {_place_key(p) for p in food_pool}
    extra_rest = [p for p in db_restaurants if _place_key(p) not in existing_rest_keys]
    food_pool = food_pool + extra_rest

    _enrich_coords(attraction_pool)
    _enrich_coords(food_pool)
    _enrich_coords(hotels)

    # Remove places whose coordinates fall outside the target city bounding box.
    # This prevents out-of-area places (e.g. Quảng Nam waterfalls mis-tagged as
    # Da Nang) from creating extreme legs in the route plan.
    city_bbox = _city_bbox(strict_target_city_key or _city_key_from_text(city))
    if city_bbox:
        attraction_pool = _filter_by_bbox(attraction_pool, city_bbox)
        food_pool = _filter_by_bbox(food_pool, city_bbox)

    # Try VRP first to get geographically optimal day assignments.
    # Falls back to LLM if VRP is unavailable or returns no routes.
    vrp_days: dict[int, dict] = {}
    try:
        from app.itinerary.vrp import optimize_itinerary as _vrp_optimize
        hotel_for_vrp = hotels[0] if hotels else None
        vrp_result = _vrp_optimize(
            hotel=hotel_for_vrp or {},
            attractions=attraction_pool[:total_days * 4],
            restaurants=food_pool[:total_days * 6],
            total_days=total_days,
        )
        if vrp_result:
            vrp_days = {int(d["day"]): d for d in vrp_result if isinstance(d.get("day"), int)}
    except Exception:
        pass

    llm_plan = None
    if not vrp_days:
        llm_plan = _llm_plan_itinerary(
            destinations=attraction_pool[:50],
            restaurants=food_pool[:50],
            hotels=hotels[:10],
            query=query,
            total_days=total_days,
            city=city,
        )

    used_attraction_keys: set[str] = set()
    used_restaurant_keys: set[str] = set()
    daily_frames: List[dict] = []

    for day in range(1, total_days + 1):
        # VRP takes priority; fall back to LLM plan when VRP has no assignment for this day
        day_plan: dict | None = vrp_days.get(day) or None
        if day_plan is None and llm_plan and isinstance(llm_plan.get("days"), list):
            for dp in llm_plan["days"]:
                if isinstance(dp, dict) and dp.get("day") == day:
                    day_plan = dp
                    break

        morning = _safe_idx(attraction_pool, day_plan.get("morning_idx") if day_plan else None)
        afternoon = _safe_idx(attraction_pool, day_plan.get("afternoon_idx") if day_plan else None)
        breakfast = _safe_idx(food_pool, day_plan.get("breakfast_idx") if day_plan else None)
        lunch = _safe_idx(food_pool, day_plan.get("lunch_idx") if day_plan else None)
        dinner = _safe_idx(food_pool, day_plan.get("dinner_idx") if day_plan else None)

        if morning is None or _place_key(morning) in used_attraction_keys:
            morning = _next_unused(attraction_pool, used_attraction_keys)
        if morning:
            used_attraction_keys.add(_place_key(morning))

        if afternoon is None or _place_key(afternoon) in used_attraction_keys:
            afternoon = _next_unused(attraction_pool, used_attraction_keys)
        if afternoon:
            used_attraction_keys.add(_place_key(afternoon))

        if breakfast is None or _place_key(breakfast) in used_restaurant_keys:
            breakfast = _nearest_unused_restaurant(food_pool, morning, used_restaurant_keys)
        if breakfast:
            used_restaurant_keys.add(_place_key(breakfast))

        if lunch is None or _place_key(lunch) in used_restaurant_keys:
            lunch = _nearest_unused_restaurant(food_pool, morning, used_restaurant_keys)
        if lunch:
            used_restaurant_keys.add(_place_key(lunch))

        if dinner is None or _place_key(dinner) in used_restaurant_keys:
            dinner = _nearest_unused_restaurant(food_pool, afternoon, used_restaurant_keys)
        if dinner:
            used_restaurant_keys.add(_place_key(dinner))

        daily_frames.append({
            "day": day,
            "morning": morning,
            "afternoon": afternoon,
            "breakfast": breakfast,
            "lunch": lunch,
            "dinner": dinner,
            "city_key": _day_city_key(
                morning, afternoon,
                strict_target_city_key or _city_key_from_text(city),
            ),
        })

    _rebalance_day_attractions(daily_frames)

    stay_plan = _select_stay_plan(
        daily_frames=daily_frames,
        hotels=hotels,
        city=city,
        strict_mode=strict_mode,
        budget_tier=budget_tier,
    )
    daily_stays = {
        int(item.get("day")): item
        for item in (stay_plan.get("daily") or [])
        if isinstance(item, dict) and isinstance(item.get("day"), int)
    }
    if stay_plan.get("segments"):
        lines.extend(
            [
                "LƯU TRÚ:",
                *[_stay_segment_line(segment) for segment in stay_plan.get("segments", [])],
                "",
            ]
        )

    for frame in daily_frames:
        day = int(frame["day"])
        morning = frame.get("morning")
        afternoon = frame.get("afternoon")
        breakfast = frame.get("breakfast")
        lunch = frame.get("lunch")
        dinner = frame.get("dinner")
        daily_stay = daily_stays.get(day) or {}
        start_hotel = daily_stay.get("start_hotel")
        end_hotel = daily_stay.get("end_hotel")

        day_route_plan = _build_day_route_plan(
            day=day,
            start_hotel=start_hotel,
            end_hotel=end_hotel,
            breakfast=breakfast,
            morning=morning,
            lunch=lunch,
            afternoon=afternoon,
            dinner=dinner,
        )
        route_plan.extend(day_route_plan)
        day_plan_entry: dict | None = None
        if llm_plan and isinstance(llm_plan.get("days"), list):
            for dp in llm_plan["days"]:
                if isinstance(dp, dict) and dp.get("day") == day:
                    day_plan_entry = dp
                    break
        day_theme = str((day_plan_entry or {}).get("day_theme") or "").strip() or _infer_day_theme(morning, afternoon)
        morning_action = str((day_plan_entry or {}).get("morning_action") or "").strip() or _infer_action(morning)
        afternoon_action = str((day_plan_entry or {}).get("afternoon_action") or "").strip() or _infer_action(afternoon)
        evening_action = str((day_plan_entry or {}).get("evening_action") or "").strip() or "thưởng thức ẩm thực địa phương"

        morning_line = (
            f"• Buổi sáng: Ăn sáng tại {_fmt(breakfast)}. Hoạt động: {morning_action} tại {_fmt(morning)}"
            if morning else
            f"• Buổi sáng: Ăn sáng tại {_fmt(breakfast)}"
        )
        afternoon_line = (
            f"• Buổi chiều: Hoạt động: {afternoon_action} tại {_fmt(afternoon)}"
            if afternoon else
            f"• Buổi chiều: Tự do / nghỉ ngơi"
        )
        lines.extend(
            [
                "",
                f"NGÀY {day} - {day_theme}",
                morning_line,
                f"• Buổi trưa: Ăn trưa tại {_fmt(lunch)}",
                afternoon_line,
                f"• Buổi tối: Ăn tối tại {_fmt(dinner)}. {evening_action}; khám phá xung quanh {str((dinner or {}).get('name') or '').strip()}",
            ]
        )

    lines.extend(
        [
            "",
            "Tổng quan: Lịch trình cân bằng giữa tham quan, trải nghiệm địa phương và thời gian di chuyển hợp lý.",
            "Lưu ý: Các khung giờ có thể điều chỉnh tùy thời tiết và giờ mở cửa thực tế.",
        ]
    )
    all_places = list({
        id(p): p
        for p in (attraction_pool + food_pool + hotels)
    }.values())

    return {
        "plan": "\n".join(lines).strip(),
        "stay_plan": stay_plan,
        "recommended_hotel": _recommended_hotel_from_stay_plan(stay_plan),
        "route_plan": route_plan,
        "all_places": all_places,
    }


_CITY_BBOXES: dict[str, tuple[float, float, float, float]] = {
    # (lat_min, lat_max, lon_min, lon_max)
    # Da Nang urban+coastal strip — excludes far-western mountains (Bà Nà, Núi Chúa
    # at lon~107.9) which require a dedicated full-day trip and create >25 km legs
    # when combined with city/beach attractions.
    "da_nang": (15.92, 16.22, 108.00, 108.45),
    "hoi_an":  (15.75, 15.95, 108.25, 108.45),
}


def _city_bbox(city_key: str) -> tuple[float, float, float, float] | None:
    return _CITY_BBOXES.get(city_key.strip().lower())


def _filter_by_bbox(
    places: List[dict],
    bbox: tuple[float, float, float, float],
) -> List[dict]:
    """Remove places whose coordinates are outside the given bounding box.
    Places without coordinates are kept (coord may be filled later or missing)."""
    lat_min, lat_max, lon_min, lon_max = bbox
    result: List[dict] = []
    for p in places:
        lat = p.get("lat")
        lon = p.get("lon")
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            result.append(p)
            continue
        if lat_min <= float(lat) <= lat_max and lon_min <= float(lon) <= lon_max:
            result.append(p)
    return result


_GEO_REBALANCE_THRESHOLD_KM = 15.0

# Districts whose centroids are reliably > 20 km apart — used when coords are absent.
_FAR_DISTRICT_PAIRS: frozenset[frozenset] = frozenset({
    frozenset({"son tra", "ngu hanh son"}),
    frozenset({"son tra", "hoa vang"}),
    frozenset({"lien chieu", "ngu hanh son"}),
    frozenset({"lien chieu", "hoa vang"}),
})


def _districts_are_far(p_a: dict | None, p_b: dict | None) -> bool:
    """Return True if two places are from a known far-apart district pair."""
    if not p_a or not p_b:
        return False
    dk_a = _primary_district_key(p_a)
    dk_b = _primary_district_key(p_b)
    if not dk_a or not dk_b or dk_a == dk_b:
        return False
    return frozenset({dk_a, dk_b}) in _FAR_DISTRICT_PAIRS


def _rebalance_day_attractions(daily_frames: List[dict]) -> None:
    """Swap afternoon attractions between days to reduce extreme same-day leg distances.

    Two strategies applied in sequence:
    1. Coordinate-based: compute haversine; swap if it reduces the worst distance.
    2. District-based fallback: for places without coords, use known far-district pairs.
    Mutates daily_frames in-place.
    """
    MAX_PASSES = 4
    for _ in range(MAX_PASSES):
        improved = False
        for i, frame_a in enumerate(daily_frames):
            m_a = frame_a.get("morning")
            af_a = frame_a.get("afternoon")
            if not m_a or not af_a:
                continue

            m_a_pt = _resolve_point_for_map(m_a)
            af_a_pt = _resolve_point_for_map(af_a)

            # Determine whether this pair is problematic
            coord_based = m_a_pt and af_a_pt
            if coord_based:
                dist_a = _haversine_km(m_a_pt[0], m_a_pt[1], af_a_pt[0], af_a_pt[1])
                pair_is_bad = dist_a > _GEO_REBALANCE_THRESHOLD_KM
            else:
                dist_a = float("inf")
                pair_is_bad = _districts_are_far(m_a, af_a)

            if not pair_is_bad:
                continue

            best_j: int | None = None
            best_score = dist_a  # lower is better
            for j, frame_b in enumerate(daily_frames):
                if j == i:
                    continue
                af_b = frame_b.get("afternoon")
                if not af_b:
                    continue

                af_b_pt = _resolve_point_for_map(af_b)
                if coord_based and af_b_pt:
                    new_dist_a = _haversine_km(m_a_pt[0], m_a_pt[1], af_b_pt[0], af_b_pt[1])
                    if new_dist_a >= dist_a:
                        continue
                    # Allow swap as long as day B doesn't exceed the HARD ceiling (1.5x threshold).
                    # We prefer fixing day A's extreme leg even if day B gets somewhat worse.
                    m_b = frame_b.get("morning")
                    if m_b:
                        m_b_pt = _resolve_point_for_map(m_b)
                        if m_b_pt and af_a_pt:
                            new_dist_b = _haversine_km(m_b_pt[0], m_b_pt[1], af_a_pt[0], af_a_pt[1])
                            if new_dist_b > _GEO_REBALANCE_THRESHOLD_KM * 1.5:
                                continue
                    if new_dist_a < best_score:
                        best_score = new_dist_a
                        best_j = j
                else:
                    # Fallback: prefer swapping with an afternoon from the same district as morning
                    dk_m = _primary_district_key(m_a)
                    dk_af_b = _primary_district_key(af_b)
                    if dk_m and dk_af_b and dk_m == dk_af_b:
                        best_j = j
                        break  # take first same-district candidate

            if best_j is not None:
                daily_frames[i]["afternoon"], daily_frames[best_j]["afternoon"] = (
                    daily_frames[best_j]["afternoon"],
                    daily_frames[i]["afternoon"],
                )
                improved = True

        if not improved:
            break


def _infer_day_theme(morning: dict | None, afternoon: dict | None) -> str:
    tags: list[str] = []
    for p in [morning, afternoon]:
        if not p:
            continue
        name = _fold(str(p.get("name") or ""))
        cat = str(p.get("category") or "").lower()
        intent = " ".join(str(t) for t in (p.get("intent_tags") or []))
        tags.append(f"{name} {cat} {intent}")
    combined = " ".join(tags).lower()
    if any(k in combined for k in ("chua", "linh", "pagoda", "temple", "den", "tam linh", "spiritual")):
        if any(k in combined for k in ("bien", "beach", "sea", "bai", "da nang")):
            return "Văn hoá & Biển cả"
        return "Văn hoá & Tâm linh"
    if any(k in combined for k in ("bien", "beach", "sea", "bai bien", "bai tam")):
        return "Biển & Thư giãn"
    if "cong vien" in combined or "park" in combined:
        return "Công viên & Khám phá thành phố"
    if any(k in combined for k in ("museum", "bao tang", "historical", "di tich", "thanh", "fortress")):
        return "Lịch sử & Di tích"
    if any(k in combined for k in ("nui", "mountain", "nature", "thac", "waterfall", "eco")):
        return "Thiên nhiên & Phiêu lưu"
    return "Tham quan & Ẩm thực địa phương"


def _infer_action(place: dict | None) -> str:
    if not place:
        return "khám phá điểm nổi bật"
    name = str(place.get("name") or "").strip()
    cat = str(place.get("category") or "").lower()
    name_fold = _fold(name)
    if any(k in name_fold for k in ("chua", "pagoda", "temple", "den")):
        return f"Tham quan {name}"
    if cat == "museum" or "bao tang" in name_fold:
        return f"Khám phá {name}"
    if any(k in name_fold for k in ("bien", "beach", "bai", "cong vien")):
        return f"Thư giãn tại {name}"
    if cat == "viewpoint" or any(k in name_fold for k in ("nui", "mountain", "deo", "pass")):
        return f"Chinh phục {name}"
    return f"Khám phá {name}"


def _filter_by_categories(places: List[dict], categories: List[str]) -> List[dict]:
    wanted = {c.lower() for c in categories}
    return [p for p in places if str(p.get("category") or "").lower() in wanted]


def _place_key(p: dict | None) -> str:
    if not p:
        return ""
    return str(p.get("name") or "").strip().lower()


def _unique_by_name(items: List[dict]) -> List[dict]:
    out: List[dict] = []
    seen: set[str] = set()
    for it in items:
        name = _place_key(it)
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(it)
    return out


def _attraction_fame_score(p: dict) -> float:
    cat = str(p.get("category") or "").lower()
    name_fold = _fold(str(p.get("name") or ""))
    score = 0.0
    if cat == "museum":
        score += 5.0
    elif cat == "viewpoint":
        score += 4.0
    elif cat == "destination":
        score += 2.0
    if name_fold.startswith("duong "):
        score -= 6.0
    return score


def _rank_attractions(places: List[dict]) -> List[dict]:
    return sorted(places, key=lambda p: (_attraction_fame_score(p), _place_key(p)), reverse=True)


def _primary_district_key(p: dict) -> str:
    areas = _extract_admin_areas(p)
    if not areas:
        return ""
    for k in (
        "hai chau",
        "son tra",
        "ngu hanh son",
        "thanh khe",
        "cam le",
        "lien chieu",
        "hoa vang",
    ):
        if k in areas:
            return k
    return sorted(areas)[0]


def _extract_admin_areas(p: dict) -> set[str]:
    explicit = p.get("admin_area_keys")
    if isinstance(explicit, list):
        out = {str(item).strip().lower() for item in explicit if str(item).strip()}
        if out:
            return out
    text = " ".join(
        [
            str(p.get("district") or ""),
            str(p.get("address") or ""),
            str(p.get("city") or ""),
        ]
    )
    s = _fold(text)
    if not s:
        return set()
    known = (
        "hai chau",
        "son tra",
        "ngu hanh son",
        "thanh khe",
        "hoa vang",
        "cam le",
        "lien chieu",
    )
    return {k for k in known if k in s}


def _target_city_keys(query: str, fallback_city: str = "") -> List[str]:
    keys = _ordered_city_keys_from_query(query)
    fallback_key = _city_key_from_text(fallback_city)
    if not keys and fallback_key:
        keys.append(fallback_key)
    seen: set[str] = set()
    ordered: List[str] = []
    for key in keys:
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _ordered_city_keys_from_query(query: str) -> List[str]:
    q = _fold(query or "")
    mentions: List[tuple[int, str]] = []
    for city_key, variants in _CITY_NAME_PATTERNS.items():
        positions = [q.find(variant) for variant in variants if variant in q]
        if positions:
            mentions.append((min(positions), city_key))
    mentions.sort()
    ordered: List[str] = []
    seen: set[str] = set()
    for _, city_key in mentions:
        if city_key in seen:
            continue
        seen.add(city_key)
        ordered.append(city_key)
    return ordered


def _planned_city_keys_per_day(query: str, total_days: int, fallback_city_key: str = "") -> List[str]:
    if total_days <= 0:
        return []
    explicit_segments = _best_explicit_city_segments(query=query, total_days=total_days)
    if explicit_segments:
        plan: List[str] = []
        for _, city_key, count in explicit_segments:
            plan.extend([city_key] * max(0, count))
            if len(plan) >= total_days:
                return plan[:total_days]
        fill_city = plan[-1] if plan else fallback_city_key
        if fill_city:
            plan.extend([fill_city] * max(0, total_days - len(plan)))
        return plan[:total_days]

    ordered_cities = _ordered_city_keys_from_query(query)
    if not ordered_cities:
        ordered_cities = [fallback_city_key] if fallback_city_key else []
    if not ordered_cities:
        return []
    if len(ordered_cities) == 1:
        return [ordered_cities[0]] * total_days

    base = total_days // len(ordered_cities)
    remainder = total_days % len(ordered_cities)
    plan: List[str] = []
    for index, city_key in enumerate(ordered_cities):
        chunk_days = base + (1 if index < remainder else 0)
        plan.extend([city_key] * max(1, chunk_days))
    return plan[:total_days]


def _best_explicit_city_segments(query: str, total_days: int) -> List[tuple[int, str, int]]:
    raw_segments = _explicit_city_day_segments(query)
    if not raw_segments:
        return []
    best: List[tuple[int, str, int]] = []
    best_score: tuple[int, int, int, int] | None = None
    n = len(raw_segments)
    for mask in range(1, 1 << n):
        subset = [raw_segments[i] for i in range(n) if mask & (1 << i)]
        total = sum(count for _, _, count in subset)
        unique_cities = len({city_key for _, city_key, _ in subset})
        if total <= total_days:
            score = (2, total, unique_cities, len(subset))
        else:
            score = (1, -(total - total_days), unique_cities, len(subset))
        if best_score is None or score > best_score:
            best_score = score
            best = subset
    return best


def _explicit_city_day_segments(query: str) -> List[tuple[int, str, int]]:
    q = _fold(query or "")
    found: List[tuple[int, str, int]] = []
    seen: set[tuple[int, str, int]] = set()
    for city_key, variants in _CITY_NAME_PATTERNS.items():
        city_pattern = "(?:" + "|".join(re.escape(variant) for variant in variants) + ")"
        before_city = re.finditer(
            rf"(\d+)\s*ngay(?:\s+(?:o|tai|di|tham quan|du lich))?(?:\s+\w+){{0,2}}\s*{city_pattern}",
            q,
        )
        after_city = re.finditer(
            rf"{city_pattern}(?:\s+\w+){{0,2}}\s*(\d+)\s*ngay",
            q,
        )
        for match in list(before_city) + list(after_city):
            try:
                count = max(1, min(int(match.group(1)), 7))
            except Exception:
                continue
            item = (match.start(), city_key, count)
            if item in seen:
                continue
            seen.add(item)
            found.append(item)
    found.sort()
    return found


def _place_city_key(p: dict) -> str:
    explicit = str(p.get("city_key") or "").strip().lower()
    if explicit:
        return explicit
    primary_area = str(p.get("primary_area_key") or "").strip().lower()
    if primary_area in _AREA_TO_CITY_KEY:
        return _AREA_TO_CITY_KEY[primary_area]
    for area in _extract_admin_areas(p):
        if area in _AREA_TO_CITY_KEY:
            return _AREA_TO_CITY_KEY[area]
    city = str(p.get("city") or "")
    addr = str(p.get("address") or "")
    blob = _fold(f"{city} {addr}")
    return _city_key_from_text(blob) or _city_key_from_text(city)


def _city_key_from_text(text: str) -> str:
    t = _fold(text or "")
    for city_key, variants in _CITY_NAME_PATTERNS.items():
        if any(variant in t for variant in variants):
            return city_key
    for area_key, city_key in _AREA_TO_CITY_KEY.items():
        if area_key in t:
            return city_key
    return ""


def _day_city_key(morning: dict | None, afternoon: dict | None, fallback: str = "") -> str:
    morning_key = _place_city_key(morning) if morning else ""
    afternoon_key = _place_city_key(afternoon) if afternoon else ""
    if morning_key and afternoon_key and morning_key == afternoon_key:
        return morning_key
    if morning_key:
        return morning_key
    if afternoon_key:
        return afternoon_key
    return fallback


def _select_stay_plan(
    daily_frames: List[dict],
    hotels: List[dict],
    city: str,
    strict_mode: bool = False,
    budget_tier: str = "",
) -> dict:
    if not daily_frames:
        return {"segments": [], "daily": [], "change_hotel": False}

    full_segment = {
        "city_key": "",
        "days": [int(frame["day"]) for frame in daily_frames],
        "anchors": [
            anchor
            for frame in daily_frames
            for anchor in (frame.get("morning"), frame.get("afternoon"))
            if anchor
        ],
    }
    main_hotel = _select_segment_hotel(
        segment=full_segment,
        hotels=hotels,
        city=city,
        strict_mode=strict_mode,
        budget_tier=budget_tier,
    )

    daily: List[dict] = []
    current_end_hotel = main_hotel
    relocation_happened = False

    for frame in daily_frames:
        day = int(frame["day"])
        city_key = str(frame.get("city_key") or "")
        anchors = [frame.get("morning"), frame.get("afternoon")]
        start_hotel = current_end_hotel or main_hotel
        needs_relocation = _should_relocate_for_day(start_hotel=start_hotel, anchors=anchors)
        end_hotel = start_hotel
        reason = "main_base"

        if needs_relocation:
            candidate_hotel = _select_segment_hotel(
                segment={
                    "city_key": city_key,
                    "days": [day],
                    "anchors": [anchor for anchor in anchors if anchor],
                },
                hotels=hotels,
                city=city,
                strict_mode=strict_mode,
                budget_tier=budget_tier,
            )
            if candidate_hotel and not _same_hotel(candidate_hotel, start_hotel):
                end_hotel = candidate_hotel
                reason = "relocated_near_far_cluster"
                relocation_happened = True

        daily.append(
            {
                "day": day,
                "city_key": city_key,
                "start_hotel": start_hotel,
                "end_hotel": end_hotel,
                "reason": reason,
            }
        )
        current_end_hotel = end_hotel or current_end_hotel

    if relocation_happened and daily and main_hotel:
        daily[-1]["end_hotel"] = main_hotel
        daily[-1]["reason"] = "return_to_main_hotel"

    segments: List[dict] = []
    for item in daily:
        day = int(item["day"])
        hotel = item.get("end_hotel")
        city_key = str(item.get("city_key") or "")
        if not hotel:
            continue
        if not segments or not _same_hotel(segments[-1].get("hotel"), hotel):
            segments.append(
                {
                    "city_key": city_key,
                    "days": [day],
                    "hotel": hotel,
                    "reason": str(item.get("reason") or ""),
                }
            )
        else:
            segments[-1]["days"].append(day)

    for segment in segments:
        segment["city_label"] = _city_label_from_key(str(segment.get("city_key") or ""), default_city=city)
        segment["days_label"] = _days_label(segment.get("days", []))

    return {
        "segments": segments,
        "daily": daily,
        "main_hotel": main_hotel,
        "change_hotel": any(
            not _same_hotel(item.get("start_hotel"), item.get("end_hotel"))
            for item in daily
        ),
    }


def _should_relocate_for_day(
    *,
    start_hotel: dict | None,
    anchors: List[dict | None],
) -> bool:
    if not start_hotel:
        return False
    distances = _hotel_to_anchor_distances_km(start_hotel, anchors)
    if not distances:
        return False
    return min(distances) > _HOTEL_RELOCATION_THRESHOLD_KM


def _hotel_to_anchor_distances_km(hotel: dict | None, anchors: List[dict | None]) -> List[float]:
    hotel_pt = _resolve_point_for_map(hotel)
    if not hotel_pt:
        return []
    distances: List[float] = []
    for anchor in anchors:
        anchor_pt = _resolve_point_for_map(anchor)
        if not anchor_pt:
            continue
        distances.append(_haversine_km(hotel_pt[0], hotel_pt[1], anchor_pt[0], anchor_pt[1]))
    return distances


def _same_hotel(a: dict | None, b: dict | None) -> bool:
    a_name = _fold(str((a or {}).get("name") or ""))
    b_name = _fold(str((b or {}).get("name") or ""))
    return bool(a_name and b_name and a_name == b_name)


def _select_segment_hotel(
    segment: dict,
    hotels: List[dict],
    city: str,
    strict_mode: bool = False,
    budget_tier: str = "",
) -> dict | None:
    anchors = [anchor for anchor in segment.get("anchors", []) if anchor]
    city_key = str(segment.get("city_key") or "")
    local_candidates = [
        hotel for hotel in _unique_by_name(hotels)
        if _place_key(hotel)
        and (not city_key or _place_city_key(hotel) == city_key)
    ]

    best_local: dict | None = None
    best_local_score = float("-inf")
    if local_candidates:
        ranked = sorted(
            local_candidates,
            key=lambda hotel: (
                _hotel_segment_score(hotel, anchors, budget_tier=budget_tier),
                _hotel_star_bonus(hotel),
                _place_key(hotel),
            ),
            reverse=True,
        )
        best_local = ranked[0]
        best_local_score = _hotel_segment_score(best_local, anchors, budget_tier=budget_tier)

    should_try_external = best_local is None or best_local_score < 24.0
    external = _query_external_accommodations(
        city_label=_city_label_from_key(city_key, default_city=city),
        anchors=anchors,
        city_key=city_key,
        need=8 if strict_mode else 5,
        budget_tier=budget_tier,
    ) if should_try_external else []
    if external:
        ranked_external = sorted(
            external,
            key=lambda hotel: (
                _hotel_segment_score(hotel, anchors, budget_tier=budget_tier),
                _hotel_star_bonus(hotel),
                _place_key(hotel),
            ),
            reverse=True,
        )
        best_external = ranked_external[0]
        best_external_score = _hotel_segment_score(best_external, anchors, budget_tier=budget_tier)
        if best_local is None or best_external_score > best_local_score + 1.0:
            return _with_pick_reason(best_external, f"stay_external:{_days_label(segment.get('days', []))}")

    if best_local is not None:
        return _with_pick_reason(best_local, f"stay_db:{_days_label(segment.get('days', []))}")
    return None


def _hotel_segment_score(hotel: dict, anchors: List[dict], budget_tier: str = "") -> float:
    star_bonus = _hotel_star_bonus(hotel)
    type_bonus = _hotel_type_bonus(hotel)
    budget_bonus = _hotel_budget_bonus(hotel, budget_tier)
    dominant_area = _dominant_area_key(anchors)
    dominant_area_bonus = 10.0 if dominant_area and _primary_district_key(hotel) == dominant_area else 0.0
    hotel_pt = _resolve_point_for_map(hotel)
    anchor_pts = [_resolve_point_for_map(anchor) for anchor in anchors if anchor]
    anchor_pts = [point for point in anchor_pts if point]
    if hotel_pt and anchor_pts:
        avg_km = sum(_haversine_km(hotel_pt[0], hotel_pt[1], pt[0], pt[1]) for pt in anchor_pts) / max(len(anchor_pts), 1)
        max_km = max(_haversine_km(hotel_pt[0], hotel_pt[1], pt[0], pt[1]) for pt in anchor_pts)
    else:
        avg_km = 6.0
        max_km = 8.0
    return 32.0 + dominant_area_bonus + star_bonus + type_bonus + budget_bonus - 2.2 * avg_km - 0.8 * max_km


def _hotel_budget_bonus(hotel: dict, budget_tier: str) -> float:
    """Adjust hotel score based on budget tier. Returns 0 when budget_tier is empty (no change)."""
    if not budget_tier:
        return 0.0
    stars = 0.0
    text = _fold(str(hotel.get("star_rating") or ""))
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if m:
        try:
            stars = float(m.group(1))
        except Exception:
            pass
    hotel_text = _fold(" ".join([
        str(hotel.get("name") or ""),
        str(hotel.get("accommodation_type") or ""),
    ]))
    is_budget_type = any(t in hotel_text for t in ("hostel", "homestay", "mini", "nha nghi", "guest house"))
    is_luxury_type = any(t in hotel_text for t in ("resort", "intercontinental", "hyatt", "marriott", "hilton", "sheraton", "sofitel", "pullman"))

    if budget_tier == "low":
        # Prefer budget types; penalize high stars
        bonus = 5.0 if is_budget_type else 0.0
        bonus -= max(0.0, stars - 2.0) * 3.0
        return bonus
    if budget_tier == "high":
        # Prefer 4-5 star and luxury brands; penalize low stars
        bonus = 6.0 if is_luxury_type else 0.0
        bonus += max(0.0, stars - 3.0) * 4.0
        bonus -= max(0.0, 3.0 - stars) * 2.0
        return bonus
    return 0.0


def _hotel_star_bonus(hotel: dict) -> float:
    text = _fold(str(hotel.get("star_rating") or ""))
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return 0.0
    try:
        return float(match.group(1)) * 1.5
    except Exception:
        return 0.0


def _hotel_type_bonus(hotel: dict) -> float:
    text = _fold(
        " ".join(
            [
                str(hotel.get("accommodation_type") or ""),
                str(hotel.get("name") or ""),
            ]
        )
    )
    if any(token in text for token in ("resort", "villa", "hotel", "khach san")):
        return 2.5
    if any(token in text for token in ("hostel", "homestay", "guest house", "guesthouse")):
        return 1.0
    return 0.0


def _dominant_area_key(anchors: List[dict]) -> str:
    counts: dict[str, int] = {}
    for anchor in anchors:
        if not anchor:
            continue
        area = _primary_district_key(anchor)
        if not area:
            continue
        counts[area] = counts.get(area, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: item[1])[0]


def cache_external_places(places: List[dict]) -> None:
    pass


def _query_external_accommodations(
    city_label: str,
    anchors: List[dict],
    city_key: str = "",
    need: int = 5,
    budget_tier: str = "",
) -> List[dict]:
    queries = _external_hotel_queries(city_label=city_label, anchors=anchors, budget_tier=budget_tier)
    out: List[dict] = []
    seen: set[str] = set()
    for query in queries:
        raw = search_places(query, source_kind="accommodations", top_k=max(20, need * 6))
        for item in raw:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            folded_name = _fold(name)
            if folded_name in seen or not _looks_like_accommodation(item):
                continue
            candidate = {
                "name": name,
                "category": "accommodation",
                "address": str(item.get("address") or ""),
                "city": city_label,
                "district": "",
                "lat": item.get("lat"),
                "lon": item.get("lon"),
                "source": "nominatim-fallback-accommodation",
                "osm_class": str(item.get("osm_class") or ""),
                "osm_type": str(item.get("osm_type") or ""),
                "query_used": query,
                "accommodation_type": "Khach san",
                "city_key": city_key or _city_key_from_text(city_label),
                "admin_area_keys": [],
            }
            if city_key and _place_city_key(candidate) and _place_city_key(candidate) != city_key:
                continue
            out.append(candidate)
            seen.add(folded_name)
            if len(out) >= need:
                cache_external_places(out)
                return out
    cache_external_places(out)
    return out


def _external_hotel_queries(city_label: str, anchors: List[dict], budget_tier: str = "") -> List[str]:
    locations: List[str] = []
    for anchor in anchors:
        if not anchor:
            continue
        for area_key in sorted(_extract_admin_areas(anchor)):
            label = _AREA_SEARCH_LABELS.get(area_key)
            if label:
                locations.append(label)
        district = str(anchor.get("district") or "").strip()
        city = str(anchor.get("city") or "").strip()
        if district and city:
            locations.append(f"{district}, {city}")
    if city_label:
        locations.append(city_label)

    deduped: List[str] = []
    seen: set[str] = set()
    for location in locations:
        key = _fold(location)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(location)

    if budget_tier == "high":
        type_terms = ["resort", "luxury hotel", "5 star hotel", "4 star hotel"]
    elif budget_tier == "low":
        type_terms = ["hostel", "mini hotel", "budget hotel", "guesthouse", "nha nghi"]
    else:
        type_terms = ["hotel", "khach san", "resort", "accommodation"]

    queries: List[str] = []
    for location in deduped:
        for term in type_terms:
            queries.append(f"{term} in {location}")
    return queries


def _looks_like_accommodation(item: dict) -> bool:
    osm_class = _fold(str(item.get("osm_class") or ""))
    osm_type = _fold(str(item.get("osm_type") or ""))
    name = _fold(str(item.get("name") or ""))
    return (
        osm_class == "tourism"
        and osm_type in {"hotel", "hostel", "guest_house", "motel", "apartment", "resort"}
    ) or any(token in name for token in ("hotel", "resort", "hostel", "villa", "homestay", "khach san"))


def _city_label_from_key(city_key: str, default_city: str = "") -> str:
    return "Da Nang"


def _days_label(days: List[int]) -> str:
    if not days:
        return ""
    if len(days) == 1:
        return f"Ngày {days[0]}"
    return f"Ngày {days[0]}-{days[-1]}"


def _stay_segment_line(segment: dict) -> str:
    hotel = segment.get("hotel")
    if hotel:
        return f"• {segment.get('days_label')}: nghỉ tại {_fmt(hotel)}"
    return f"• {segment.get('days_label')}: chưa tìm được khách sạn phù hợp"


def _recommended_hotel_from_stay_plan(stay_plan: dict) -> dict | None:
    segments = stay_plan.get("segments", [])
    if not segments:
        return None
    if len(segments) == 1:
        hotel = dict(segments[0].get("hotel") or {})
        if hotel:
            hotel["reason"] = str(segments[0].get("reason") or "")
        return hotel or None
    return {
        "type": "multi_city_stay",
        "reason": "Switching hotels per day cluster more than 30km apart to optimize movement.",
        "segments": [
            {
                "days": segment.get("days", []),
                "city_key": segment.get("city_key"),
                "city_label": segment.get("city_label"),
                "hotel": segment.get("hotel"),
                "reason": segment.get("reason"),
            }
            for segment in segments
        ],
    }


def _extract_city(query: str, places: List[dict]) -> str:
    target_city_keys = _target_city_keys(query)
    if len(target_city_keys) > 1:
        return " / ".join(_city_label_from_key(city_key, default_city=city_key.replace("_", " ").title()) for city_key in target_city_keys)
    if len(target_city_keys) == 1:
        return _city_label_from_key(target_city_keys[0], default_city=target_city_keys[0].replace("_", " ").title())
    for p in places:
        city_key = _place_city_key(p)
        if city_key:
            return _city_label_from_key(city_key, default_city=city_key.replace("_", " ").title())
        city = str(p.get("city") or "").strip()
        if city:
            return city
    return "Da Nang"


def _fmt(p: dict | None) -> str:
    if not p:
        return "—"
    name = str(p.get("name") or "").strip()
    address = str(p.get("address") or "").strip()
    map_url = (
        str(p.get("map_place_uri") or "").strip()
        or str(p.get("google_maps_uri") or "").strip()
        or _shared_place_map_url(p)
    )
    map_bit = f" — Map: {map_url}" if map_url else ""
    if address:
        return f"{name} ({address}){map_bit}"
    return f"{name}{map_bit}".strip() or "—"


def _with_pick_reason(p: dict | None, reason: str) -> dict | None:
    if not p:
        return p
    out = dict(p)
    out["selection_reason"] = reason
    return out


def _segment_map_url(a: dict | None, b: dict | None, engine: str = "fossgis_osrm_car") -> str:
    return _shared_segment_map_url(a, b, engine=engine)


def _build_day_route_plan(
    *,
    day: int,
    start_hotel: dict | None,
    end_hotel: dict | None,
    breakfast: dict | None,
    morning: dict | None,
    lunch: dict | None,
    afternoon: dict | None,
    dinner: dict | None,
) -> List[dict]:
    route_plan: List[dict] = []
    legs = [
        ("Departure", start_hotel, breakfast),
        ("Morning", breakfast, morning),
        ("Noon", morning, lunch),
        ("Afternoon", lunch, afternoon),
        ("Evening", afternoon, dinner),
        ("Return to hotel", dinner, end_hotel),
    ]
    for sequence, (leg_label, origin, destination) in enumerate(legs, start=1):
        leg = _build_route_leg(
            day=day,
            sequence=sequence,
            leg_label=leg_label,
            origin=origin,
            destination=destination,
        )
        if leg is not None:
            route_plan.append(leg)
    return route_plan


def _build_route_leg(
    *,
    day: int,
    sequence: int,
    leg_label: str,
    origin: dict | None,
    destination: dict | None,
) -> dict | None:
    if not origin or not destination:
        return None
    from_name = str(origin.get("name") or "").strip()
    to_name = str(destination.get("name") or "").strip()
    if not from_name or not to_name or from_name == to_name:
        return None

    payload: dict[str, object] = {
        "day": day,
        "day_label": f"Ngày {day}",
        "sequence": sequence,
        "leg_label": leg_label,
        "from": from_name,
        "to": to_name,
        "segment_map_url": _segment_map_url(origin, destination),
    }

    a_pt, b_pt = _resolve_segment_points(origin, destination)
    if not a_pt or not b_pt:
        return payload

    a_lat, a_lon = a_pt
    b_lat, b_lon = b_pt
    origin_point = GeoPoint(lat=a_lat, lon=a_lon)
    destination_point = GeoPoint(lat=b_lat, lon=b_lon)
    fastest = _fastest_route_by_map(origin_point, destination_point)
    same_place = _fold(from_name) == _fold(to_name)
    exact_same_point = a_lat == b_lat and a_lon == b_lon

    if fastest:
        shown_km = float(fastest["distance_km"])
        if not same_place and shown_km < 0.2:
            shown_km = 0.2
        if exact_same_point and same_place:
            shown_km = 0.0
        payload.update(
            {
                "distance_km": round(shown_km, 2),
                "eta_min": max(3, int(fastest["eta_min"])),
                "recommended_mode": str(fastest.get("mode") or ""),
                "mode_label": str(fastest.get("mode_label") or ""),
                "routing_source": str(fastest.get("source") or "trackasia"),
            }
        )
        return payload

    km = _haversine_km(a_lat, a_lon, b_lat, b_lon)
    if not same_place and km < 0.2:
        km = 0.2
    eta_min = max(5, int(round(km / 28 * 60)))
    mode_label = "di bo" if km <= 1.0 else "Grab hoac oto"
    payload.update(
        {
            "distance_km": round(km, 2),
            "eta_min": eta_min,
            "recommended_mode": "pedestrian" if km <= 1.0 else "car",
            "mode_label": mode_label,
            "routing_source": "haversine_estimate",
        }
    )
    return payload


def _resolve_point_for_map(p: dict | None) -> tuple[float, float] | None:
    return _shared_resolve_point_for_map(p)


def _resolve_segment_points(
    a: dict | None,
    b: dict | None,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    return _shared_resolve_segment_points(a, b)


def _fastest_route_by_map(origin: GeoPoint, destination: GeoPoint) -> dict | None:
    mode_labels = {
        "car": "ô tô / Grab",
        "truck": "xe tải",
        "scooter": "xe máy",
        "pedestrian": "đi bộ",
    }
    best: dict | None = None
    for mode in configured_route_modes():
        est = estimate_route(origin, destination, travel_mode=mode)
        if not est:
            continue
        cand = {
            "mode": mode,
            "mode_label": mode_labels.get(mode, mode),
            "eta_min": max(1, int(round(est.travel_time_s / 60))),
            "distance_km": float(est.distance_m) / 1000,
            "source": "trackasia",
        }
        if best is None or cand["eta_min"] < best["eta_min"]:
            best = cand
    return best


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return _shared_haversine_km(lat1, lon1, lat2, lon2)


