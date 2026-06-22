from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.graph.intake import Subtask
from app.places.rag import retrieve_trip_artifacts

_SUBTASK_CATEGORY_TO_SOURCE_KIND = {
    "destination": "destinations",
    "restaurant": "restaurants",
    "accommodation": "accommodations",
    "entertainment": "entertainment",
}


@dataclass
class RetrievalResult:
    places: list[dict[str, Any]] = field(default_factory=list)
    guide: str = ""
    transport: list[str] | None = None
    recommended_hotel: dict[str, Any] | None = None
    mobility_plan: dict[str, Any] | None = None
    trace: list[str] = field(default_factory=list)
    local_candidates_considered: int = 0


def retrieve_for_subtasks(
    collected_info: dict[str, Any],
    subtasks: list[Subtask],
    *,
    top_k: int = 5,
    with_plan: bool = True,
) -> RetrievalResult:
    if not subtasks:
        return _retrieve_full(collected_info, top_k=top_k, with_plan=with_plan)

    query = _build_query(collected_info)

    # Run a single full retrieval to get transport, hotel, mobility.
    base = retrieve_trip_artifacts(
        query=query,
        category=None,
        top_k=top_k,
        with_plan=with_plan,
    )

    # Augment with per-subtask targeted retrieval to cover all required categories.
    all_place_ids: set[str] = {
        str(p.get("place_id") or "").strip()
        for p in base.places
        if str(p.get("place_id") or "").strip()
    }
    extra_places: list[dict[str, Any]] = []

    for subtask in subtasks:
        source_kind = _SUBTASK_CATEGORY_TO_SOURCE_KIND.get(subtask.category)
        if not source_kind:
            continue
        subtask_query = f"{subtask.description} {query}".strip()
        result = retrieve_trip_artifacts(
            query=subtask_query,
            category=source_kind,
            top_k=max(3, top_k // len(subtasks)),
            with_plan=False,
        )
        for place in result.places:
            pid = str(place.get("place_id") or "").strip()
            if pid and pid not in all_place_ids:
                all_place_ids.add(pid)
                extra_places.append(place)

    merged_places = list(base.places) + extra_places
    merged_trace = list(base.trace)
    if extra_places:
        merged_trace.append("subtask_retrieval")

    merged_places, merged_trace = _supplement_hotels_if_needed(
        merged_places, collected_info, merged_trace
    )

    return RetrievalResult(
        places=merged_places,
        guide=base.guide,
        transport=base.transport,
        recommended_hotel=base.recommended_hotel,
        mobility_plan=base.mobility_plan,
        trace=merged_trace,
        local_candidates_considered=base.local_candidates_considered + len(extra_places),
    )


def _retrieve_full(
    collected_info: dict[str, Any],
    *,
    top_k: int,
    with_plan: bool,
) -> RetrievalResult:
    query = _build_query(collected_info)
    base = retrieve_trip_artifacts(
        query=query,
        category=None,
        top_k=top_k,
        with_plan=with_plan,
    )
    places, trace = _supplement_hotels_if_needed(list(base.places), collected_info, list(base.trace))
    return RetrievalResult(
        places=places,
        guide=base.guide,
        transport=base.transport,
        recommended_hotel=base.recommended_hotel,
        mobility_plan=base.mobility_plan,
        trace=trace,
        local_candidates_considered=base.local_candidates_considered,
    )


def _supplement_hotels_if_needed(
    places: list[dict[str, Any]],
    collected_info: dict[str, Any],
    trace: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    hotel_places = [p for p in places if str(p.get("category") or "").lower() == "accommodation"]
    if len(hotel_places) >= 2:
        return places, trace

    destination = str(collected_info.get("destination") or "Da Nang")
    extra = _search_hotels_via_trackasia(destination, top_k=5)
    if not extra:
        return places, trace

    existing_names = {str(p.get("name") or "").strip().lower() for p in places}
    added = 0
    for hotel in extra:
        name = str(hotel.get("name") or "").strip().lower()
        if name and name not in existing_names:
            existing_names.add(name)
            places.append(hotel)
            added += 1

    if added:
        trace = list(trace) + ["trackasia_hotel_fallback"]
    return places, trace


def _search_hotels_via_trackasia(destination: str, *, top_k: int = 5) -> list[dict[str, Any]]:
    from app.tools.trackasia import geocode_address

    query = "khách sạn Đà Nẵng"
    city = "Đà Nẵng"

    results = geocode_address(query, limit=top_k)
    out: list[dict[str, Any]] = []
    for r in results:
        lat = r.get("lat")
        lon = r.get("lon")
        if lat is None or lon is None:
            continue
        out.append({
            "name": str(r.get("name") or "").strip(),
            "category": "accommodation",
            "city": city,
            "address": str(r.get("address") or "").strip(),
            "lat": lat,
            "lon": lon,
            "source": "trackasia_textsearch",
            "retrieval_tier": "trackasia_fallback",
            "customer_fit_score": 30.0,
            "retrieval_relevance": 0.3,
            "verification_status": "external_search",
            "place_id": f"ta_{r.get('place_id') or ''}",
        })
    return out


def _build_query(collected_info: dict[str, Any]) -> str:
    parts = [
        str(collected_info.get("destination") or ""),
        str(collected_info.get("days") or ""),
        str(collected_info.get("interests") or ""),
    ]
    return " ".join(p for p in parts if p).strip()
