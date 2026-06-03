from __future__ import annotations

from typing import Any, List, NotRequired, TypedDict


class TravelGraphState(TypedDict):
    message: str
    top_k: int
    with_plan: bool
    category: str | None
    trace: List[str]

    intake_complete: NotRequired[bool]
    collected_info: NotRequired[dict[str, str]]
    missing_fields: NotRequired[list[str]]
    follow_up_questions: NotRequired[list[str]]
    subtasks: NotRequired[list[dict[str, str]]]
    rag_query: NotRequired[str]
    planning_query: NotRequired[str]
    retry_query: NotRequired[str | None]
    needs_replan: NotRequired[bool]
    timings: NotRequired[dict[str, Any]]

    places: NotRequired[list[dict[str, Any]]]
    all_places_for_validation: NotRequired[list[dict[str, Any]]]
    local_candidates_considered: NotRequired[int]
    weather: NotRequired[dict[str, Any] | None]
    guide: NotRequired[str]
    transport: NotRequired[list[str] | None]
    recommended_hotel: NotRequired[dict[str, Any] | None]
    mobility_plan: NotRequired[dict[str, Any] | None]
    context: NotRequired[str]
    research: NotRequired[str | None]
    plan: NotRequired[str | None]
    stay_plan: NotRequired[dict[str, Any] | None]
    stay_recommendations: NotRequired[list[dict[str, Any]] | None]
    plan_validation: NotRequired[dict[str, Any] | None]
    itinerary_retry_attempted: NotRequired[bool]
    answer: NotRequired[str]
    coordinator_plan: NotRequired[str | None]
    sources: NotRequired[list[dict[str, Any]]]
    verified_places: NotRequired[list[dict[str, Any]]]
    route_plan: NotRequired[list[dict[str, Any]]]
    grounding: NotRequired[dict[str, Any] | None]
    conversation_stage: NotRequired[str]
    response_payload: NotRequired[dict[str, Any]]
