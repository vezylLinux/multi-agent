from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from time import perf_counter
from typing import Any

from app.graph.intake import evaluate_intake, Subtask, intake_and_decompose
from app.graph.retrieval import retrieve_for_subtasks
from app.graph.state import TravelGraphState
from app.itinerary.validation import (
    build_retry_query,
    should_retry_itinerary,
    validate_itinerary_plan,
)
from app.places.rag import (
    build_context_payload,
    build_coordinator_output,
    build_grounded_answer,
    build_itinerary_artifacts,
    build_research_output,
    build_source_artifacts,
    retrieve_trip_artifacts,
)
from app.itinerary.formatter import (
    build_time_confirmation_question,
    format_planning_answer,
)
from app.places.metadata import extract_trip_days
from app.itinerary.stay import build_stay_recommendations

_INTAKE_FOLLOW_UP_ANSWER = (
    "Chưa đủ thông tin để tạo lịch trình tốt nhất cho bạn. "
    "Vui lòng trả lời các câu hỏi bổ sung bên dưới để chúng tôi hiểu rõ hơn về nhu cầu của bạn."
)

_TRACE_TITLE_MAP = {
    "intake_agent": "Intake Agent (A1)",
    "retrieval_agent": "Retrieval Agent (A3)",
    "planning_agent": "Plan Generation Agent (A4)",
    "validator_agent": "Validator Agent (A5)",
    "response_service": "Response Service",
    "hybrid_vector_rag": "Hybrid Vector + Chroma RAG",
    "db_only_local_sufficient": "Local Catalog Sufficient",
    "db_only_local_limited_results": "Local Catalog Limited",
    "db_only_not_enough_attractions": "Missing Attraction Coverage",
    "db_only_attractions_ready": "Attractions Ready",
    "planning_retry_strict": "Strict Replan",
    "context_builder_agent": "Context Builder Agent",
    "research_agent": "Research Agent",
    "itinerary_builder": "Itinerary Builder",
    "itinerary_validator": "Itinerary Validator",
    "llm_agent": "LLM Answer Agent",
    "coordinator_agent": "Coordinator Agent",
}


def _append_trace(state: TravelGraphState, *items: str) -> list[str]:
    trace = list(state.get("trace", []))
    trace.extend(items)
    return trace


def _copy_timings(state: TravelGraphState) -> dict[str, Any]:
    return dict(state.get("timings") or {})


_BUDGET_LOW_KW = ("budget", "low", "tiet kiem", "re", "backpacker", "hostel", "gia re", "tiet", "binh dan", "nha nghi")
_BUDGET_HIGH_KW = ("luxury", "high", "cao cap", "sang trong", "4 sao", "5 sao", "resort", "4-5 sao", "cao cấp", "sang trọng")


def _infer_budget_tier(budget_str: str) -> str:
    from app.places.metadata import fold_text
    text = fold_text(budget_str)
    if any(kw in text for kw in _BUDGET_LOW_KW):
        return "low"
    if any(kw in text for kw in _BUDGET_HIGH_KW):
        return "high"
    return ""


def _should_generate_plan(state: TravelGraphState) -> bool:
    return bool(state.get("with_plan")) and not state.get("category")


def _reconcile_days(collected: dict[str, Any], message: str) -> dict[str, Any]:
    # When the source text has an explicit 'X ngày' / 'X days', use that as source
    # of truth for days. The LLM tends to anchor on the 'Y đêm' / 'Y nights' part
    # of 'X ngày Y đêm' and returns the wrong number; this keeps collected_info.days
    # consistent with extract_trip_days() used by the validator and itinerary builder.
    regex_days = extract_trip_days(message, default=None)
    if regex_days is not None:
        collected = dict(collected)
        collected["days"] = str(regex_days)
    return collected


def intake_node(state: TravelGraphState) -> dict[str, Any]:
    started = perf_counter()
    message = state.get("message", "")
    combined = intake_and_decompose(message)
    timings = _copy_timings(state)
    timings["intake_ms"] = round((perf_counter() - started) * 1000, 1)
    if combined is not None:
        result = {
            "intake_complete": combined.is_complete,
            "collected_info": _reconcile_days(combined.collected, message),
            "missing_fields": combined.missing_fields,
            "follow_up_questions": combined.follow_up_questions,
            "trace": _append_trace(state, "intake_agent"),
            "timings": timings,
        }
        if combined.subtasks:
            result["subtasks"] = [
                {"category": s.category, "area": s.area, "description": s.description}
                for s in combined.subtasks
            ]
        return result
    intake = evaluate_intake(message)
    return {
        "intake_complete": intake.is_complete,
        "collected_info": _reconcile_days(intake.collected, message),
        "missing_fields": intake.missing_fields,
        "follow_up_questions": intake.follow_up_questions,
        "trace": _append_trace(state, "intake_agent"),
        "timings": timings,
    }



def retrieval_node(state: TravelGraphState) -> dict[str, Any]:
    started = perf_counter()
    timings = _copy_timings(state)
    subtasks_raw = state.get("subtasks") or []
    subtasks = [
        Subtask(
            category=s["category"],
            area=s["area"],
            description=s["description"],
        )
        for s in subtasks_raw
        if isinstance(s, dict)
    ]
    result = retrieve_for_subtasks(
        state.get("collected_info") or {},
        subtasks,
        top_k=int(state.get("top_k", 5) or 5),
        with_plan=bool(state.get("with_plan", False)),
    )
    timings["retrieval_ms"] = round((perf_counter() - started) * 1000, 1)
    return {
        "places": result.places,
        "local_candidates_considered": result.local_candidates_considered,
        "guide": result.guide,
        "transport": result.transport,
        "recommended_hotel": result.recommended_hotel,
        "mobility_plan": result.mobility_plan,
        "trace": _append_trace(state, *result.trace, "retrieval_agent"),
        "timings": timings,
    }


def planning_node(state: TravelGraphState) -> dict[str, Any]:
    total_started = perf_counter()
    timings = _copy_timings(state)

    step_started = perf_counter()
    collected = state.get("collected_info") or {}
    q_extra = " ".join(str(collected.get(k) or "") for k in ("destination", "days", "interests")).strip()
    budget_tier = _infer_budget_tier(str(collected.get("budget") or ""))
    base_message = (state.get("message") or "").strip()
    rag_query = f"{q_extra} {base_message}".strip() if q_extra else base_message
    timings["planning_prepare_query_ms"] = round((perf_counter() - step_started) * 1000, 1)
    retry_query = str(state.get("retry_query") or "").strip()

    # Places and context come from retrieval_node (A3) via state.
    places = list(state.get("places", []) or [])
    local_candidates_considered = int(state.get("local_candidates_considered", 0) or 0)
    guide = str(state.get("guide") or "")
    transport = state.get("transport")
    recommended_hotel = state.get("recommended_hotel")
    mobility_plan = state.get("mobility_plan")
    planning_trace = ["planning_agent"]

    if retry_query:
        planning_trace.append("planning_retry_strict")
    timings["planning_retrieval_ms"] = 0.0

    step_started = perf_counter()
    context = build_context_payload(
        query=rag_query,
        places=places,
        transport=transport,
        recommended_hotel=recommended_hotel,
        mobility_plan=mobility_plan,
        guide=guide,
    )
    timings["planning_context_ms"] = round((perf_counter() - step_started) * 1000, 1)

    research = None
    plan = None
    stay_plan = None
    coordinator_plan = None
    route_plan = None
    itinerary_query = rag_query

    if _should_generate_plan(state):
        step_started = perf_counter()
        research = build_research_output(
            query=rag_query,
            places=places,
            transport=transport,
        )
        timings["planning_research_ms"] = round((perf_counter() - step_started) * 1000, 1)
        itinerary_query = retry_query or rag_query
        step_started = perf_counter()
        itinerary = build_itinerary_artifacts(
            query=itinerary_query,
            places=places,
            strict_mode=bool(retry_query),
            budget_tier=budget_tier,
        )
        timings["planning_itinerary_ms"] = round((perf_counter() - step_started) * 1000, 1)
        plan = itinerary.get("plan")
        stay_plan = itinerary.get("stay_plan")
        route_plan = itinerary.get("route_plan")
        recommended_hotel = itinerary.get("recommended_hotel") or recommended_hotel
        all_places_for_validation = itinerary.get("all_places") or places
        step_started = perf_counter()
        coordinator_plan = build_coordinator_output(
            query=rag_query,
            itinerary=str(plan or ""),
            transport=transport,
        )
        timings["planning_coordinator_ms"] = round((perf_counter() - step_started) * 1000, 1)
    else:
        timings["planning_research_ms"] = 0.0
        timings["planning_itinerary_ms"] = 0.0
        timings["planning_coordinator_ms"] = 0.0

    step_started = perf_counter()
    source_artifacts = build_source_artifacts(
        places=places,
        local_candidates_considered=local_candidates_considered,
    )
    timings["planning_sources_ms"] = round((perf_counter() - step_started) * 1000, 1)
    answer = ""
    if not _should_generate_plan(state):
        step_started = perf_counter()
        answer = build_grounded_answer(
            query=rag_query,
            context=context,
            verified_places=source_artifacts.verified_places,
        )
        timings["planning_answer_ms"] = round((perf_counter() - step_started) * 1000, 1)
    else:
        timings["planning_answer_ms"] = 0.0
    timings["planning_total_ms"] = round((perf_counter() - total_started) * 1000, 1)

    return {
        "rag_query": rag_query,
        "planning_query": itinerary_query,
        "places": places,
        "local_candidates_considered": local_candidates_considered,
        "guide": guide,
        "transport": transport,
        "recommended_hotel": recommended_hotel,
        "mobility_plan": mobility_plan,
        "context": context,
        "research": research,
        "plan": plan,
        "stay_plan": stay_plan,
        "answer": answer,
        "coordinator_plan": coordinator_plan,
        "sources": source_artifacts.sources,
        "verified_places": source_artifacts.verified_places,
        "route_plan": route_plan or source_artifacts.route_plan,
        "grounding": source_artifacts.grounding,
        "all_places_for_validation": all_places_for_validation if _should_generate_plan(state) else places,
        "retry_query": None,
        "needs_replan": False,
        "trace": _append_trace(state, *planning_trace),
        "timings": timings,
    }


def validator_node(state: TravelGraphState) -> dict[str, Any]:
    started = perf_counter()
    timings = _copy_timings(state)
    if not _should_generate_plan(state):
        timings["validator_ms"] = round((perf_counter() - started) * 1000, 1)
        return {
            "plan_validation": None,
            "needs_replan": False,
            "retry_query": None,
            "trace": _append_trace(state, "validator_agent"),
            "timings": timings,
        }

    query = state.get("rag_query", state.get("message", ""))
    retry_attempted = bool(state.get("itinerary_retry_attempted", False))
    collected_info = state.get("collected_info") or {}
    raw_interests = str(collected_info.get("interests") or "")
    interests = {token.strip().lower() for token in raw_interests.split(",") if token.strip()}
    validation = validate_itinerary_plan(
        query=query,
        plan=str(state.get("plan", "") or ""),
        places=state.get("all_places_for_validation") or state.get("places", []),
        interests=interests,
        route_plan=state.get("route_plan") or [],
    )
    needs_replan = should_retry_itinerary(
        validation,
        retry_attempted=retry_attempted,
    )
    validation["retried"] = retry_attempted

    if needs_replan:
        retry_query = build_retry_query(query, validation)
        validation["retry_query"] = retry_query
        issues = validation.get("issues") or []
        augmented_places = _augment_places_for_retry(
            state.get("places") or [],
            issues,
        )
        if "too_many_long_legs" in issues or "extreme_leg_distance" in issues:
            augmented_places = _filter_places_by_distance(
                augmented_places,
                state.get("recommended_hotel"),
                max_km=20.0,
            )
        timings["validator_ms"] = round((perf_counter() - started) * 1000, 1)
        return {
            "plan_validation": validation,
            "needs_replan": True,
            "retry_query": retry_query,
            "itinerary_retry_attempted": True,
            "places": augmented_places,
            "trace": _append_trace(state, "validator_agent"),
            "timings": timings,
        }

    timings["validator_ms"] = round((perf_counter() - started) * 1000, 1)
    return {
        "plan_validation": validation,
        "needs_replan": False,
        "retry_query": None,
        "itinerary_retry_attempted": retry_attempted,
        "trace": _append_trace(state, "validator_agent"),
        "timings": timings,
    }


def clarify_response_node(state: TravelGraphState) -> dict[str, Any]:
    from app.graph.intake import generate_contextual_suggestions
    
    timings = _copy_timings(state)
    trace = _append_trace(state, "response_service")
    collected = state.get("collected_info") or {}
    follow_up_questions = list(state.get("follow_up_questions", []))
    
    # Add contextual suggestions based on collected information
    suggestions = generate_contextual_suggestions(collected)
    all_questions = follow_up_questions + suggestions
    
    response_payload = {
        "answer": _INTAKE_FOLLOW_UP_ANSWER,
        "conversation_stage": "intake",
        "collected_info": collected,
        "missing_fields": state.get("missing_fields", []),
        "follow_up_questions": all_questions,
        "trace": trace,
        "sources": [],
        "plan": None,
        "stay_plan": None,
        "plan_validation": None,
        "research": None,
        "coordinator_plan": None,
        "transport": None,
        "recommended_hotel": None,
        "mobility_plan": None,
        "verified_places": None,
        "route_plan": None,
        "grounding": None,
        "debug_steps": _build_debug_steps(state, stage="intake"),
    }
    return {
        "answer": _INTAKE_FOLLOW_UP_ANSWER,
        "conversation_stage": "intake",
        "trace": trace,
        "timings": timings,
        "response_payload": response_payload,
    }


def response_node(state: TravelGraphState) -> dict[str, Any]:
    started = perf_counter()
    timings = _copy_timings(state)
    if _should_generate_plan(state):
        follow_up_questions = [
            build_time_confirmation_question(
                str((state.get("collected_info") or {}).get("destination") or "")
            )
        ]
        stay_recommendations = build_stay_recommendations(
            query=state.get("rag_query", state.get("message", "")),
            places=state.get("places", []),
            recommended_hotel=state.get("recommended_hotel"),
        )
        formatted_answer = format_planning_answer(
            query=state.get("rag_query", state.get("message", "")),
            collected_info=state.get("collected_info"),
            research=state.get("research"),
            plan=state.get("plan"),
            coordinator_plan=state.get("coordinator_plan"),
            transport=state.get("transport"),
            recommended_hotel=state.get("recommended_hotel"),
            mobility_plan=state.get("mobility_plan"),
            stay_plan=state.get("stay_plan"),
            stay_recommendations=stay_recommendations,
            plan_validation=state.get("plan_validation"),
            verified_places=state.get("verified_places"),
        )
    else:
        follow_up_questions = []
        stay_recommendations = None
        formatted_answer = str(state.get("answer") or "").strip()
    trace = _append_trace(state, "response_service")
    timings["response_ms"] = round((perf_counter() - started) * 1000, 1)
    response_payload = {
        "answer": formatted_answer,
        "conversation_stage": "planning",
        "collected_info": state.get("collected_info"),
        "missing_fields": [],
        "follow_up_questions": follow_up_questions,
        "trace": trace,
        "sources": state.get("sources", []),
        "plan": state.get("plan"),
        "stay_plan": state.get("stay_plan"),
        "stay_recommendations": stay_recommendations,
        "plan_validation": state.get("plan_validation"),
        "research": state.get("research"),
        "coordinator_plan": state.get("coordinator_plan"),
        "transport": state.get("transport"),
        "recommended_hotel": state.get("recommended_hotel"),
        "mobility_plan": state.get("mobility_plan"),
        "verified_places": state.get("verified_places"),
        "route_plan": state.get("route_plan"),
        "grounding": _merge_grounding_with_validation(
            state.get("grounding"),
            state.get("plan_validation"),
        ),
        "debug_steps": _build_debug_steps(
            {
                **state,
                "trace": trace,
                "timings": timings,
            },
            stage="planning",
        ),
    }
    return {
        "answer": formatted_answer,
        "conversation_stage": "planning",
        "trace": trace,
        "timings": timings,
        "response_payload": response_payload,
    }


def _merge_grounding_with_validation(
    grounding: dict[str, Any] | None,
    plan_validation: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not grounding and not plan_validation:
        return grounding
    merged = dict(grounding or {})
    if plan_validation is not None:
        merged["plan_validation"] = plan_validation
    return merged


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(a))


def _filter_places_by_distance(
    places: list[dict[str, Any]],
    hotel: dict[str, Any] | None,
    max_km: float = 20.0,
) -> list[dict[str, Any]]:
    if not hotel:
        return places
    h_lat = hotel.get("lat")
    h_lon = hotel.get("lon")
    if not isinstance(h_lat, (int, float)) or not isinstance(h_lon, (int, float)):
        return places

    _NON_ATTRACTION = {"restaurant", "accommodation"}

    # Separate attractions (need filtering) from others (always kept)
    attractions = [
        (i, p) for i, p in enumerate(places)
        if str(p.get("category") or "").lower() not in _NON_ATTRACTION
        and isinstance(p.get("lat"), (int, float))
        and isinstance(p.get("lon"), (int, float))
    ]
    no_filter = [p for p in places if str(p.get("category") or "").lower() in _NON_ATTRACTION]
    no_coords = [p for p in places if not isinstance(p.get("lat"), (int, float)) or not isinstance(p.get("lon"), (int, float))]

    if not attractions:
        return places

    # Try Distance Matrix for actual road distances from hotel to each attraction
    road_distances: list[float | None] = [None] * len(attractions)
    try:
        from app.tools.trackasia import get_distance_matrix
        points = [(float(h_lat), float(h_lon))] + [
            (float(p.get("lat")), float(p.get("lon"))) for _, p in attractions
        ]
        matrix = get_distance_matrix(points, sources=[0], profile="car")
        if matrix and matrix.get("distances"):
            row = matrix["distances"][0]  # distances from hotel (index 0) to all others
            for j, dist in enumerate(row[1:]):  # skip index 0 (hotel→hotel)
                if isinstance(dist, (int, float)):
                    road_distances[j] = dist
    except Exception:
        pass

    kept_attractions: list[dict[str, Any]] = []
    for j, (_, p) in enumerate(attractions):
        road_km = road_distances[j]
        if road_km is not None:
            if road_km <= max_km:
                kept_attractions.append(p)
        else:
            # fallback to haversine when Distance Matrix unavailable
            p_lat, p_lon = float(p["lat"]), float(p["lon"])
            if _haversine(h_lat, h_lon, p_lat, p_lon) <= max_km:
                kept_attractions.append(p)

    if len(kept_attractions) < 2:
        return places
    return no_filter + no_coords + kept_attractions


def _augment_places_for_retry(
    current_places: list[dict[str, Any]],
    issues: list[str],
) -> list[dict[str, Any]]:
    existing_ids: set[str] = {
        str(p.get("place_id") or "").strip()
        for p in current_places
        if str(p.get("place_id") or "").strip()
    }
    extra: list[dict[str, Any]] = []

    def _fetch(query: str, category: str | None, top_k: int) -> None:
        result = retrieve_trip_artifacts(query=query, category=category, top_k=top_k)
        for p in result.places:
            pid = str(p.get("place_id") or "").strip()
            if pid and pid not in existing_ids:
                existing_ids.add(pid)
                extra.append(p)

    if "missing_beach_alignment" in issues:
        _fetch("beach bien my khe son tra coastal da nang", "destinations", 4)
    if "missing_culture_alignment" in issues:
        _fetch("museum heritage cultural di tich bao tang van hoa da nang", "destinations", 4)
    if "missing_food_alignment" in issues:
        _fetch("restaurant am thuc da nang", "restaurants", 4)

    return list(current_places) + extra


def route_after_intake(state: TravelGraphState) -> str:
    return "clarify" if not state.get("intake_complete") else "planning"


def route_after_validation(state: TravelGraphState) -> str:
    return "planning" if state.get("needs_replan") else "response"


def _build_debug_steps(state: TravelGraphState, *, stage: str) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    collected = dict(state.get("collected_info") or {})
    missing = list(state.get("missing_fields") or [])
    trace = [str(item).strip() for item in state.get("trace", []) if str(item).strip()]
    rag_query = str(state.get("rag_query") or state.get("message") or "").strip()
    planning_query = str(state.get("planning_query") or rag_query).strip()
    places = state.get("places", []) or []
    verification = state.get("plan_validation") or {}
    route_plan = state.get("route_plan") or []
    transport = state.get("transport") or []
    sources = state.get("sources") or []
    timings = dict(state.get("timings") or {})

    steps.append(
        {
            "key": "intake",
            "title": "1. Intake Agent",
            "status": "done" if not missing else "needs_input",
            "summary": (
                "All required inputs collected, proceeding to planning."
                if not missing
                else "Some required inputs are still missing."
            ),
            "details": {
                "collected_info": collected,
                "missing_fields": missing,
                "follow_up_questions": state.get("follow_up_questions") or [],
                "effective_request": str(state.get("message") or "").strip(),
                "timings_ms": timings,
            },
        }
    )

    if stage == "intake":
        steps.append(
            {
                "key": "planning_blocked",
                "title": "2. Planning Agent",
                "status": "waiting",
                "summary": "Waiting for user to provide destination, days, and interests.",
                "details": {
                    "required_fields": ["destination", "days", "interests"],
                    "missing_fields": missing,
                },
            }
        )
        return steps

    steps.append(
        {
            "key": "planning",
            "title": "2. Planning Agent",
            "status": "done",
            "summary": "Combined retrieval, scoring, research, itinerary, and grounding in one agent.",
            "details": {
                "rag_query": rag_query,
                "planning_query": planning_query,
                "places_found": len(places),
                "sources_ready": len(sources),
                "local_candidates_considered": int(state.get("local_candidates_considered", 0) or 0),
                "context_ready": bool(str(state.get("context") or "").strip()),
                "research_ready": bool(str(state.get("research") or "").strip()),
                "plan_ready": bool(state.get("plan")),
                "stay_plan_ready": bool(state.get("stay_plan")),
                "recommended_hotel_ready": bool(state.get("recommended_hotel")),
                "route_items": len(route_plan),
                "transport_options": list(transport[:5]),
                "retry_mode": bool(state.get("itinerary_retry_attempted")),
                "trace": [_TRACE_TITLE_MAP.get(item, item) for item in trace],
                "timings_ms": {
                    key: value
                    for key, value in timings.items()
                    if str(key).startswith("planning_")
                },
            },
        }
    )
    steps.append(
        {
            "key": "validation",
            "title": "3. Validator Agent",
            "status": (
                "waiting"
                if state.get("needs_replan")
                else ("done" if verification else "skipped")
            ),
            "summary": (
                "Validator requires planning agent to regenerate the itinerary."
                if state.get("needs_replan")
                else (
                    "Itinerary passed validation."
                    if verification and bool(verification.get("passed", True))
                    else "Validation completed and retained existing warnings."
                )
            ),
            "details": {
                "validation": verification,
                "retry_query": state.get("retry_query"),
                "retried": bool(verification.get("retried", False)),
                "timings_ms": {
                    "validator_ms": timings.get("validator_ms"),
                },
            },
        }
    )
    steps.append(
        {
            "key": "response",
            "title": "4. Response Service",
            "status": "done",
            "summary": "Formatting final output for the UI and user.",
            "details": {
                "answer_ready": bool(str(state.get("answer") or "").strip()),
                "coordinator_ready": bool(str(state.get("coordinator_plan") or "").strip()),
                "final_trace": [_TRACE_TITLE_MAP.get(item, item) for item in trace],
                "timings_ms": {
                    "response_ms": timings.get("response_ms"),
                    "intake_ms": timings.get("intake_ms"),
                },
            },
        }
    )
    return steps
