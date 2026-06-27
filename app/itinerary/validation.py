from __future__ import annotations

import json
import time
from typing import Any

from app.places.metadata import extract_trip_days

_SOFT_ISSUES = {"too_many_self_service_meals"}

_VALID_ISSUES = {
    "daily_count_mismatch",
    "missing_daily_structure",
    "too_many_self_service_meals",
    "empty_place_pool",
    "unrealistic_schedule",
    "too_many_long_legs",
    "extreme_leg_distance",
}

_SYSTEM_PROMPT = "You are a travel itinerary quality checker. Return only valid JSON, no explanation."

_VALIDATE_TEMPLATE = """Validate this {days}-day travel plan.
Trip: destination={destination}, interests={interests}

Plan:
{plan}

Check for issues from this list (include only those that apply):
- "daily_count_mismatch": number of DAY sections != {days}
- "missing_daily_structure": any day is missing a morning or afternoon slot
- "too_many_self_service_meals": self-service meals where guests prepare their own food exceed {days}
- "empty_place_pool": the plan has no activities at all
- "unrealistic_schedule": more than 5 activities in a single day

Return JSON only:
{{"passed": true, "issues": [], "reason": "brief explanation"}}"""


def validate_itinerary_plan(
    query: str,
    plan: str,
    places: list[dict[str, Any]],
    interests: set[str] | None = None,
    route_plan: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    total_days = extract_trip_days(query) or 1
    interest_tags = interests if interests is not None else _extract_interest_tags(query)

    llm_result = _llm_validate(
        query=query,
        plan=plan or "",
        places=places or [],
        interests=interest_tags,
        total_days=total_days,
    )

    distance_issues, distance_metrics = _check_distances(route_plan or [], total_days)

    if llm_result is not None:
        merged_issues = llm_result["issues"] + [i for i in distance_issues if i not in llm_result["issues"]]
        passed = not any(i for i in merged_issues if i not in _SOFT_ISSUES)
        return {
            "passed": passed,
            "issues": merged_issues,
            "reason": llm_result.get("reason", ""),
            "metrics": {"validator": "llm", "days_expected": total_days, **distance_metrics},
        }

    return {
        "passed": not any(i for i in distance_issues if i not in _SOFT_ISSUES),
        "issues": distance_issues,
        "metrics": {"validator": "unavailable", "days_expected": total_days, **distance_metrics},
    }


def _llm_validate(
    query: str,
    plan: str,
    places: list[dict[str, Any]],
    interests: set[str],
    total_days: int,
) -> dict[str, Any] | None:
    if not plan.strip():
        return None

    from app.core.settings import get_settings
    settings = get_settings()
    openrouter_key = (settings.openrouter_api_key or "").strip()
    if not openrouter_key:
        return None

    try:
        from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
    except Exception:
        return None

    interests_str = ", ".join(sorted(interests)) if interests else "general"
    prompt = _VALIDATE_TEMPLATE.format(
        days=total_days,
        destination=_extract_destination(query),
        interests=interests_str,
        plan=plan.strip(),
    )

    client = OpenAI(
        base_url=settings.openrouter_base_url,
        api_key=openrouter_key,
        timeout=max(1, int(settings.openrouter_request_timeout_s or 25)),
    )
    transient = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
    resp = None
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=settings.openrouter_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            break
        except transient:
            if attempt >= 1:
                return None
            time.sleep(1)

    if resp is None:
        return None

    raw = (resp.choices[0].message.content or "").strip()
    return _parse_llm_validation(raw)


def _parse_llm_validation(raw: str) -> dict[str, Any] | None:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(raw[start:end + 1])
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    issues = [
        str(i).strip()
        for i in (payload.get("issues") or [])
        if str(i).strip() in _VALID_ISSUES
    ]
    return {
        "passed": not any(i for i in issues if i not in _SOFT_ISSUES),
        "issues": issues,
        "reason": str(payload.get("reason") or ""),
    }


def build_retry_query(query: str, validation: dict[str, Any]) -> str:
    issues = {str(item).strip().lower() for item in validation.get("issues", [])}
    additions: list[str] = []
    if "too_many_self_service_meals" in issues:
        additions.append("prioritize restaurants near attractions reduce self-service")
    if "too_many_long_legs" in issues or "extreme_leg_distance" in issues:
        additions.append("cluster nearby places avoid legs over 15km")
    if "unrealistic_schedule" in issues:
        additions.append("limit to 4 activities per day")
    suffix = " ".join(additions).strip()
    if not suffix:
        return query
    return f"{query} {suffix}".strip()


def should_retry_itinerary(validation: dict[str, Any], retry_attempted: bool) -> bool:
    if retry_attempted or validation.get("passed"):
        return False
    retryable = {
        "too_many_long_legs",
        "extreme_leg_distance",
        "unrealistic_schedule",
    }
    return any(issue in retryable for issue in validation.get("issues", []))


_TRANSIT_LEG_LABELS = {"Departure", "Return to hotel"}


def _check_distances(
    route_plan: list[dict[str, Any]],
    total_days: int,
) -> tuple[list[str], dict[str, Any]]:
    leg_distances = [
        float(leg["distance_km"])
        for leg in route_plan
        if isinstance(leg.get("distance_km"), (int, float))
        and float(leg["distance_km"]) > 0
        and leg.get("leg_label") not in _TRANSIT_LEG_LABELS
    ]
    issues: list[str] = []
    if len([d for d in leg_distances if d > 18.0]) > max(1, total_days - 1):
        issues.append("too_many_long_legs")
    if leg_distances and max(leg_distances) > 25.0:
        issues.append("extreme_leg_distance")
    return issues, {
        "long_leg_count_gt_18km": len([d for d in leg_distances if d > 18.0]),
        "max_leg_km": round(max(leg_distances), 1) if leg_distances else 0.0,
        "distance_source": "route_plan" if route_plan else "unavailable",
    }


def _extract_interest_tags(query: str) -> set[str]:
    from app.places.metadata import fold_text
    from app.places.scoring import INTEREST_KEYWORDS

    q = fold_text(query)
    return {tag for tag, kws in INTEREST_KEYWORDS.items() if any(kw in q for kw in kws)}


def _extract_destination(query: str) -> str:
    return "Da Nang"
