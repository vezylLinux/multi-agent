from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


_REQUIRED = ["destination", "days", "interests"]
_VALID_CATEGORIES = {"destination", "restaurant", "accommodation", "entertainment"}
_REQUIRED_INTAKE = ["destination", "days", "interests"]

_SYSTEM_PROMPT = "You are a travel request analysis assistant."

_USER_TEMPLATE = """Analyze the travel request below and return JSON.

Request: "{message}"

Schema (return JSON only, no explanation):
{{
  "destination": "destination name (Da Nang district or area, default Da Nang if unclear)",
  "days": "trip duration in DAYS as a string integer. CRITICAL: when input uses 'X ngày Y đêm' or 'X days Y nights' (X is always greater than Y by 1), you MUST return X (the larger number, days). Examples: '4 ngày 3 đêm' -> '4'; '3 ngày 2 đêm' -> '3'; '2 ngày 1 đêm' -> '2'; '5 days 4 nights' -> '5'. If only nights are given (e.g. '3 đêm'), return nights+1. Leave empty if no count is mentioned.",
  "interests": "comma-separated interests — choose from: food, beach, museum, heritage, spiritual, shopping, cafe, nature, nightlife, family — leave empty if not mentioned",
  "budget": "budget description (e.g. 'budget', 'luxury'), leave empty if not mentioned",
  "companion": "companion type (e.g. 'family', 'friends', 'couple', 'solo'), leave empty if not mentioned"
}}"""

_COMBINED_TEMPLATE = """Analyze the travel request below.

Request: "{message}"

Return valid JSON with schema (return JSON only, no explanation):
{{
  "destination": "destination name (Da Nang district or area, default Da Nang if unclear)",
  "days": "number of days as a string number ('3'), leave empty if not mentioned",
  "interests": "comma-separated interests — choose from: food, beach, museum, heritage, spiritual, shopping, cafe, nature, nightlife, family — leave empty if not mentioned",
  "budget": "budget description, leave empty if not mentioned",
  "companion": "companion type (family/friends/couple/solo), leave empty if not mentioned",
  "subtasks": [
    {{
      "category": "destination|restaurant|accommodation|entertainment",
      "area": "specific area name",
      "description": "short description of the type of place to find"
    }}
  ]
}}

Rules for subtasks:
- Always create a "destination" category subtask
- Always create a "restaurant" subtask
- Create an "accommodation" subtask if days >= 2
- Create an "entertainment" subtask only if interests include giai_tri_dem"""


@dataclass
class IntakeResult:
    is_complete: bool
    collected: dict[str, str]
    missing_fields: list[str]
    follow_up_questions: list[str]


@dataclass
class Subtask:
    category: str
    area: str
    description: str


@dataclass
class IntakeDecompResult:
    collected: dict[str, str]
    is_complete: bool
    missing_fields: list[str]
    follow_up_questions: list[str]
    subtasks: list[Subtask] = field(default_factory=list)
    fallback_used: bool = False


def evaluate_intake(message: str) -> IntakeResult:
    text = (message or "").strip()
    if not text:
        return _make_result({})
    return _make_result(_llm_extract(text) or {})


def intake_and_decompose(message: str) -> IntakeDecompResult | None:
    """Single LLM call combining A1 intake extraction and A2 task decomposition.

    Returns None if LLM is unavailable so callers can fall back to separate calls.
    """
    text = (message or "").strip()
    if not text:
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

    client = OpenAI(
        base_url=settings.openrouter_base_url,
        api_key=openrouter_key,
        timeout=max(1, int(settings.openrouter_request_timeout_s or 25)),
    )
    prompt = _COMBINED_TEMPLATE.format(message=text)
    transient = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
    resp = None
    for attempt in range(3):
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
            if attempt >= 2:
                return None
            time.sleep(2**attempt)

    if resp is None:
        return None

    raw = (resp.choices[0].message.content or "").strip()
    parsed = _parse_combined_output(raw)
    if parsed is None:
        return None

    collected, subtasks = parsed
    missing = [f for f in _REQUIRED_INTAKE if not collected.get(f)]
    return IntakeDecompResult(
        collected=collected,
        is_complete=len(missing) == 0,
        missing_fields=missing,
        follow_up_questions=[question_for_field(f) for f in missing],
        subtasks=subtasks,
        fallback_used=False,
    )


def question_for_field(field: str) -> str:
    prompts = {
        "destination": "Bạn muốn đến khu vực nào ở Đà Nẵng?",
        "days": "Chuyến đi của bạn kéo dài mấy ngày?",
        "interests": (
            "Bạn thích những trải nghiệm gì? "
            "(ví dụ: ẩm thực, di tích lịch sử, bảo tàng, biển, mua sắm, cà phê, tâm linh, thiên nhiên...)"
        ),
    }
    return prompts[field]


def _llm_extract(text: str) -> dict[str, str] | None:
    from app.core.settings import get_settings

    settings = get_settings()
    openrouter_key = (settings.openrouter_api_key or "").strip()
    if not openrouter_key:
        return None

    try:
        from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
    except Exception:
        return None

    client = OpenAI(
        base_url=settings.openrouter_base_url,
        api_key=openrouter_key,
        timeout=max(1, int(settings.openrouter_request_timeout_s or 25)),
    )
    prompt = _USER_TEMPLATE.format(message=text)
    transient = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
    resp = None
    for attempt in range(3):
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
            if attempt >= 2:
                return None
            time.sleep(2**attempt)

    if resp is None:
        return None

    raw = (resp.choices[0].message.content or "").strip()
    return _parse_llm_output(raw)


def _parse_llm_output(raw: str) -> dict[str, str] | None:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(raw[start : end + 1])
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return {
        "destination": str(payload.get("destination") or "").strip(),
        "days": str(payload.get("days") or "").strip(),
        "interests": str(payload.get("interests") or "").strip(),
        "budget": str(payload.get("budget") or "").strip(),
        "companion": str(payload.get("companion") or "").strip(),
    }


def _make_result(collected: dict[str, Any]) -> IntakeResult:
    full = {
        "destination": str(collected.get("destination") or "").strip(),
        "days": str(collected.get("days") or "").strip(),
        "interests": str(collected.get("interests") or "").strip(),
        "budget": str(collected.get("budget") or "").strip(),
        "companion": str(collected.get("companion") or "").strip(),
    }
    missing = [f for f in _REQUIRED if not full[f]]
    questions = [question_for_field(f) for f in missing]
    return IntakeResult(
        is_complete=len(missing) == 0,
        collected=full,
        missing_fields=missing,
        follow_up_questions=questions,
    )


def _parse_combined_output(raw: str) -> tuple[dict[str, str], list[Subtask]] | None:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(raw[start : end + 1])
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    collected = {
        "destination": str(payload.get("destination") or "").strip(),
        "days": str(payload.get("days") or "").strip(),
        "interests": str(payload.get("interests") or "").strip(),
        "budget": str(payload.get("budget") or "").strip(),
        "companion": str(payload.get("companion") or "").strip(),
    }
    subtasks: list[Subtask] = []
    for item in payload.get("subtasks") or []:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "").strip().lower()
        if category not in _VALID_CATEGORIES:
            continue
        area = str(item.get("area") or "").strip()
        description = str(item.get("description") or "").strip()
        if not area or not description:
            continue
        subtasks.append(Subtask(category=category, area=area, description=description))
    return collected, subtasks
