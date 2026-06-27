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
        "destination": (
            "🏖️ Bạn muốn khám phá khu vực nào ở Đà Nẵng? "
            "(ví dụ: Bãi Mỹ Khê, Bán đảo Sơn Trà, Phố cổ, khu trung tâm, hoặc bạn muốn khám phá toàn bộ Đà Nẵng?)"
        ),
        "days": (
            "📅 Chuyến đi của bạn kéo dài bao lâu? "
            "(vui lòng cho biết số ngày, ví dụ: 3 ngày 2 đêm, 4 ngày...)"
        ),
        "interests": (
            "❤️ Bạn thích những trải nghiệm gì? "
            "(có thể chọn nhiều, ví dụ: ẩm thực, di tích lịch sử, bảo tàng, biển, mua sắm, cà phê, tâm linh, thiên nhiên, nightlife, hoạt động gia đình...)"
        ),
        "budget": (
            "💰 Ngân sách dự tính của bạn là bao nhiêu? "
            "(ví dụ: tiết kiệm/budget, trung bình, cao cấp/luxury)"
        ),
        "companion": (
            "👥 Bạn đi du lịch với ai? "
            "(ví dụ: một mình, bạn bè, gia đình, đôi...)"
        ),
    }
    return prompts.get(field, f"Vui lòng cung cấp thông tin về: {field}")


def generate_contextual_suggestions(collected: dict[str, str]) -> list[str]:
    """Generate contextual suggestions based on what user has already provided."""
    suggestions = []
    
    destination = str(collected.get("destination", "")).strip().lower()
    days = str(collected.get("days", "")).strip()
    interests = str(collected.get("interests", "")).strip().lower()
    budget = str(collected.get("budget", "")).strip().lower()
    companion = str(collected.get("companion", "")).strip().lower()
    
    # Suggestions based on destination
    if destination and "sơn trà" in destination:
        suggestions.append("💡 Tip: Bán đảo Sơn Trà nổi tiếng với rừng nguyên sinh - đừng quên ghé Thảo Cầm Viên!")
    elif destination and "mỹ khê" in destination:
        suggestions.append("💡 Tip: Bãi Mỹ Khê là bãi biển đẹp nhất - thời gian tốt nhất là sáng sớm!")
    
    # Suggestions based on days
    if days:
        try:
            day_count = int(days)
            if day_count <= 2:
                suggestions.append("💡 Tip: Với " + days + " ngày, bạn nên tập trung vào các điểm chính. Gợi ý: Bãi Mỹ Khê + Phố cổ hoặc Thảo Cầm Viên.")
            elif day_count <= 4:
                suggestions.append("💡 Tip: Với " + days + " ngày, bạn có thể kết hợp biển, lịch sử và ẩm thực địa phương.")
            else:
                suggestions.append("💡 Tip: Với " + days + " ngày, bạn đủ thời gian khám phá đầy đủ Đà Nẵng bao gồm cả những địa điểm lân cận.")
        except Exception:
            pass
    
    # Suggestions based on interests
    if interests:
        if "biển" in interests or "beach" in interests.lower():
            suggestions.append("🌊 Địa điểm biển nổi tiếng: Bãi Mỹ Khê, bãi Khe Ngang, bãi Nam Ô.")
        if "ẩm thực" in interests or "food" in interests.lower():
            suggestions.append("🍜 Không nên bỏ lỡ: Mì Quảng, Bánh canh cua, Cơm gà, các quán ăn ở chợ Hàn.")
        if "lịch sử" in interests or "heritage" in interests.lower():
            suggestions.append("🏛️ Điểm tham quan: Phố cổ, Chùa Thái Hà, Di tích Nạn Hành, Cầu Vàng.")
        if "tâm linh" in interests or "spiritual" in interests.lower():
            suggestions.append("🙏 Các chùa và đền nổi tiếng: Chùa Tam Bảo, Đền Mẫu Cô Gái, Chùa Linh Ứng.")
        if "thiên nhiên" in interests or "nature" in interests.lower():
            suggestions.append("🏔️ Tự nhiên: Bán đảo Sơn Trà, Thành phố Đèn khu du lịch Đà Nẵng, Suối Bàn Tay.")
        if "mua sắm" in interests or "shopping" in interests.lower():
            suggestions.append("🛍️ Mua sắm: Chợ Hàn, Việt Nam Grand Plaza, các cửa hàng lưu niệm trên phố Tràng Tiền.")
        if "cà phê" in interests or "cafe" in interests.lower():
            suggestions.append("☕ Quán cà phê đẹp: Cà phê teahouse dọc sông Hàn, các quán cà phê nghệ thuật ở phố cổ.")
        if "nightlife" in interests:
            suggestions.append("🌙 Hoạt động buổi tối: Sky bar, Bar trên mái, Chợ Đêm, khu ăn chơi trên bãi biển.")
        if "gia đình" in interests or "family" in interests.lower():
            suggestions.append("👨‍👩‍👧 Hoạt động gia đình: Thảo Cầm Viên, Công viên nước, Bãi biển an toàn, tiểu đoàn công viên.")
    
    # Suggestions based on budget
    if budget:
        if "tiết kiệm" in budget or "budget" in budget or "rẻ" in budget:
            suggestions.append("💰 Gợi ý tiết kiệm: Ăn ở chợ địa phương, khám phá các điểm miễn phí như phố cổ, bãi biển công cộng.")
        elif "cao cấp" in budget or "luxury" in budget.lower():
            suggestions.append("✨ Resort sang trọng: InterContinental, Fusion Maia, Sonasea Phòng khách Đà Nẵng - các nhà hàng Michelin.")
    
    # Suggestions based on companion
    if companion:
        if "gia đình" in companion:
            suggestions.append("👨‍👩‍👧‍👦 Gợi ý gia đình: Chọn chỗ ở gần biển, tránh những hoạt động mạo hiểm, ưu tiên các điểm an toàn cho trẻ em.")
        elif "bạn bè" in companion:
            suggestions.append("👯 Gợi ý bạn bè: Khám phá nightlife, các hoạt động thể thao nước, những quán bar trên biển.")
        elif "đôi" in companion:
            suggestions.append("💑 Gợi ý đôi: Những quán cà phê lãng mạn, dạo phố vào chiều tối, sunset cruise trên sông Hàn.")
        elif "một mình" in companion:
            suggestions.append("🎒 Gợi ý một mình: Các chuyến tham gia nhóm, hoạt động ngoài trời, những quán cà phê với cộng đồng backpacker.")
    
    return suggestions


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
