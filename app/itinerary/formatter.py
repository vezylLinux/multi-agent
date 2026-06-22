from __future__ import annotations

import re
from typing import Any

from app.places.metadata import normalize_address_text
from app.places.metadata import extract_trip_days

_RESEARCH_HIGHLIGHTS_HEADER = "ĐIỂM NỔI BẬT:"
_RESEARCH_STAY_HEADER = "LƯU TRÚ GỢI Ý:"
_RESEARCH_TIPS_HEADER = "MẸO DU LỊCH:"
_PLAN_STAY_HEADER = "LƯU TRÚ:"
_PLAN_SUMMARY_PREFIX = "Tổng quan:"
_PLAN_NOTE_PREFIX = "Lưu ý:"
_COORDINATOR_TOP_HEADER = "3 TRẢI NGHIỆM NỔI BẬT CẦN ƯU TIÊN:"
_COORDINATOR_CHALLENGE_HEADER = "THÁCH THỨC & CÁCH XỬ LÝ:"
_COORDINATOR_CHECKLIST_HEADER = "DANH SÁCH VIỆC CẦN LÀM TRƯỚC CHUYẾN ĐI:"


def build_time_confirmation_question(destination: str) -> str:
    place_label = destination or "điểm đến này"
    return (
        f"Bạn dự định đi {place_label} vào thời gian cụ thể nào? "
        "(ví dụ: cuối tháng 6, mùa hè, dịp lễ Quốc khánh) "
        "Khi bạn xác nhận ngày, tôi có thể kiểm tra thời tiết và nhắc những gì cần chuẩn bị."
    )


def format_planning_answer(
    *,
    query: str,
    collected_info: dict[str, Any] | None,
    research: str | None,
    plan: str | None,
    coordinator_plan: str | None,
    transport: list[str] | None,
    recommended_hotel: dict[str, Any] | None,
    mobility_plan: dict[str, Any] | None,
    stay_plan: dict[str, Any] | None,
    stay_recommendations: list[dict[str, Any]] | None,
    plan_validation: dict[str, Any] | None,
    verified_places: list[dict[str, Any]] | None,
) -> str:
    destination = _destination_label(query=query, collected_info=collected_info)
    days = _trip_days_label(query=query, collected_info=collected_info)
    research_sections = _parse_research_summary(research or "")
    itinerary_sections = _parse_itinerary_plan(plan or "")
    coordinator_sections = _parse_coordinator_plan(coordinator_plan or "")
    highlight_lines = _choose_highlights(research_sections, verified_places or [], coordinator_sections)
    stay_lines = _format_stay_lines(
        recommended_hotel=recommended_hotel,
        stay_plan=stay_plan,
        stay_recommendations=stay_recommendations,
        fallback_lines=itinerary_sections["stay_lines"],
    )
    transport_lines = _format_transport_lines(transport or [])
    checklist_lines = coordinator_sections["checklist_lines"]
    challenge_lines = coordinator_sections["challenge_lines"]
    tips_lines = _merge_unique_lines(
        research_sections["tip_lines"] + itinerary_sections["note_lines"] + challenge_lines + checklist_lines
    )
    intro = research_sections["overview"]
    if not intro:
        intro = (
            f"Dưới đây là kế hoạch {days} tại {destination} "
            "để bạn tham khảo — có thể điều chỉnh tùy nhu cầu thực tế."
        )

    lines: list[str] = [
        f"KẾ HOẠCH DU LỊCH - {destination.upper()}",
        "",
        "Tóm tắt:",
        intro,
    ]
    if highlight_lines:
        lines.extend(["", "Điểm nổi bật:", *highlight_lines])

    if stay_lines:
        lines.extend(["", "Lưu trú gợi ý:", *stay_lines])

    if itinerary_sections["day_blocks"]:
        lines.extend(["", "Lịch trình theo ngày:", *itinerary_sections["day_blocks"]])

    if transport_lines:
        lines.append("")
        lines.append("Di chuyển:")
        lines.extend(transport_lines)

    if tips_lines:
        lines.extend(["", "Mẹo & lưu ý:", *tips_lines[:10]])

    return "\n".join(line for line in lines if line is not None).strip()


def _destination_label(query: str, collected_info: dict[str, Any] | None) -> str:
    destination = str((collected_info or {}).get("destination") or "").strip()
    if destination:
        return destination
    return "Da Nang"


def _trip_days_label(query: str, collected_info: dict[str, Any] | None) -> str:
    raw_days = str((collected_info or {}).get("days") or "").strip()
    if raw_days.isdigit():
        return f"{raw_days} ngày"
    days = extract_trip_days(query)
    if days:
        return f"{days} ngày"
    return "vài ngày"


def _parse_research_summary(text: str) -> dict[str, Any]:
    lines = [line.rstrip() for line in text.splitlines()]
    overview_parts: list[str] = []
    section = "overview"
    highlights: list[str] = []
    stay_lines: list[str] = []
    tip_lines: list[str] = []

    for raw_line in lines[1:]:
        line = raw_line.strip()
        if not line:
            continue
        if line == _RESEARCH_HIGHLIGHTS_HEADER:
            section = "highlights"
            continue
        if line == _RESEARCH_STAY_HEADER:
            section = "stay"
            continue
        if line == _RESEARCH_TIPS_HEADER:
            section = "tips"
            continue
        if section == "overview":
            overview_parts.append(line)
        elif section == "highlights":
            highlights.append(_dashify(line))
        elif section == "stay":
            stay_lines.append(_dashify(line))
        elif section == "tips":
            tip_lines.append(_dashify(line))

    return {
        "overview": " ".join(overview_parts).strip(),
        "highlights": highlights,
        "stay_lines": stay_lines,
        "tip_lines": tip_lines,
    }


def _parse_itinerary_plan(text: str) -> dict[str, Any]:
    lines = [line.rstrip() for line in text.splitlines()]
    stay_lines: list[str] = []
    day_blocks: list[str] = []
    note_lines: list[str] = []

    current_day_title = ""
    current_day_lines: list[str] = []
    in_stay = False

    for raw_line in lines[1:]:
        line = raw_line.strip()
        if not line:
            continue
        if line == _PLAN_STAY_HEADER:
            in_stay = True
            continue
        if line.startswith("DAY ") or line.startswith("NGÀY "):
            if current_day_title:
                day_blocks.append(_render_day_block(current_day_title, current_day_lines))
            current_day_title = line
            current_day_lines = []
            in_stay = False
            continue
        if line.startswith(_PLAN_SUMMARY_PREFIX):
            note_lines.append(_dashify(line.split(":", 1)[1].strip()))
            continue
        if line.startswith(_PLAN_NOTE_PREFIX):
            note_lines.append(_dashify(line.split(":", 1)[1].strip()))
            continue
        if in_stay:
            cleaned = _clean_user_facing_line(line)
            if cleaned:
                stay_lines.append(_dashify(cleaned))
        elif current_day_title:
            cleaned = _clean_user_facing_line(line)
            if cleaned:
                current_day_lines.append(cleaned)

    if current_day_title:
        day_blocks.append(_render_day_block(current_day_title, current_day_lines))

    return {
        "stay_lines": stay_lines,
        "day_blocks": day_blocks,
        "note_lines": note_lines,
    }


def _parse_coordinator_plan(text: str) -> dict[str, Any]:
    lines = [line.rstrip() for line in text.splitlines()]
    top_lines: list[str] = []
    challenge_lines: list[str] = []
    checklist_lines: list[str] = []
    section = ""

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line == _COORDINATOR_TOP_HEADER:
            section = "top"
            continue
        if line == _COORDINATOR_CHALLENGE_HEADER:
            section = "challenge"
            continue
        if line == _COORDINATOR_CHECKLIST_HEADER:
            section = "checklist"
            continue
        if line.endswith(":") and line.isupper():
            section = ""
            continue
        if section == "top":
            top_lines.append(_dashify(line))
        elif section == "challenge":
            challenge_lines.append(_dashify(line))
        elif section == "checklist":
            checklist_lines.append(_dashify(line.replace("□", "").strip()))

    return {
        "top_lines": top_lines,
        "challenge_lines": challenge_lines,
        "checklist_lines": checklist_lines,
    }


def _choose_highlights(
    research_sections: dict[str, Any],
    verified_places: list[dict[str, Any]],
    coordinator_sections: dict[str, Any],
) -> list[str]:
    if research_sections["highlights"]:
        return research_sections["highlights"][:5]
    if coordinator_sections["top_lines"]:
        return coordinator_sections["top_lines"][:5]

    ranked = sorted(
        [
            item for item in verified_places
            if str(item.get("name") or "").strip()
        ],
        key=lambda item: float(item.get("customer_fit_score") or 0.0),
        reverse=True,
    )
    out: list[str] = []
    for item in ranked[:5]:
        name = str(item.get("name") or "").strip()
        category = str(item.get("category") or "").strip()
        address = normalize_address_text(str(item.get("address") or ""))
        detail = name
        if category and category.strip().lower() not in {"destination", "tourism"}:
            detail += f" - {category}"
        if address:
            detail += f" ({address})"
        out.append(f"- {detail}")
    return out


def _format_stay_lines(
    *,
    recommended_hotel: dict[str, Any] | None,
    stay_plan: dict[str, Any] | None,
    stay_recommendations: list[dict[str, Any]] | None,
    fallback_lines: list[str],
) -> list[str]:
    if stay_recommendations:
        out: list[str] = []
        for item in stay_recommendations[:2]:
            segment = str(item.get("segment") or "").strip()
            name = str(item.get("name") or "").strip()
            price_note = str(item.get("price_note") or "").strip()
            address = normalize_address_text(str(item.get("address") or ""))
            why_fit = str(item.get("why_fit") or "").strip()
            if not segment or not name:
                continue
            out.append(f"- {segment}: {name}")
            if price_note:
                out.append(f"- Giá ước tính: {price_note}")
            if address:
                out.append(f"- Địa chỉ: {address}")
            if why_fit:
                out.append(f"- Phù hợp vì: {why_fit}")
            out.append("")
        return [line for line in out if line.strip()]

    if isinstance(recommended_hotel, dict) and recommended_hotel:
        if recommended_hotel.get("type") == "multi_city_stay":
            out: list[str] = []
            for segment in recommended_hotel.get("segments", []) or []:
                hotel = segment.get("hotel") or {}
                hotel_name = str(hotel.get("name") or "").strip() or "hotel name not available"
                city_label = str(segment.get("city_label") or segment.get("city_key") or "").strip()
                days = segment.get("days") or []
                day_label = _days_compact_label(days)
                address = str(hotel.get("address") or "").strip()

                if city_label:
                    line += f" ({city_label})"
                if address:
                    line += f" - {address}"
                out.append(line)
            if out:
                return out
        name = str(recommended_hotel.get("name") or "").strip()
        if name:
            out = [f"- Khách sạn gợi ý: {name}"]
            address = normalize_address_text(str(recommended_hotel.get("address") or ""))
            if address:
                out.append(f"- Địa chỉ: {address}")
            return out

    if stay_plan:
        out = []
        for segment in stay_plan.get("segments", []) or []:
            hotel = segment.get("hotel") or {}
            hotel_name = str(hotel.get("name") or "").strip()
            if not hotel_name:
                continue
            line = f"- {str(segment.get('days_label') or _days_compact_label(segment.get('days') or [])).strip()}: {hotel_name}"
            address = str(hotel.get("address") or "").strip()
            if address:
                line += f" - {address}"
            out.append(line)
        if out:
            return out

    return fallback_lines[:4]


def _format_transport_lines(transport: list[str]) -> list[str]:
    return [_dashify(line) for line in transport if str(line).strip()]


def _render_day_block(title: str, lines: list[str]) -> str:
    block = [title]
    block.extend(lines)
    return "\n".join(block).strip()


def _clean_user_facing_line(line: str) -> str:
    text = str(line or "").strip()
    if not text:
        return ""
    if re.match(r"^[-•]?\s*why it fits:", text, flags=re.IGNORECASE):
        return ""
    if re.match(r"^(?:route|movement) summary:\s*$", text, flags=re.IGNORECASE):
        return ""
    if re.match(r"^[-•]?\s*leg\s+\d+\s*:", text, flags=re.IGNORECASE):
        return ""
    if re.match(r"^[-•]?\s*Leg link:", text, flags=re.IGNORECASE):
        return ""
    if re.match(r"^[-•]?\s*Overnight\s*:", text, flags=re.IGNORECASE):
        return ""

    if text.startswith("• Day route map:"):
        return ""
    if text.startswith("• Place order:"):
        return ""
    if re.match(r"^(?:[-•]+\s*)?Leg link:\s*https?://\S+$", text, flags=re.IGNORECASE):
        return ""
    if "->" in text and "http" in text and not re.search(r"\b(?:km|min)\b", text, flags=re.IGNORECASE):
        return ""
    if text == "Leg map:":
        return ""

    text = re.sub(
        r"\s*—\s*Source:\s*.*?(?=(?:\s*—\s*(?:Reason|Map):)|(?:\.\s+Activity:)|(?:\.\s+Leg link:)|(?:\.\s*$)|$)",
        "",
        text,
    )
    text = re.sub(
        r"\s*—\s*Reason:\s*.*?(?=(?:\s*—\s*Map:)|(?:\.\s+Activity:)|(?:\.\s+Leg link:)|(?:\.\s*$)|$)",
        "",
        text,
    )
    text = re.sub(r"\s*\((?:source map|map source):\s*[^)]+\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*-\s*destination\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*,\s*", "(", text)
    text = re.sub(r",\s*,+", ", ", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s*—\s*Map:\s*(https?://\S+)", r" — Map: \1", text)
    text = text.replace(" . ", ". ")
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def _merge_unique_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw_line in lines:
        line = _dashify(raw_line)
        normalized = line.lstrip("- ").strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(line)
    return out


def _dashify(line: str) -> str:
    text = str(line or "").strip()
    if not text:
        return ""
    if text.startswith("- "):
        return text
    if text.startswith("• "):
        return "- " + text[2:].strip()
    if text.startswith("□ "):
        return "- " + text[2:].strip()
    return "- " + text


def _days_compact_label(days: Any) -> str:
    if not isinstance(days, list) or not days:
        return "schedule date undefined"
    if len(days) == 1:
        return f"Day {days[0]}"
    return f"Days {days[0]}-{days[-1]}"
