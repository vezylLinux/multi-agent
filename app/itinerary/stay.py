from __future__ import annotations

import json
import re
import time
from typing import Any

from app.core.settings import get_settings
from app.places.metadata import normalize_address_text


def build_stay_recommendations(
    *,
    query: str,
    places: list[dict[str, Any]],
    recommended_hotel: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    candidates = _collect_candidates(places, recommended_hotel=recommended_hotel)
    if len(candidates) < 2:
        return _fallback_recommendations(candidates)

    budget_pool = _budget_pool(candidates)
    premium_pool = _premium_pool(candidates)
    if not budget_pool or not premium_pool:
        return _fallback_recommendations(candidates)

    selected = _llm_recommendations(query=query, budget_pool=budget_pool, premium_pool=premium_pool)
    if selected:
        return selected
    return _fallback_recommendations(candidates)


def _collect_candidates(
    places: list[dict[str, Any]],
    *,
    recommended_hotel: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []

    for place in places:
        if str(place.get("category") or "").strip().lower() != "accommodation":
            continue
        name = str(place.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        enriched = dict(place)
        price_note, min_price, max_price = _extract_price_info(enriched)
        enriched["price_note"] = price_note
        enriched["min_price_vnd"] = min_price
        enriched["max_price_vnd"] = max_price
        enriched["address"] = normalize_address_text(str(enriched.get("address") or ""))
        rows.append(enriched)

    if recommended_hotel:
        name = str(recommended_hotel.get("name") or "").strip()
        if name and name.lower() not in seen:
            extra = dict(recommended_hotel)
            price_note, min_price, max_price = _extract_price_info(extra)
            extra["price_note"] = price_note
            extra["min_price_vnd"] = min_price
            extra["max_price_vnd"] = max_price
            extra["address"] = normalize_address_text(str(extra.get("address") or ""))
            rows.append(extra)

    rows = [row for row in rows if row.get("price_note")]
    rows.sort(key=_stay_score, reverse=True)
    return rows


def _extract_price_info(place: dict[str, Any]) -> tuple[str, int | None, int | None]:
    direct = str(place.get("price_range") or "").strip()
    if direct:
        parsed = _parse_price_range(direct)
        return direct, parsed[0], parsed[1]

    for field in ("list_snippet", "detail_content", "description"):
        text = str(place.get(field) or "").strip()
        if not text:
            continue
        match = re.search(
            r"Gi[áa]\s*:\s*([^\n\r]+?)(?=(?:Điện thoại|Email|Website|Fax|$))",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        price_note = match.group(1).strip(" .")
        parsed = _parse_price_range(price_note)
        if price_note and (parsed[0] or parsed[1]):
            return price_note, parsed[0], parsed[1]

    return "", None, None


def _parse_price_range(text: str) -> tuple[int | None, int | None]:
    numbers = re.findall(r"(?<!\d)(\d{1,3}(?:[.,]\d{3})+|\d{6,9})(?!\d)", text)
    if not numbers:
        return None, None
    parsed: list[int] = []
    for item in numbers[:2]:
        digits = re.sub(r"[^\d]", "", item)
        if len(digits) < 6:
            continue
        parsed.append(int(digits))
    if not parsed:
        return None, None
    if len(parsed) == 1:
        return parsed[0], parsed[0]
    return min(parsed), max(parsed)


def _stay_score(place: dict[str, Any]) -> float:
    fit = float(place.get("customer_fit_score") or 0.0)
    retrieval = float(place.get("retrieval_relevance") or 0.0) * 100
    detail_richness = sum(
        1
        for field in ("description", "detail_content", "list_snippet")
        if str(place.get(field) or "").strip()
    ) * 3.0
    price_bonus = 5.0 if place.get("price_note") else 0.0
    return fit + retrieval + detail_richness + price_bonus


def _budget_pool(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priced = [item for item in candidates if isinstance(item.get("min_price_vnd"), int)]
    if not priced:
        return []
    priced.sort(key=lambda item: (int(item.get("min_price_vnd") or 0), -_stay_score(item)))
    cutoff = max(2, min(len(priced), 6))
    return priced[:cutoff]


def _premium_pool(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priced = [item for item in candidates if isinstance(item.get("max_price_vnd"), int)]
    if not priced:
        return []
    priced.sort(
        key=lambda item: (
            int(item.get("max_price_vnd") or 0),
            _stay_score(item),
        ),
        reverse=True,
    )
    cutoff = max(2, min(len(priced), 6))
    return priced[:cutoff]


def _candidate_context(label: str, pool: list[dict[str, Any]]) -> str:
    lines = [f"## {label}"]
    for item in pool[:4]:
        snippet = (
            str(item.get("detail_content") or "").strip()
            or str(item.get("description") or "").strip()
            or str(item.get("list_snippet") or "").strip()
        )
        snippet = re.sub(r"\s+", " ", snippet)[:420]
        lines.extend(
            [
                f"- Name: {item.get('name','')}",
                f"  - Price estimate: {item.get('price_note','')}",
                f"  - Address: {item.get('address','')}",
                f"  - Stars: {item.get('star_rating','')}",
                f"  - Data suggestion: {snippet}",
            ]
        )
    return "\n".join(lines)


def _llm_recommendations(
    *,
    query: str,
    budget_pool: list[dict[str, Any]],
    premium_pool: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    settings = get_settings()
    openrouter_key = (settings.openrouter_api_key or "").strip()
    if not openrouter_key:
        return []

    try:
        from openai import (  # type: ignore
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            OpenAI,
            RateLimitError,
        )
    except Exception:
        return []

    allowed = [str(item.get("name") or "").strip() for item in [*budget_pool, *premium_pool] if str(item.get("name") or "").strip()]
    if len(set(allowed)) < 2:
        return []

    prompt = f"""You are a travel assistant selecting hotels for an itinerary.

Requirements:
- Select exactly 2 DIFFERENT hotels.
- One hotel for the "Budget" segment.
- One hotel for the "Premium" segment.
- Base the explanation on the price estimate and description data provided.
- Do not use technical reasons like stay_db, retrieval, source.
- Prioritize hotels that suit the following itinerary: {query}

Candidate list:
{_candidate_context("Budget", budget_pool)}

{_candidate_context("Premium", premium_pool)}

Only select names from the candidate list above.
Return valid JSON with schema:
{{
  "recommendations": [
    {{
      "segment": "Budget",
      "name": "string",
      "price_note": "string",
      "why_fit": "string"
    }},
    {{
      "segment": "Premium",
      "name": "string",
      "price_note": "string",
      "why_fit": "string"
    }}
  ]
}}
"""

    client = OpenAI(
        base_url=settings.openrouter_base_url,
        api_key=openrouter_key,
        timeout=max(1, int(settings.openrouter_request_timeout_s or 25)),
    )
    transient = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
    kwargs = {
        "model": settings.openrouter_model,
        "messages": [
            {"role": "system", "content": "You are a travel accommodation advisor."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    if settings.openrouter_reasoning_enabled:
        kwargs["extra_body"] = {"reasoning": {"enabled": True}}

    raw = ""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(**kwargs)
            raw = (resp.choices[0].message.content or "").strip()
            break
        except transient:
            if attempt >= 2:
                return []
            time.sleep(2**attempt)
        except Exception:
            return []
    if not raw:
        return []

    try:
        payload = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return []
        try:
            payload = json.loads(raw[start : end + 1])
        except Exception:
            return []

    recs = payload.get("recommendations") or []
    allowed_set = set(allowed)
    all_candidates = {str(item.get("name") or "").strip(): item for item in [*budget_pool, *premium_pool]}
    out: list[dict[str, Any]] = []
    used: set[str] = set()
    for item in recs:
        segment = str((item or {}).get("segment") or "").strip()
        name = str((item or {}).get("name") or "").strip()
        why_fit = str((item or {}).get("why_fit") or "").strip()
        if segment not in {"Budget", "Premium"} or name not in allowed_set or name in used:
            continue
        candidate = all_candidates.get(name) or {}
        out.append(
            {
                "segment": segment,
                "name": name,
                "price_note": str((item or {}).get("price_note") or candidate.get("price_note") or "").strip(),
                "why_fit": why_fit,
                "address": normalize_address_text(str(candidate.get("address") or "")),
            }
        )
        used.add(name)
    if len(out) != 2:
        return []
    out.sort(key=lambda item: 0 if item["segment"] == "Budget" else 1)
    return out


def _fallback_recommendations(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(candidates) < 2:
        return []
    budget_pool = _budget_pool(candidates)
    premium_pool = _premium_pool(candidates)
    if not budget_pool or not premium_pool:
        return []

    budget = budget_pool[0]
    premium = next((item for item in premium_pool if _name(item) != _name(budget)), premium_pool[0])
    if _name(budget) == _name(premium):
        alternative = next((item for item in candidates if _name(item) != _name(budget)), None)
        if alternative is None:
            return []
        premium = alternative

    return [
        _fallback_item("Budget", budget),
        _fallback_item("Premium", premium),
    ]


def _fallback_item(segment: str, item: dict[str, Any]) -> dict[str, Any]:
    blurb = (
        str(item.get("detail_content") or "").strip()
        or str(item.get("description") or "").strip()
        or str(item.get("list_snippet") or "").strip()
    )
    reason = re.sub(r"\s+", " ", blurb).strip()
    if len(reason) > 180:
        reason = reason[:177].rstrip(" ,.;") + "..."
    if not reason:
        reason = "Has a suitable location and accommodation details worth considering."
    return {
        "segment": segment,
        "name": str(item.get("name") or "").strip(),
        "price_note": str(item.get("price_note") or "").strip(),
        "why_fit": reason,
        "address": normalize_address_text(str(item.get("address") or "")),
    }


def _name(item: dict[str, Any]) -> str:
    return str(item.get("name") or "").strip().lower()
