from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from typing import Any


_AREA_TO_CITY_KEY = {
    "hai chau": "da_nang",
    "son tra": "da_nang",
    "ngu hanh son": "da_nang",
    "thanh khe": "da_nang",
    "cam le": "da_nang",
    "lien chieu": "da_nang",
    "hoa vang": "da_nang",
}

_INTENT_ORDER = (
    "food",
    "beach",
    "museum",
    "heritage",
    "spiritual",
    "shopping",
    "cafe",
    "nature",
    "nightlife",
    "family",
)


def fold_text(text: str) -> str:
    base = (text or "").replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", base)
    no_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", no_marks).strip().lower()


def normalize_address_text(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip(" ,")
    if not value:
        return ""
    parts = [part.strip(" ,") for part in value.split(",")]
    parts = [part for part in parts if part]
    normalized = ", ".join(parts)
    normalized = re.sub(r"\(\s*,\s*", "(", normalized)
    normalized = re.sub(r",\s*,+", ", ", normalized)
    normalized = re.sub(r"\(\s*\)", "", normalized)
    return normalized.strip(" ,")


def city_key_from_text(text: str) -> str:
    folded = fold_text(text)
    if "da nang" in folded or "danang" in folded:
        return "da_nang"
    for area_key, city_key in _AREA_TO_CITY_KEY.items():
        if area_key in folded:
            return city_key
    return ""


def enrich_place_record(
    place: dict[str, Any],
    *,
    llm_client: Any = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    enriched = dict(place)
    for field in ("address", "formatted_address", "map_formatted_address", "google_formatted_address"):
        if field in enriched:
            enriched[field] = normalize_address_text(str(enriched.get(field) or ""))
    if llm_client is not None:
        enriched["intent_tags"] = infer_intent_tags(enriched, llm_client, llm_model or "")
    enriched["place_id"] = build_place_id(enriched)
    return enriched


def enrich_places(
    places: list[dict[str, Any]],
    *,
    llm_client: Any = None,
    llm_model: str | None = None,
) -> list[dict[str, Any]]:
    return [enrich_place_record(place, llm_client=llm_client, llm_model=llm_model) for place in places]


def infer_intent_tags(
    place: dict[str, Any],
    client: Any,
    model: str,
) -> list[str]:
    try:
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
    except Exception:
        return []

    name = str(place.get("name") or "").strip()
    category = str(place.get("category") or "").strip()
    destination_type = str(place.get("destination_type") or "").strip()
    address = str(place.get("address") or "").strip()
    description = str(place.get("description") or "")[:300]

    valid_tags = ", ".join(_INTENT_ORDER)
    prompt = (
        f"Classify this tourism place by intent tags.\n\n"
        f"Valid tags only: {valid_tags}\n\n"
        f"Place:\n"
        f"- Name: {name}\n"
        f"- Category: {category}\n"
        f"- Type: {destination_type}\n"
        f"- Address: {address}\n"
        f"- Description: {description}\n\n"
        f'Return ONLY a JSON array of matching tags. Example: ["food", "cafe"]\n'
        f"Return [] if none match. No explanation."
    )

    transient = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
    resp = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            break
        except transient:
            if attempt >= 2:
                return []
            time.sleep(2 ** attempt)
        except Exception:
            return []

    if resp is None:
        return []

    raw = (resp.choices[0].message.content or "").strip()
    try:
        tags = json.loads(raw)
    except Exception:
        start, end = raw.find("["), raw.rfind("]")
        if start < 0 or end <= start:
            return []
        try:
            tags = json.loads(raw[start : end + 1])
        except Exception:
            return []

    if not isinstance(tags, list):
        return []

    valid_set = set(_INTENT_ORDER)
    return [tag for tag in _INTENT_ORDER if tag in {t for t in tags if isinstance(t, str)} & valid_set]


def build_place_id(place: dict[str, Any]) -> str:
    if str(place.get("place_id") or "").strip():
        return str(place.get("place_id")).strip()
    payload = {
        "name": str(place.get("name") or "").strip(),
        "address": str(place.get("address") or "").strip(),
        "category": str(place.get("category") or "").strip().lower(),
        "source": str(place.get("source") or "").strip().lower(),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return digest


_DAY_PATTERN = re.compile(r"(\d+)[\s-]*(?:ng[aà]y|days?)", re.IGNORECASE)


def extract_trip_days(
    query: str,
    *,
    default: int | None = 1,
    max_days: int = 7,
) -> int | None:
    match = _DAY_PATTERN.search(query or "")
    if not match:
        return default
    try:
        days = int(match.group(1))
    except Exception:
        return default
    return max(1, min(days, max_days))
