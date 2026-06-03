from __future__ import annotations

import json
import unicodedata
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from app.core.settings import get_settings

_DA_NANG_CENTER = {"latitude": 16.0544, "longitude": 108.2022}
_SEARCH_RADIUS_M = 60_000  # 60 km covers all of Da Nang + surrounding area


@dataclass(frozen=True)
class GooglePlaceResolution:
    place_id: str
    display_name: str = ""
    formatted_address: str = ""
    lat: float | None = None
    lon: float | None = None
    google_maps_uri: str = ""
    business_status: str = ""
    primary_type: str = ""
    types: list[str] = field(default_factory=list)
    match_score: float = 0.0
    query_used: str = ""
    moved_place_id: str = ""
    coordinate_source: str = ""
    coordinate_confidence: str = ""


def google_places_available() -> bool:
    settings = get_settings()
    return bool(str(settings.google_maps_api_key or "").strip())


def resolve_place_record(place: dict[str, Any]) -> GooglePlaceResolution | None:
    settings = get_settings()
    api_key = str(settings.google_maps_api_key or "").strip()
    if not api_key:
        return None

    name = str(place.get("name") or "").strip()
    address = str(place.get("address") or "").strip()
    city = str(place.get("city") or "Đà Nẵng").strip()

    if not name:
        return None

    query = f"{name}, {address}" if address else f"{name} {city} Việt Nam"

    result = _text_search(query, api_key, settings)
    if result is None and address:
        result = _text_search(f"{name} {city} Việt Nam", api_key, settings)

    if result is None:
        return None

    score = _name_similarity(name, result.get("displayName", {}).get("text", ""))
    lat = result.get("location", {}).get("latitude")
    lon = result.get("location", {}).get("longitude")
    if not isinstance(lat, float) or not isinstance(lon, float):
        return None

    return GooglePlaceResolution(
        place_id=result.get("id", ""),
        display_name=result.get("displayName", {}).get("text", ""),
        formatted_address=result.get("formattedAddress", ""),
        lat=lat,
        lon=lon,
        google_maps_uri=result.get("googleMapsUri", ""),
        business_status=result.get("businessStatus", ""),
        primary_type=result.get("primaryType", ""),
        types=result.get("types", []),
        match_score=score,
        query_used=query,
        coordinate_source="google_places",
        coordinate_confidence="high" if score >= 0.7 else "medium",
    )


def _text_search(query: str, api_key: str, settings: Any) -> dict | None:
    base_url = str(settings.google_places_base_url or "https://places.googleapis.com").rstrip("/")
    url = f"{base_url}/v1/places:searchText"
    field_mask = "places.id,places.displayName,places.formattedAddress,places.location,places.businessStatus,places.primaryType,places.types,places.googleMapsUri"

    body = {
        "textQuery": query,
        "languageCode": settings.google_places_language_code or "vi",
        "regionCode": settings.google_places_region_code or "VN",
        "maxResultCount": 3,
        "locationBias": {
            "circle": {
                "center": _DA_NANG_CENTER,
                "radius": float(_SEARCH_RADIUS_M),
            }
        },
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": field_mask,
        },
        method="POST",
    )

    try:
        timeout = int(settings.google_places_request_timeout_s or 10)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

    places = data.get("places") or []
    return places[0] if places else None


def _fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(c) != "Mn"
    ).strip()


def _name_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    fa, fb = _fold(a), _fold(b)
    if fa == fb:
        return 1.0
    if fa in fb or fb in fa:
        return 0.85
    words_a = set(fa.split())
    words_b = set(fb.split())
    if not words_a or not words_b:
        return 0.0
    overlap = len(words_a & words_b)
    return overlap / max(len(words_a), len(words_b))
