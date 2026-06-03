from __future__ import annotations

import csv
import json
import sys
import hashlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.settings import get_settings
from app.core.database import init_db
from app.places.metadata import enrich_place_record, normalize_address_text
from app.places.repository import upsert_places
from app.tools.google_places import GooglePlaceResolution, google_places_available, resolve_place_record

RAW_DIR = ROOT / "data" / "crawl" / "processed"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_JSON = OUTPUT_DIR / "unified_places.json"
OUTPUT_JSONL = OUTPUT_DIR / "unified_places.jsonl"
CACHE_DIR = ROOT / "data" / "cache"
PLACES_CACHE_JSON = CACHE_DIR / "places_resolver.json"
LEGACY_GOOGLE_CACHE_JSON = CACHE_DIR / "google_places_resolver.json"

RAW_SOURCES = (
    (RAW_DIR / "dest_danang.json", "json", "destination", "local-processed:dest_danang.json"),
    (RAW_DIR / "rest_danang.json", "json", "restaurant", "local-processed:rest_danang.json"),
    (RAW_DIR / "cslt_danang.json", "json", "accommodation", "local-processed:cslt_danang.json"),
    (RAW_DIR / "vcgt_danang.csv", "csv", "entertainment", "local-processed:vcgt_danang.csv"),
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    settings = get_settings()
    use_places_enrichment = bool(
        (settings.places_resolver_enabled or settings.google_places_enrich_enabled)
        and google_places_available()
    )
    places_cache = _load_places_cache() if use_places_enrichment else {}
    rows, resolver_stats = _load_and_enrich_places(
        use_places_enrichment=use_places_enrichment,
        places_cache=places_cache,
    )
    OUTPUT_JSON.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    OUTPUT_JSONL.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} normalized places to {OUTPUT_JSON}")
    print(f"Wrote {len(rows)} normalized places to {OUTPUT_JSONL}")
    inserted = upsert_places(rows)
    print(f"Upserted {inserted} places into SQLite")
    print("Run 'python scripts/ingest_to_chroma.py' to sync chunks into Chroma.")
    if use_places_enrichment:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        PLACES_CACHE_JSON.write_text(
            json.dumps(places_cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            "Places resolver enrichment:"
            f" hits={resolver_stats['resolved']}, misses={resolver_stats['missed']},"
            f" from_cache={resolver_stats['cache_hits']}"
        )
        print(f"Wrote resolver cache to {PLACES_CACHE_JSON}")


def _load_and_enrich_places(
    use_places_enrichment: bool,
    places_cache: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    resolver_stats = {"resolved": 0, "missed": 0, "cache_hits": 0}
    for path, fmt, default_category, default_source in RAW_SOURCES:
        for row in _read_rows(path=path, fmt=fmt):
            prepared = _prepare_row(
                row=row,
                default_category=default_category,
                default_source=default_source,
            )
            if use_places_enrichment and _should_resolve_place(prepared):
                resolution, cache_hit = _resolve_with_cache(prepared, places_cache)
                if cache_hit:
                    resolver_stats["cache_hits"] += 1
                if resolution:
                    prepared = _apply_place_resolution(prepared, resolution)
                    resolver_stats["resolved"] += 1
                else:
                    resolver_stats["missed"] += 1
            enriched = enrich_place_record(prepared)
            place_id = str(enriched.get("place_id") or "")
            if not place_id or place_id in seen:
                continue
            seen.add(place_id)
            items.append(enriched)
    items.sort(
        key=lambda row: (
            str(row.get("city_key") or ""),
            str(row.get("category") or ""),
            str(row.get("primary_area_key") or ""),
            str(row.get("name") or ""),
        )
    )
    return items, resolver_stats


def _read_rows(path: Path, fmt: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if fmt == "json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []
    if fmt == "csv":
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return []
        return [dict(row) for row in csv.DictReader(text.splitlines())]
    return []


def _prepare_row(
    row: dict[str, Any],
    default_category: str,
    default_source: str,
) -> dict[str, Any]:
    prepared: dict[str, Any] = {}
    for key, value in row.items():
        if value in (None, ""):
            continue
        prepared[key] = value
    prepared["category"] = str(prepared.get("category") or default_category)
    prepared["source"] = str(prepared.get("source") or default_source)
    prepared.setdefault("source_file", default_source)
    for field in ("address", "formatted_address", "map_formatted_address", "google_formatted_address"):
        if field in prepared:
            prepared[field] = normalize_address_text(str(prepared.get(field) or ""))
    prepared["lat"] = _coerce_float(prepared.get("lat"))
    prepared["lon"] = _coerce_float(prepared.get("lon"))
    return prepared


def _coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _should_resolve_place(row: dict[str, Any]) -> bool:
    if str(row.get("map_place_id") or "").strip() or str(row.get("google_place_id") or "").strip():
        return False
    settings = get_settings()
    override_coordinates = bool(
        settings.places_resolver_override_coordinates
        or settings.google_places_override_coordinates
    )
    return not (
        isinstance(row.get("lat"), (int, float))
        and isinstance(row.get("lon"), (int, float))
        and not override_coordinates
    )


def _load_places_cache() -> dict[str, Any]:
    cache_path = PLACES_CACHE_JSON if PLACES_CACHE_JSON.exists() else LEGACY_GOOGLE_CACHE_JSON
    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_with_cache(
    row: dict[str, Any],
    places_cache: dict[str, Any],
) -> tuple[GooglePlaceResolution | None, bool]:
    cache_key = _places_cache_key(row)
    if cache_key in places_cache:
        cached = places_cache.get(cache_key)
        if not cached:
            return None, True
        return _resolution_from_cache(cached), True

    resolution = resolve_place_record(row)
    places_cache[cache_key] = _resolution_to_cache(resolution)
    return resolution, False


def _places_cache_key(row: dict[str, Any]) -> str:
    payload = {
        "resolver_version": 2,
        "name": str(row.get("name") or "").strip(),
        "address": str(row.get("address") or "").strip(),
        "city": str(row.get("city") or "").strip(),
        "category": str(row.get("category") or "").strip().lower(),
        "source": str(row.get("source") or "").strip().lower(),
    }
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _resolution_to_cache(resolution: GooglePlaceResolution | None) -> dict[str, Any] | None:
    if not resolution:
        return None
    return {
        "place_id": resolution.place_id,
        "display_name": resolution.display_name,
        "formatted_address": resolution.formatted_address,
        "lat": resolution.lat,
        "lon": resolution.lon,
        "map_uri": resolution.google_maps_uri,
        "business_status": resolution.business_status,
        "primary_type": resolution.primary_type,
        "types": resolution.types,
        "match_score": resolution.match_score,
        "query_used": resolution.query_used,
        "moved_place_id": resolution.moved_place_id,
        "coordinate_source": resolution.coordinate_source,
        "coordinate_confidence": resolution.coordinate_confidence,
    }


def _resolution_from_cache(payload: dict[str, Any]) -> GooglePlaceResolution | None:
    if not isinstance(payload, dict):
        return None
    place_id = str(payload.get("place_id") or "").strip()
    if not place_id:
        return None
    return GooglePlaceResolution(
        place_id=place_id,
        display_name=str(payload.get("display_name") or "").strip(),
        formatted_address=str(payload.get("formatted_address") or "").strip(),
        lat=_coerce_float(payload.get("lat")),
        lon=_coerce_float(payload.get("lon")),
        google_maps_uri=str(payload.get("map_uri") or payload.get("google_maps_uri") or "").strip(),
        business_status=str(payload.get("business_status") or "").strip(),
        primary_type=str(payload.get("primary_type") or "").strip(),
        types=[str(item).strip() for item in (payload.get("types") or []) if str(item).strip()],
        match_score=float(payload.get("match_score") or 0.0),
        query_used=str(payload.get("query_used") or "").strip(),
        moved_place_id=str(payload.get("moved_place_id") or "").strip(),
        coordinate_source=str(payload.get("coordinate_source") or "nominatim_search").strip(),
        coordinate_confidence=str(payload.get("coordinate_confidence") or "resolved_place").strip(),
    )


def _apply_place_resolution(
    row: dict[str, Any],
    resolution: GooglePlaceResolution,
) -> dict[str, Any]:
    merged = dict(row)
    merged["map_provider"] = "nominatim"
    merged["map_place_id"] = resolution.place_id
    merged["map_display_name"] = resolution.display_name
    merged["map_formatted_address"] = resolution.formatted_address
    merged["map_place_uri"] = resolution.google_maps_uri
    merged["map_business_status"] = resolution.business_status
    merged["map_primary_type"] = resolution.primary_type
    merged["map_types"] = resolution.types
    merged["map_match_score"] = resolution.match_score
    merged["map_query_used"] = resolution.query_used
    merged["google_place_id"] = resolution.place_id
    merged["google_display_name"] = resolution.display_name
    merged["google_formatted_address"] = resolution.formatted_address
    merged["google_maps_uri"] = resolution.google_maps_uri
    merged["google_business_status"] = resolution.business_status
    merged["google_primary_type"] = resolution.primary_type
    merged["google_types"] = resolution.types
    merged["google_match_score"] = resolution.match_score
    merged["google_query_used"] = resolution.query_used
    merged["coordinate_source"] = resolution.coordinate_source
    merged["coordinate_confidence"] = resolution.coordinate_confidence
    if not str(merged.get("address") or "").strip() and resolution.formatted_address:
        merged["address"] = resolution.formatted_address

    settings = get_settings()
    should_override_coords = bool(
        settings.places_resolver_override_coordinates
        or settings.google_places_override_coordinates
    )
    if (
        resolution.lat is not None
        and resolution.lon is not None
        and (
            should_override_coords
            or merged.get("lat") in (None, "")
            or merged.get("lon") in (None, "")
        )
    ):
        merged["lat"] = resolution.lat
        merged["lon"] = resolution.lon
    return merged


if __name__ == "__main__":
    main()
