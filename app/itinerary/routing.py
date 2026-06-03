from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from typing import Any
from urllib.parse import quote

from app.places.metadata import fold_text

_TRACKASIA_HOST = "https://maps.track-asia.com"
_TRACKASIA_DEFAULT_MODE = "driving"
_TRACKASIA_DEFAULT_PLACE_ZOOM = 16


@dataclass(frozen=True)
class ResolvedMapLocation:
    lat: float
    lon: float
    label: str
    address: str
    source: str


def resolve_location_for_map(
    place: dict[str, Any] | None,
) -> ResolvedMapLocation | None:
    """Resolve a place dict to coordinates for map display and routing.

    Priority:
      1. DB lat/lon — trusted source after batch geocoding via geocode_places.py
      2. Area centroid — last resort when coords are missing
    """
    if not place:
        return None

    lat = place.get("lat")
    lon = place.get("lon")
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        address = str(place.get("address") or "").strip()
        return ResolvedMapLocation(
            lat=float(lat),
            lon=float(lon),
            label=_display_label(place=place, fallback_address=address),
            address=address or str(place.get("name") or "").strip(),
            source="db_coordinates",
        )

    return None


def resolve_point_for_map(place: dict[str, Any] | None) -> tuple[float, float] | None:
    resolved = resolve_location_for_map(place)
    if not resolved:
        return None
    return resolved.lat, resolved.lon


def resolve_segment_points(
    a: dict[str, Any] | None,
    b: dict[str, Any] | None,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    a_loc, b_loc = resolve_segment_locations(a, b)
    a_point = (a_loc.lat, a_loc.lon) if a_loc else None
    b_point = (b_loc.lat, b_loc.lon) if b_loc else None
    return a_point, b_point


def resolve_segment_locations(
    a: dict[str, Any] | None,
    b: dict[str, Any] | None,
) -> tuple[ResolvedMapLocation | None, ResolvedMapLocation | None]:
    return resolve_location_for_map(a), resolve_location_for_map(b)


def place_map_url(place: dict[str, Any] | None) -> str:
    resolved = resolve_location_for_map(place)
    if not resolved:
        return ""
    return _trackasia_place_url(resolved)


def osm_directions_url(
    stops: list[dict | None],
    engine: str = "fossgis_osrm_car",
) -> str:
    del engine
    resolved_stops: list[ResolvedMapLocation] = []
    for stop in stops:
        resolved = resolve_location_for_map(stop)
        if not resolved:
            continue
        if resolved_stops and _same_coordinates(resolved_stops[-1], resolved):
            continue
        resolved_stops.append(resolved)
    if len(resolved_stops) < 2:
        return ""
    return _trackasia_route_url(resolved_stops[0], resolved_stops[-1])


def segment_map_url(
    a: dict | None,
    b: dict | None,
    engine: str = "fossgis_osrm_car",
) -> str:
    del engine
    a_loc, b_loc = resolve_segment_locations(a, b)
    if not a_loc or not b_loc or _same_coordinates(a_loc, b_loc):
        return ""
    return _trackasia_route_url(a_loc, b_loc)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = radians(lat1)
    p2 = radians(lat2)
    dp = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)
    a = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return 2 * r * asin(sqrt(a))


def _trackasia_place_url(loc: ResolvedMapLocation) -> str:
    token = _trackasia_point_token(loc)
    lat = f"{loc.lat:.7f}"
    lng = f"{loc.lon:.7f}"
    return f"{_TRACKASIA_HOST}/place/{token}#map={_TRACKASIA_DEFAULT_PLACE_ZOOM}/{lat}/{lng}"


def _trackasia_route_url(
    origin: ResolvedMapLocation,
    destination: ResolvedMapLocation,
    *,
    mode: str = _TRACKASIA_DEFAULT_MODE,
) -> str:
    query = (
        f"mode={quote(str(mode), safe='')}"
        f"&origin={_trackasia_point_token(origin)}"
        f"&destination={_trackasia_point_token(destination)}"
    )
    return f"{_TRACKASIA_HOST}/routes/?{query}"


def _trackasia_point_token(loc: ResolvedMapLocation) -> str:
    lat = f"{loc.lat:.6f}"
    lng = f"{loc.lon:.6f}"
    token = f"latlon:{lat}:{lng}"
    name = (loc.label or "").strip()
    if name:
        token = f"{token}@{quote(name, safe='')}"
    return token


def _display_label(
    *,
    place: dict[str, Any],
    fallback_name: str = "",
    fallback_address: str = "",
) -> str:
    name = str(place.get("name") or "").strip()
    if name:
        return name
    if fallback_name:
        return fallback_name
    if fallback_address:
        return fallback_address.split(",")[0].strip()
    return ""


def _same_coordinates(a: ResolvedMapLocation, b: ResolvedMapLocation) -> bool:
    return round(a.lat, 6) == round(b.lat, 6) and round(a.lon, 6) == round(b.lon, 6)


def _same_place(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    a_name = fold_text(str((a or {}).get("name") or ""))
    b_name = fold_text(str((b or {}).get("name") or ""))
    return bool(a_name and b_name and a_name == b_name)
