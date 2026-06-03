"""
Batch-geocode places in SQLite using TrackAsia text search.

Two-phase workflow:
  Phase 1 (--clear-centroids): NULL out lat/lon for all places whose stored
           coordinate is a known area centroid (generic fallback value).
  Phase 2 (default): Geocode all places with lat IS NULL using TrackAsia.

Usage:
    python scripts/geocode_places.py --dry-run          # preview queries, no writes
    python scripts/geocode_places.py --clear-centroids  # phase 1: null centroid coords
    python scripts/geocode_places.py                    # phase 2: geocode missing coords
    python scripts/geocode_places.py --limit 50         # process first 50 only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in {"utf-8", "utf8"}:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.database import get_connection

_CENTROIDS: set[tuple[float, float]] = {
    (round(lat, 4), round(lon, 4))
    for lat, lon in [
        (16.0544, 108.2207), (16.0975, 108.2637), (16.0207, 108.2522),
        (16.0678, 108.1960), (16.0162, 108.2045), (16.0749, 108.1482),
        (15.9967, 108.0678), (15.8801, 108.3380), (15.5736, 108.4740),
        (15.8927, 108.2538), (15.7897, 108.1204), (15.8796, 107.9806),
        (15.6940, 108.3283), (15.4697, 108.2847), (15.4330, 108.6187),
        (16.0544, 108.2022), (15.5394, 108.0191),
    ]
}

_LAT_MIN, _LAT_MAX = 15.3, 16.5
_LON_MIN, _LON_MAX = 107.5, 108.7

_TEXTSEARCH_URL = "https://maps.track-asia.com/api/v2/place/textsearch/json"

# Adaptive rate limiter state (module-level so phase_geocode can share it)
_rate_delay: float = 0.3   # start fast
_RATE_MIN: float = 0.15    # floor — don't go below this
_RATE_MAX: float = 30.0    # ceiling on backoff
_RATE_BACKOFF: float = 2.0 # multiply on 429
_RATE_RECOVER: float = 0.9 # multiply on success (slowly speed up)


def _is_centroid(lat: float | None, lon: float | None) -> bool:
    if lat is None or lon is None:
        return False
    return (round(float(lat), 4), round(float(lon), 4)) in _CENTROIDS


def _build_query(place: dict) -> str:
    name = re.sub(r"[&+#@]", " ", str(place.get("name") or "").strip())
    address = str(place.get("address") or "").strip()
    city = str(place.get("city") or "Đà Nẵng").strip()
    if address:
        return f"{name}, {address}".strip()
    return f"{name} {city} Vietnam".strip()


def _geocode_raw(query: str, api_key: str, timeout_s: int = 8) -> tuple[float, float] | None | str:
    """Call TrackAsia text search directly.

    Returns:
        (lat, lon)  — found
        None        — not found / bad result
        "rate_limit" — HTTP 429, caller should back off and retry
        "transient"  — timeout / 5xx, caller should retry
    """
    params = {"query": query, "key": api_key, "new_admin": "true", "include_old_admin": "true"}
    url = f"{_TEXTSEARCH_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "geocode-script/1.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 429:
            return "rate_limit"
        if e.code >= 500:
            return "transient"
        return None
    except (urllib.error.URLError, TimeoutError, OSError):
        return "transient"
    except Exception:
        return None

    if str(data.get("status") or "").upper() not in {"OK", ""}:
        return None
    for item in (data.get("results") or []):
        try:
            loc = item.get("geometry", {}).get("location", {})
            lat, lon = float(loc["lat"]), float(loc["lng"])
        except (KeyError, TypeError, ValueError):
            continue
        if _LAT_MIN <= lat <= _LAT_MAX and _LON_MIN <= lon <= _LON_MAX:
            return lat, lon
    return None


def _geocode(query: str, api_key: str, *, max_retries: int = 4) -> tuple[float, float] | None:
    """Geocode with adaptive rate control and retry logic."""
    global _rate_delay
    attempt = 0
    while attempt <= max_retries:
        result = _geocode_raw(query, api_key)

        if isinstance(result, tuple):
            # Success — gradually recover speed
            _rate_delay = max(_RATE_MIN, _rate_delay * _RATE_RECOVER)
            return result

        if result == "rate_limit":
            wait = min(_rate_delay * _RATE_BACKOFF, _RATE_MAX)
            _rate_delay = wait
            print(f"[429 rate-limit, wait {wait:.1f}s]", end=" ", flush=True)
            time.sleep(wait)
            attempt += 1
            continue

        if result == "transient":
            wait = min(1.5 * (2 ** attempt), 20.0)
            print(f"[transient, wait {wait:.1f}s]", end=" ", flush=True)
            time.sleep(wait)
            attempt += 1
            continue

        return None  # definitive not-found

    return None


def _update_coords(conn, place_id: str, lat: float | None, lon: float | None, payload_json: str) -> None:
    try:
        payload = json.loads(payload_json or "{}")
    except Exception:
        payload = {}
    payload["lat"] = lat
    payload["lon"] = lon
    conn.execute(
        "UPDATE places SET lat=?, lon=?, payload_json=?, updated_at=datetime('now') WHERE place_id=?",
        (lat, lon, json.dumps(payload, ensure_ascii=False), place_id),
    )


def phase_clear_centroids(conn, *, dry_run: bool) -> int:
    rows = conn.execute(
        "SELECT place_id, name, lat, lon, payload_json FROM places WHERE lat IS NOT NULL AND lon IS NOT NULL"
    ).fetchall()
    count = 0
    for row in rows:
        if _is_centroid(row["lat"], row["lon"]):
            if dry_run:
                print(f"  WOULD NULL  ({row['lat']}, {row['lon']})  {row['name'][:60]}")
            else:
                _update_coords(conn, row["place_id"], None, None, row["payload_json"])
            count += 1
    if not dry_run and count:
        conn.commit()
    return count


def phase_geocode(conn, *, dry_run: bool, limit: int, api_key: str) -> tuple[int, int, int]:
    global _rate_delay
    rows = conn.execute("""
        SELECT place_id, name, category, address, city, district, payload_json
        FROM places
        WHERE lat IS NULL OR lon IS NULL
        ORDER BY
            CASE category
                WHEN 'destination'   THEN 1
                WHEN 'restaurant'    THEN 2
                WHEN 'entertainment' THEN 3
                ELSE 4
            END,
            name
    """).fetchall()

    if limit > 0:
        rows = rows[:limit]

    ok = fail = skip = 0
    last_t = 0.0

    for i, row in enumerate(rows, 1):
        query = _build_query(dict(row))
        cat = row["category"] or "?"
        name = row["name"] or "?"
        print(f"[{i}/{len(rows)}] [{cat}] {name[:50]}", end=" ... ", flush=True)

        if not query.strip():
            print("SKIP (no query)")
            skip += 1
            continue

        if dry_run:
            print(f"query={query!r}")
            skip += 1
            continue

        # Adaptive pacing: wait the current delay since last request
        elapsed = time.monotonic() - last_t
        if elapsed < _rate_delay:
            time.sleep(_rate_delay - elapsed)
        last_t = time.monotonic()

        result = _geocode(query, api_key)
        if result:
            lat, lon = result
            _update_coords(conn, row["place_id"], lat, lon, row["payload_json"])
            conn.commit()
            print(f"OK  ({lat:.5f}, {lon:.5f})  [delay={_rate_delay:.2f}s]")
            ok += 1
        else:
            print("FAIL")
            fail += 1

    return ok, fail, skip


def main() -> None:
    parser = argparse.ArgumentParser(description="Geocode places via TrackAsia text search")
    parser.add_argument("--clear-centroids", action="store_true",
                        help="Phase 1: NULL out all centroid-coord places, then exit")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max places to geocode in phase 2 (0 = all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview actions without writing")
    args = parser.parse_args()

    from app.core.settings import get_settings
    s = get_settings()
    api_key = str(s.trackasia_api_key or "").strip()
    if not (api_key and s.trackasia_enabled and s.trackasia_geocode_enabled):
        print("ERROR: TrackAsia geocoding is not enabled. Check TRACKASIA_API_KEY in .env")
        sys.exit(1)

    conn = get_connection()

    if args.clear_centroids:
        print("Phase 1: clearing centroid coordinates ...")
        if args.dry_run:
            print("DRY RUN — no writes\n")
        count = phase_clear_centroids(conn, dry_run=args.dry_run)
        print(f"{'Would clear' if args.dry_run else 'Cleared'} {count} centroid place(s).")
        conn.close()
        return

    print(f"Phase 2: geocoding with adaptive rate (start={_rate_delay}s, min={_RATE_MIN}s, max={_RATE_MAX}s) ...")
    if args.dry_run:
        print("DRY RUN — no writes\n")
    ok, fail, skip = phase_geocode(conn, dry_run=args.dry_run, limit=args.limit, api_key=api_key)
    conn.close()

    print(f"\nDone: {ok} updated, {fail} failed, {skip} skipped")
    if ok > 0 and not args.dry_run:
        print("Run 'python scripts/ingest_to_chroma.py --recreate' to rebuild Chroma embeddings.")


if __name__ == "__main__":
    main()
