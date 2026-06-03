"""
Geocode places missing lat/lon using Google Places API (Text Search).

Processes by priority: destination → restaurant → entertainment → accommodation
Skips places that already have coordinates.

Usage:
    python scripts/geocode_google.py               # all missing coords
    python scripts/geocode_google.py --limit 79    # first N by priority
    python scripts/geocode_google.py --dry-run     # preview queries, no writes
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in {"utf-8", "utf8"}:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.core.database import get_connection
from app.tools.google_places import google_places_available, resolve_place_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max places to process (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no DB writes")
    parser.add_argument("--min-score", type=float, default=0.5, help="Min name match score to accept (default 0.5)")
    args = parser.parse_args()

    if not google_places_available():
        print("ERROR: GOOGLE_MAPS_API_KEY not set in .env")
        sys.exit(1)

    conn = get_connection()
    conn.row_factory = __import__("sqlite3").Row

    rows = conn.execute("""
        SELECT place_id, name, category, address, city, district, payload_json
        FROM places
        WHERE (lat IS NULL OR lon IS NULL OR lat = 0)
        ORDER BY
            CASE category
                WHEN 'destination'   THEN 1
                WHEN 'restaurant'    THEN 2
                WHEN 'entertainment' THEN 3
                ELSE 4
            END,
            name
    """).fetchall()

    if args.limit > 0:
        rows = rows[:args.limit]

    total = len(rows)
    print(f"Places needing coordinates: {total}")
    if args.dry_run:
        print("(dry-run mode — no writes)")

    ok = fail = skip = 0
    for i, row in enumerate(rows, 1):
        place = dict(row)
        name = place.get("name") or "?"
        cat = place.get("category") or "?"
        print(f"[{i}/{total}] [{cat}] {name[:50]} ...", end=" ", flush=True)

        if args.dry_run:
            print(f"query would be: {name!r}")
            skip += 1
            continue

        resolution = resolve_place_record(place)
        if resolution is None or resolution.lat is None:
            print("FAIL")
            fail += 1
            time.sleep(0.1)
            continue

        if resolution.match_score < args.min_score:
            print(f"LOW_SCORE ({resolution.match_score:.2f}) '{resolution.display_name[:40]}'")
            fail += 1
            time.sleep(0.1)
            continue

        import json as _json
        try:
            payload = _json.loads(place.get("payload_json") or "{}")
        except Exception:
            payload = {}
        payload["lat"] = resolution.lat
        payload["lon"] = resolution.lon
        payload["google_place_id"] = resolution.place_id
        payload["google_maps_uri"] = resolution.google_maps_uri

        conn.execute(
            """UPDATE places
               SET lat=?, lon=?, google_place_id=?, google_maps_uri=?,
                   payload_json=?, updated_at=datetime('now')
               WHERE place_id=?""",
            (
                resolution.lat, resolution.lon,
                resolution.place_id, resolution.google_maps_uri,
                _json.dumps(payload, ensure_ascii=False),
                place["place_id"],
            ),
        )
        conn.commit()
        print(f"OK ({resolution.lat:.5f}, {resolution.lon:.5f}) score={resolution.match_score:.2f}")
        ok += 1
        time.sleep(0.05)

    print(f"\nDone. ok={ok}  fail/low_score={fail}  skip={skip}")


if __name__ == "__main__":
    main()
