"""Re-geocode all places in the DB via TrackAsia Text Search API.

Output: data/geocode_results.csv
Columns: place_id, name, category, original_lat, original_lon,
         new_lat, new_lon, new_address, ta_place_id, query_used, status

Status values: found | not_found | error

The script is resumable: already-processed place_ids are skipped.
Rate: 2 seconds between API calls.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUTPUT_CSV = ROOT / "data" / "geocode_results.csv"
DB_PATH = ROOT / "data" / "travel.db"
RATE_DELAY_S = 2.0
TEXTSEARCH_URL = "https://maps.track-asia.com/api/v2/place/textsearch/json"
REQUEST_TIMEOUT_S = 10

FIELDNAMES = [
    "place_id", "name", "category",
    "original_lat", "original_lon",
    "new_lat", "new_lon", "new_address", "ta_place_id",
    "query_used", "status",
]


def _load_api_key() -> str:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("TRACKASIA_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("TRACKASIA_API_KEY", "").strip()


def _load_places() -> list[dict]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT payload_json FROM places")
    rows = cur.fetchall()
    conn.close()
    out = []
    for row in rows:
        try:
            p = json.loads(row["payload_json"])
            if isinstance(p, dict):
                out.append(p)
        except Exception:
            pass
    return out


def _load_done_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    done = set()
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("place_id") or "").strip()
            if pid:
                done.add(pid)
    return done


def _build_query(place: dict) -> str:
    name = str(place.get("name") or "").strip()
    address = str(place.get("address") or "").strip()
    city = str(place.get("city") or "Đà Nẵng").strip()
    if address:
        return f"{name} {address}"
    return f"{name} {city}"


def _textsearch(query: str, api_key: str) -> dict | None:
    params = {
        "query": query,
        "key": api_key,
        "new_admin": "true",
        "include_old_admin": "true",
    }
    url = f"{TEXTSEARCH_URL}?{urllib.parse.urlencode(params, safe=',')}"
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "geocode-script/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _parse_first_result(data: dict) -> dict | None:
    if str(data.get("status") or "").upper() not in {"OK", ""}:
        return None
    results = data.get("results") or []
    if not results or not isinstance(results[0], dict):
        return None
    item = results[0]
    geometry = item.get("geometry") if isinstance(item.get("geometry"), dict) else {}
    location = geometry.get("location") if isinstance(geometry.get("location"), dict) else {}
    lat = location.get("lat")
    lon = location.get("lng")
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return None
    address = str(
        item.get("formatted_address")
        or item.get("old_formatted_address")
        or item.get("name")
        or ""
    ).strip()
    ta_place_id = str(item.get("place_id") or "").strip()
    return {"lat": lat, "lon": lon, "address": address, "ta_place_id": ta_place_id}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    api_key = _load_api_key()
    if not api_key:
        print("ERROR: TRACKASIA_API_KEY not found in .env or environment.")
        sys.exit(1)

    places = _load_places()
    print(f"Loaded {len(places)} places from DB.")

    done_ids = _load_done_ids(OUTPUT_CSV)
    remaining = [p for p in places if str(p.get("place_id") or "").strip() not in done_ids]
    print(f"Already done: {len(done_ids)} | Remaining: {len(remaining)}")

    if not remaining:
        print("All places already geocoded.")
        return

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0

    with open(OUTPUT_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        total = len(remaining)
        found = not_found = errors = 0

        for i, place in enumerate(remaining, start=1):
            place_id = str(place.get("place_id") or "").strip()
            name = str(place.get("name") or "").strip()
            category = str(place.get("category") or "").strip()
            orig_lat = place.get("lat")
            orig_lon = place.get("lon")
            query = _build_query(place)

            print(f"[{i}/{total}] {name[:50]} ...", end=" ", flush=True)

            data = _textsearch(query, api_key)

            if data is None:
                status = "error"
                row = {
                    "place_id": place_id, "name": name, "category": category,
                    "original_lat": orig_lat, "original_lon": orig_lon,
                    "new_lat": "", "new_lon": "", "new_address": "", "ta_place_id": "",
                    "query_used": query, "status": status,
                }
                errors += 1
                print("ERROR (network/timeout)")
            else:
                result = _parse_first_result(data)
                if result:
                    status = "found"
                    row = {
                        "place_id": place_id, "name": name, "category": category,
                        "original_lat": orig_lat, "original_lon": orig_lon,
                        "new_lat": result["lat"], "new_lon": result["lon"],
                        "new_address": result["address"], "ta_place_id": result["ta_place_id"],
                        "query_used": query, "status": status,
                    }
                    found += 1
                    print(f"OK  lat={result['lat']:.5f} lon={result['lon']:.5f}")
                else:
                    status = "not_found"
                    row = {
                        "place_id": place_id, "name": name, "category": category,
                        "original_lat": orig_lat, "original_lon": orig_lon,
                        "new_lat": "", "new_lon": "", "new_address": "", "ta_place_id": "",
                        "query_used": query, "status": status,
                    }
                    not_found += 1
                    print("NOT FOUND")

            writer.writerow(row)
            f.flush()

            if i < total:
                time.sleep(RATE_DELAY_S)

    print()
    print(f"Done. found={found} not_found={not_found} errors={errors}")
    print(f"Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
