import json
import csv
import re
import unicodedata
from pathlib import Path
from functools import lru_cache

from app.places.scoring import add_fit_scores
from app.places.metadata import enrich_places
from app.places.repository import list_places
from app.places.vector_rag import retrieve_place_candidates

LOCAL_FETCH_MULTIPLIER = 2
MIN_LOCAL_FIT_TO_SKIP_EXTERNAL = 40.0

DEFAULT_CITIES = ["Đà Nẵng"]

_root = Path(__file__).resolve().parents[2]
_PROCESSED_DEST_FILE = _root / "data" / "crawl" / "processed" / "dest_danang.json"
_PROCESSED_VCGT_FILE = _root / "data" / "crawl" / "processed" / "vcgt_danang.csv"
_PROCESSED_REST_FILE = _root / "data" / "crawl" / "processed" / "rest_danang.json"
_PROCESSED_ACCOM_FILE = _root / "data" / "crawl" / "processed" / "cslt_danang.json"
_UNIFIED_FILE = _root / "data" / "processed" / "unified_places.json"


def search_processed_places(
    query: str,
    source_kind: str,
    top_k: int = 8,
) -> list[dict]:
    """Hybrid retrieval: Chroma vector search first, local catalog fallback, then merge."""
    terms = [_fold_text(t) for t in query.split() if t.strip()]
    if not terms:
        return []
    fetch_k = max(top_k * 3, top_k + 8, 24)
    rows = _search_unified_catalog(terms=terms, source_kind=source_kind, top_k=fetch_k)
    if not rows:
        if source_kind == "destinations":
            rows = _search_processed_dest_json(terms, fetch_k)
        elif source_kind == "entertainment":
            rows = _search_processed_vcgt_csv(terms, fetch_k)
        elif source_kind == "restaurants":
            rows = _search_processed_rest_json(terms, fetch_k)
        elif source_kind == "accommodations":
            rows = _search_processed_accommodation_json(terms, fetch_k)
        else:
            rows = []
    vector_rows = retrieve_place_candidates(
        query=query,
        source_kind=source_kind,
        top_k=fetch_k,
    )
    rows = _merge_retrieval_rows(rows, vector_rows)
    if not rows:
        return []
    enriched = enrich_places(rows)
    scored = add_fit_scores(enriched, query)
    for row in scored:
        has_lexical = bool(row.pop("_has_lexical_hit", False))
        has_vector = bool(row.pop("_has_vector_hit", False))
        lexical_origin = str(row.get("retrieval_origin") or "").strip() or "local_processed"
        if has_lexical and has_vector:
            if lexical_origin == "chroma":
                row["retrieval_tier"] = "hybrid_vector_chroma"
            else:
                row["retrieval_tier"] = "hybrid_vector_local"
        elif has_vector:
            row["retrieval_tier"] = "vector_rag"
        else:
            row.setdefault("retrieval_tier", lexical_origin)
    scored.sort(
        key=lambda item: (
            float(item.get("customer_fit_score") or 0.0),
            float(item.get("retrieval_relevance") or 0.0),
        ),
        reverse=True,
    )
    return scored[:top_k]


@lru_cache(maxsize=1)
def _load_unified_catalog() -> list[dict]:
    pg_rows = list_places()
    if pg_rows:
        return [dict(item) for item in pg_rows if isinstance(item, dict)]

    rows: list[dict] = []
    seen: set[str] = set()
    if _UNIFIED_FILE.exists():
        try:
            payload = json.loads(_UNIFIED_FILE.read_text(encoding="utf-8"))
        except Exception:
            payload = []
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                place_id = str(item.get("place_id") or "").strip().lower()
                key = place_id or f"{_fold_text(str(item.get('name') or ''))}|{_fold_text(str(item.get('address') or ''))}"
                if key in seen:
                    continue
                seen.add(key)
                rows.append(item)
    return rows


def _search_unified_catalog(terms: list[str], source_kind: str, top_k: int) -> list[dict]:
    category_map = {
        "destinations": {"destination"},
        "entertainment": {"entertainment"},
        "restaurants": {"restaurant"},
        "accommodations": {"accommodation"},
    }
    wanted_categories = category_map.get(source_kind)
    if not wanted_categories:
        return []

    rows: list[dict] = []
    for item in _load_unified_catalog():
        category = str(item.get("category") or "").strip().lower()
        if category not in wanted_categories:
            continue
        city = str(item.get("city") or "")
        if city and city not in DEFAULT_CITIES and "da nang" not in _fold_text(city) and "danang" not in _fold_text(city):
            continue
        blob = _fold_text(
            " ".join(
                [
                    str(item.get("name") or ""),
                    str(item.get("description") or ""),
                    str(item.get("detail_content") or ""),
                    str(item.get("list_snippet") or ""),
                    str(item.get("address") or ""),
                    str(item.get("district") or ""),
                    str(item.get("city") or ""),
                    str(item.get("google_formatted_address") or ""),
                    str(item.get("google_primary_type") or ""),
                    " ".join(str(value) for value in (item.get("google_types") or []) if str(value).strip()),
                    str(item.get("map_formatted_address") or ""),
                    str(item.get("map_primary_type") or ""),
                    " ".join(str(value) for value in (item.get("map_types") or []) if str(value).strip()),
                ]
            )
        )
        rel = _score_blob(terms, blob)
        if rel <= 0:
            continue
        row = dict(item)
        row["retrieval_score"] = 0.0
        row["retrieval_relevance"] = round(rel, 4)
        row["retrieval_origin"] = "local_processed"
        rows.append(row)
    rows.sort(key=lambda x: float(x.get("retrieval_relevance") or 0), reverse=True)
    return rows[:top_k]


def _score_blob(terms: list[str], blob: str) -> float:
    if not terms:
        return 0.0
    hits = sum(1 for t in terms if t in blob)
    if hits <= 0:
        return 0.0
    return min(1.0, hits / max(len(terms), 1))


def _merge_retrieval_rows(primary_rows: list[dict], vector_rows: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}

    def row_key(row: dict) -> str:
        place_id = str(row.get("place_id") or "").strip().lower()
        if place_id:
            return f"place:{place_id}"
        name = _fold_text(str(row.get("name") or ""))
        address = _fold_text(str(row.get("address") or ""))
        return f"name:{name}|addr:{address}"

    def ingest(row: dict, *, from_vector: bool) -> None:
        key = row_key(row)
        current = merged.get(key)
        incoming = dict(row)
        incoming["_has_vector_hit"] = bool(from_vector)
        incoming["_has_lexical_hit"] = not from_vector
        if current is None:
            merged[key] = incoming
            return

        current_rel = float(current.get("retrieval_relevance") or 0.0)
        incoming_rel = float(incoming.get("retrieval_relevance") or 0.0)
        current["retrieval_relevance"] = round(max(current_rel, incoming_rel), 4)
        current["retrieval_score"] = max(
            float(current.get("retrieval_score") or 0.0),
            float(incoming.get("retrieval_score") or 0.0),
        )
        current["_has_vector_hit"] = bool(current.get("_has_vector_hit")) or bool(incoming.get("_has_vector_hit"))
        current["_has_lexical_hit"] = bool(current.get("_has_lexical_hit")) or bool(incoming.get("_has_lexical_hit"))

        if incoming_rel > current_rel:
            preferred, secondary = incoming, current
        else:
            preferred, secondary = current, incoming
        preferred["_has_vector_hit"] = bool(current.get("_has_vector_hit")) or bool(incoming.get("_has_vector_hit"))
        preferred["_has_lexical_hit"] = bool(current.get("_has_lexical_hit")) or bool(incoming.get("_has_lexical_hit"))
        for field, value in secondary.items():
            if field.startswith("_"):
                continue
            if preferred.get(field) in (None, "", [], {}):
                preferred[field] = value
        snippets = []
        for candidate in [current.get("rag_snippets"), incoming.get("rag_snippets")]:
            if not isinstance(candidate, list):
                continue
            for snippet in candidate:
                text = str(snippet).strip()
                if text and text not in snippets:
                    snippets.append(text)
        if snippets:
            preferred["rag_snippets"] = snippets[:2]
        merged[key] = preferred

    for row in primary_rows:
        ingest(row, from_vector=False)
    for row in vector_rows:
        ingest(row, from_vector=True)

    return list(merged.values())


def _search_processed_dest_json(terms: list[str], top_k: int) -> list[dict]:
    try:
        data = json.loads(_PROCESSED_DEST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    rows: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        city = str(item.get("city") or "")
        if city and city not in DEFAULT_CITIES:
            continue
        blob = _fold_text(" ".join([
            str(item.get("name") or ""),
            str(item.get("description") or ""),
            str(item.get("detail_content") or ""),
            str(item.get("list_snippet") or ""),
            str(item.get("address") or ""),
            str(item.get("district") or ""),
            city,
        ]))
        rel = _score_blob(terms, blob)
        if rel <= 0:
            continue
        rows.append({
            "name": item.get("name"),
            "category": "destination",
            "description": item.get("description") or "",
            "detail_content": item.get("detail_content") or "",
            "address": item.get("address") or "",
            "district": item.get("district") or "",
            "destination_type": item.get("destination_type") or "",
            "city": city,
            "list_snippet": item.get("list_snippet") or "",
            "lat": item.get("lat"),
            "lon": item.get("lon"),
            "detail_url": item.get("detail_url") or "",
            "item_id": item.get("item_id") or "",
            "source_category_code": item.get("source_category_code") or "",
            "phone": item.get("phone") or "",
            "website": item.get("website") or "",
            "source": "local-processed:dest_danang.json",
            "retrieval_score": 0.0,
            "retrieval_relevance": round(rel, 4),
            "retrieval_origin": "local_processed",
        })
    rows.sort(key=lambda x: float(x.get("retrieval_relevance") or 0), reverse=True)
    return rows[:top_k]


def _search_processed_vcgt_csv(terms: list[str], top_k: int) -> list[dict]:
    try:
        text = _PROCESSED_VCGT_FILE.read_text(encoding="utf-8")
    except Exception:
        return []
    rows: list[dict] = []
    reader = csv.DictReader(text.splitlines())
    for item in reader:
        city = str(item.get("city") or "")
        if city and city not in DEFAULT_CITIES:
            continue
        blob = _fold_text(" ".join([
            str(item.get("name") or ""),
            str(item.get("description") or ""),
            str(item.get("list_snippet") or ""),
            str(item.get("address") or ""),
            str(item.get("district") or ""),
            city,
        ]))
        rel = _score_blob(terms, blob)
        if rel <= 0:
            continue
        rows.append({
            "name": item.get("name"),
            "category": "entertainment",
            "description": item.get("description") or "",
            "address": item.get("address") or "",
            "district": item.get("district") or "",
            "city": city,
            "list_snippet": item.get("list_snippet") or "",
            "lat": None,
            "lon": None,
            "detail_url": item.get("detail_url") or "",
            "item_id": item.get("item_id") or "",
            "source_category_code": item.get("source_category_code") or "",
            "phone": item.get("phone") or "",
            "website": item.get("website") or "",
            "source": "local-processed:vcgt_danang.csv",
            "retrieval_score": 0.0,
            "retrieval_relevance": round(rel, 4),
            "retrieval_origin": "local_processed",
        })
    rows.sort(key=lambda x: float(x.get("retrieval_relevance") or 0), reverse=True)
    return rows[:top_k]


def _search_processed_rest_json(terms: list[str], top_k: int) -> list[dict]:
    try:
        data = json.loads(_PROCESSED_REST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    rows: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        city = str(item.get("city") or "")
        if city and city not in DEFAULT_CITIES:
            continue
        blob = _fold_text(" ".join([
            str(item.get("name") or ""),
            str(item.get("description") or ""),
            str(item.get("list_snippet") or ""),
            str(item.get("address") or ""),
            str(item.get("district") or ""),
            city,
        ]))
        rel = _score_blob(terms, blob)
        if rel <= 0:
            continue
        rows.append({
            "name": item.get("name"),
            "category": "restaurant",
            "description": item.get("description") or "",
            "address": item.get("address") or "",
            "district": item.get("district") or "",
            "city": city,
            "list_snippet": item.get("list_snippet") or "",
            "lat": item.get("lat"),
            "lon": item.get("lon"),
            "detail_url": item.get("detail_url") or "",
            "item_id": item.get("item_id") or "",
            "source_category_code": item.get("source_category_code") or "",
            "phone": item.get("phone") or "",
            "website": item.get("website") or "",
            "source": "local-processed:rest_danang.json",
            "retrieval_score": 0.0,
            "retrieval_relevance": round(rel, 4),
            "retrieval_origin": "local_processed",
        })
    rows.sort(key=lambda x: float(x.get("retrieval_relevance") or 0), reverse=True)
    return rows[:top_k]


def _search_processed_accommodation_json(terms: list[str], top_k: int) -> list[dict]:
    try:
        data = json.loads(_PROCESSED_ACCOM_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    rows: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        city = str(item.get("city") or "")
        if city and city not in DEFAULT_CITIES:
            continue
        blob = _fold_text(" ".join([
            str(item.get("name") or ""),
            str(item.get("description") or ""),
            str(item.get("list_snippet") or ""),
            str(item.get("address") or ""),
            str(item.get("district") or ""),
            str(item.get("star_rating") or ""),
            str(item.get("accommodation_type") or ""),
            city,
        ]))
        rel = _score_blob(terms, blob)
        if rel <= 0:
            continue
        rows.append({
            "name": item.get("name"),
            "category": "accommodation",
            "description": item.get("description") or "",
            "detail_content": item.get("detail_content") or "",
            "address": item.get("address") or "",
            "district": item.get("district") or "",
            "city": city,
            "list_snippet": item.get("list_snippet") or "",
            "lat": item.get("lat"),
            "lon": item.get("lon"),
            "detail_url": item.get("detail_url") or "",
            "item_id": item.get("item_id") or "",
            "source_category_code": item.get("source_category_code") or "",
            "phone": item.get("phone") or "",
            "website": item.get("website") or "",
            "star_rating": item.get("star_rating") or "",
            "price_range": item.get("price_range") or "",
            "num_rooms": item.get("num_rooms") or "",
            "accommodation_type": item.get("accommodation_type") or "",
            "listing_source_type": item.get("listing_source_type") or "",
            "source": "local-processed:cslt_danang.json",
            "retrieval_score": 0.0,
            "retrieval_relevance": round(rel, 4),
            "retrieval_origin": "local_processed",
        })
    rows.sort(key=lambda x: float(x.get("retrieval_relevance") or 0), reverse=True)
    return rows[:top_k]


def _fold_text(text: str) -> str:
    base = (text or "").replace("đ", "d").replace("Đ", "D")
    n = unicodedata.normalize("NFD", base)
    no_marks = "".join(ch for ch in n if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", no_marks).strip().lower()
