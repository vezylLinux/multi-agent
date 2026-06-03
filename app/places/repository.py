from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from app.core.database import get_connection, get_cursor
from app.places.metadata import enrich_place_record


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def clear_place_caches() -> None:
    list_places.cache_clear()
    load_places_lookup.cache_clear()
    list_place_chunks.cache_clear()
    try:
        from app.places.search import _load_unified_catalog

        _load_unified_catalog.cache_clear()
    except Exception:
        pass
    try:
        from app.places.vector_rag import _load_rag_documents, _load_unified_places_by_id

        _load_rag_documents.cache_clear()
        _load_unified_places_by_id.cache_clear()
    except Exception:
        pass


def upsert_places(places: list[dict[str, Any]]) -> int:
    if not places:
        clear_place_caches()
        return 0

    now = _utc_now()
    values: list[tuple[Any, ...]] = []
    for raw in places:
        enriched = enrich_place_record(raw)
        place_id = str(enriched.get("place_id") or "").strip()
        name = str(enriched.get("name") or "").strip()
        category = str(enriched.get("category") or "").strip()
        if not place_id or not name or not category:
            continue
        values.append(
            (
                place_id,
                name,
                category,
                str(enriched.get("city") or "").strip() or None,
                str(enriched.get("city_key") or "").strip() or None,
                str(enriched.get("district") or "").strip() or None,
                str(enriched.get("ward") or "").strip() or None,
                str(enriched.get("address") or "").strip() or None,
                str(enriched.get("description") or "").strip() or None,
                str(enriched.get("detail_content") or "").strip() or None,
                str(enriched.get("list_snippet") or "").strip() or None,
                str(enriched.get("source") or "").strip() or None,
                str(enriched.get("source_category_code") or "").strip() or None,
                str(enriched.get("destination_type") or "").strip() or None,
                str(enriched.get("item_id") or "").strip() or None,
                str(enriched.get("detail_url") or "").strip() or None,
                str(enriched.get("website") or "").strip() or None,
                str(enriched.get("phone") or "").strip() or None,
                str(enriched.get("planner_role") or "").strip() or None,
                str(enriched.get("primary_area_key") or "").strip() or None,
                _json_text(enriched.get("admin_area_keys") or []),
                _json_text(enriched.get("intent_tags") or []),
                str(enriched.get("density_bucket") or "").strip() or None,
                str(enriched.get("verification_status") or "").strip() or None,
                enriched.get("lat"),
                enriched.get("lon"),
                _json_text(enriched),
                now,
                now,
            )
        )

    if not values:
        clear_place_caches()
        return 0

    sql = """
        INSERT INTO places (
            place_id, name, category, city, city_key, district, ward,
            address, description, detail_content, list_snippet, source,
            source_category_code, destination_type, item_id, detail_url,
            website, phone, planner_role, primary_area_key,
            admin_area_keys_json, intent_tags_json, density_bucket,
            verification_status, lat, lon, payload_json, created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(place_id) DO UPDATE SET
            name = excluded.name,
            category = excluded.category,
            city = excluded.city,
            city_key = excluded.city_key,
            district = excluded.district,
            ward = excluded.ward,
            address = excluded.address,
            description = excluded.description,
            detail_content = excluded.detail_content,
            list_snippet = excluded.list_snippet,
            source = excluded.source,
            source_category_code = excluded.source_category_code,
            destination_type = excluded.destination_type,
            item_id = excluded.item_id,
            detail_url = excluded.detail_url,
            website = excluded.website,
            phone = excluded.phone,
            planner_role = excluded.planner_role,
            primary_area_key = excluded.primary_area_key,
            admin_area_keys_json = excluded.admin_area_keys_json,
            intent_tags_json = excluded.intent_tags_json,
            density_bucket = excluded.density_bucket,
            verification_status = excluded.verification_status,
            lat = excluded.lat,
            lon = excluded.lon,
            payload_json = excluded.payload_json,
            updated_at = excluded.updated_at
    """
    connection = get_connection()
    try:
        cursor = connection.cursor()
        cursor.executemany(sql, values)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    clear_place_caches()
    return len(values)


@lru_cache(maxsize=1)
def list_places() -> list[dict[str, Any]]:
    with get_cursor() as cursor:
        cursor.execute(
            """
            SELECT payload_json
            FROM places
            ORDER BY city_key NULLS LAST, category, primary_area_key NULLS LAST, name
            """
        )
        rows = cursor.fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload_json")
        if not payload:
            continue
        try:
            item = json.loads(payload)
        except Exception:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


@lru_cache(maxsize=1)
def load_places_lookup() -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for place in list_places():
        place_id = str(place.get("place_id") or "").strip()
        if place_id:
            lookup[place_id] = dict(place)
    return lookup


def replace_place_chunks(documents: list[dict[str, Any]]) -> int:
    now = _utc_now()
    connection = get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM place_chunks")
        if documents:
            chunk_values = [
                (
                    str(doc.get("doc_id") or "").strip(),
                    str(doc.get("place_id") or "").strip(),
                    str(doc.get("title") or "").strip() or None,
                    str(doc.get("city") or "").strip() or None,
                    str(doc.get("category") or "").strip() or None,
                    int(doc.get("chunk_index") or 0),
                    str(doc.get("document_text") or "").strip(),
                    _json_text(doc.get("metadata") or {}),
                    _json_text(doc.get("embedding")) if isinstance(doc.get("embedding"), list) else None,
                    str(doc.get("embedding_model") or "").strip() or None,
                    now,
                    now,
                )
                for doc in documents
                if str(doc.get("doc_id") or "").strip()
                and str(doc.get("place_id") or "").strip()
                and str(doc.get("document_text") or "").strip()
            ]
            if chunk_values:
                cursor.executemany(
                    """
                    INSERT INTO place_chunks (
                        doc_id, place_id, title, city, category, chunk_index,
                        document_text, metadata_json, embedding_json,
                        embedding_model, created_at, updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    chunk_values,
                )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    clear_place_caches()
    return len(documents)


@lru_cache(maxsize=1)
def list_place_chunks() -> list[dict[str, Any]]:
    with get_cursor() as cursor:
        cursor.execute(
            """
            SELECT
                doc_id,
                place_id,
                title,
                city,
                category,
                chunk_index,
                document_text,
                metadata_json,
                embedding_json,
                embedding_model
            FROM place_chunks
            ORDER BY place_id, chunk_index
            """
        )
        rows = cursor.fetchall()
    documents: list[dict[str, Any]] = []
    for row in rows:
        metadata = {}
        embedding = None
        if row.get("metadata_json"):
            try:
                loaded = json.loads(row["metadata_json"])
                if isinstance(loaded, dict):
                    metadata = loaded
            except Exception:
                metadata = {}
        if row.get("embedding_json"):
            try:
                loaded = json.loads(row["embedding_json"])
                if isinstance(loaded, list):
                    embedding = loaded
            except Exception:
                embedding = None
        documents.append(
            {
                "doc_id": row["doc_id"],
                "place_id": row["place_id"],
                "title": row.get("title") or "",
                "city": row.get("city") or "",
                "category": row.get("category") or "",
                "chunk_index": int(row.get("chunk_index") or 0),
                "document_text": row.get("document_text") or "",
                "metadata": metadata,
                "embedding": embedding,
                "embedding_model": row.get("embedding_model") or "",
            }
        )
    return documents
