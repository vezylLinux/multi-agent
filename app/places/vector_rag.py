from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from app.core.settings import get_settings
from app.places.repository import list_place_chunks, load_places_lookup
from app.places.metadata import enrich_place_record, fold_text

ROOT = Path(__file__).resolve().parents[2]
RAG_JSON = ROOT / "data" / "rag" / "rag_documents.json"
UNIFIED_JSON = ROOT / "data" / "processed" / "unified_places.json"
_DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
_SOURCE_KIND_CATEGORY_MAP = {
    "destinations": {"destination"},
    "entertainment": {"entertainment"},
    "restaurants": {"restaurant"},
    "accommodations": {"accommodation"},
}


@dataclass(frozen=True)
class ChunkHit:
    doc_id: str
    place_id: str
    title: str
    category: str
    city: str
    document_text: str
    metadata: dict[str, Any]
    score: float


def place_document_text(place: dict[str, Any]) -> str:
    name = str(place.get("name") or "").strip()
    category = _category_label(str(place.get("category") or ""))
    area = ", ".join(
        part
        for part in [
            str(place.get("district") or "").strip(),
            str(place.get("city") or "").strip(),
        ]
        if part
    )
    address = str(place.get("address") or "").strip()
    planner_role = _planner_role_label(str(place.get("planner_role") or ""))
    intent_tags = ", ".join(str(tag) for tag in (place.get("intent_tags") or []) if str(tag).strip())
    description = _truncate_text(
        " ".join(
            part
            for part in [
                str(place.get("description") or "").strip(),
                str(place.get("detail_content") or "").strip(),
                str(place.get("list_snippet") or "").strip(),
            ]
            if part
        ),
        limit=1200,
    )
    parts = [
        f"{name}.",
        f"Loai: {category}.",
        f"Vai tro lap lich: {planner_role}.",
    ]
    if area:
        parts.append(f"Khu vuc: {area}.")
    if address:
        parts.append(f"Dia chi: {address}.")
    if intent_tags:
        parts.append(f"Intent tags: {intent_tags}.")
    if str(place.get("destination_type") or "").strip():
        parts.append(f"Loai hinh: {str(place.get('destination_type')).strip()}.")
    if description:
        parts.append(f"Mo ta: {description}.")
    return " ".join(parts).strip()


def chunk_text(
    text: str,
    *,
    max_words: int | None = None,
    overlap_words: int | None = None,
) -> list[str]:
    settings = get_settings()
    max_words = max_words or max(60, int(settings.rag_chunk_size_words or 120))
    overlap_words = overlap_words if overlap_words is not None else int(settings.rag_chunk_overlap_words or 24)
    overlap_words = max(0, min(overlap_words, max_words // 2))

    compact = " ".join((text or "").split())
    if not compact:
        return []

    sentences = [part.strip() for part in re.split(r"(?<=[\.\!\?])\s+", compact) if part.strip()]
    if not sentences:
        sentences = [compact]

    chunks: list[str] = []
    current_words: list[str] = []

    def flush_current() -> None:
        nonlocal current_words
        if not current_words:
            return
        chunks.append(" ".join(current_words).strip())
        if overlap_words > 0:
            current_words = current_words[-overlap_words:]
        else:
            current_words = []

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        if len(words) > max_words:
            flush_current()
            start = 0
            step = max(1, max_words - overlap_words)
            while start < len(words):
                piece = words[start : start + max_words]
                if piece:
                    chunks.append(" ".join(piece).strip())
                if start + max_words >= len(words):
                    break
                start += step
            current_words = []
            continue

        if current_words and len(current_words) + len(words) > max_words:
            flush_current()
        current_words.extend(words)

    flush_current()

    deduped: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        normalized = chunk.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def build_chunk_documents(places: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return build_chunk_documents_with_config(places)


def build_chunk_documents_with_config(
    places: Iterable[dict[str, Any]],
    *,
    max_words: int | None = None,
    overlap_words: int | None = None,
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for place in places:
        metadata = {
            "place_id": place.get("place_id"),
            "city": place.get("city"),
            "city_key": place.get("city_key"),
            "category": place.get("category"),
            "district": place.get("district"),
            "ward": place.get("ward") or "",
            "primary_area_key": place.get("primary_area_key"),
            "admin_area_keys": place.get("admin_area_keys") or [],
            "intent_tags": place.get("intent_tags") or [],
            "planner_role": place.get("planner_role"),
            "density_bucket": place.get("density_bucket"),
            "verification_status": place.get("verification_status"),
            "destination_type": place.get("destination_type") or "",
            "source": place.get("source"),
            "source_category_code": place.get("source_category_code") or "",
            "item_id": place.get("item_id") or "",
            "detail_url": place.get("detail_url") or "",
            "map_provider": place.get("map_provider") or "",
            "map_place_id": place.get("map_place_id") or "",
            "map_business_status": place.get("map_business_status") or "",
            "google_place_id": place.get("google_place_id") or "",
            "google_business_status": place.get("google_business_status") or "",
            "coordinate_source": place.get("coordinate_source") or "",
            "coordinate_confidence": place.get("coordinate_confidence") or "",
            "has_description": bool(str(place.get("description") or "").strip()),
            "has_website": bool(str(place.get("website") or "").strip()),
            "has_phone": bool(str(place.get("phone") or "").strip()),
        }
        for index, chunk in enumerate(
            chunk_text(
                place_document_text(place),
                max_words=max_words,
                overlap_words=overlap_words,
            )
        ):
            documents.append(
                {
                    "doc_id": f"{place.get('place_id')}_{index}",
                    "place_id": place.get("place_id"),
                    "title": str(place.get("name") or "").strip(),
                    "city": str(place.get("city") or "").strip(),
                    "category": str(place.get("category") or "").strip(),
                    "chunk_index": index,
                    "document_text": chunk,
                    "metadata": metadata,
                }
            )
    return documents


def attach_embeddings(documents: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    texts = [str(doc.get("document_text") or "").strip() for doc in documents]
    if not texts:
        return documents, None
    vectors, model_name = embed_texts(texts, input_type="passage")
    if not vectors or len(vectors) != len(documents):
        return documents, model_name
    enriched: list[dict[str, Any]] = []
    for doc, vector in zip(documents, vectors):
        item = dict(doc)
        item["embedding"] = vector
        if model_name:
            item["embedding_model"] = model_name
        enriched.append(item)
    return enriched, model_name


def embed_texts(
    texts: list[str],
    *,
    input_type: str = "passage",
) -> tuple[list[list[float]], str | None]:
    settings = get_settings()
    provider = str(settings.embedding_provider or "sentence_transformers").strip().lower()
    model_name = str(settings.embedding_model or _DEFAULT_EMBEDDING_MODEL).strip() or _DEFAULT_EMBEDDING_MODEL
    if not texts:
        return [], model_name

    if provider in {"sentence_transformers", "sentence-transformers", "local"}:
        return _embed_with_sentence_transformers(texts=texts, model_name=model_name, input_type=input_type)
    if provider in {"openai"}:
        return _embed_with_openai(texts=texts, model_name=model_name)
    return [], model_name


_OPENAI_EMBED_BATCH_SIZE = 512


def _embed_with_openai(
    texts: list[str],
    *,
    model_name: str,
) -> tuple[list[list[float]], str | None]:
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return [], model_name

    settings = get_settings()
    api_key = (settings.openrouter_api_key or "").strip()
    base_url = (settings.openrouter_base_url or "https://api.openai.com/v1").strip()
    if not api_key:
        return [], model_name

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60)
    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), _OPENAI_EMBED_BATCH_SIZE):
        batch = texts[i : i + _OPENAI_EMBED_BATCH_SIZE]
        try:
            response = client.embeddings.create(model=model_name, input=batch)
            batch_vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            all_vectors.extend(batch_vectors)
        except Exception:
            return [], model_name
    return all_vectors, model_name


def _embed_with_sentence_transformers(
    texts: list[str],
    *,
    model_name: str,
    input_type: str,
) -> tuple[list[list[float]], str | None]:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return [], model_name

    prepared = [_prepare_embedding_text(text, input_type=input_type, model_name=model_name) for text in texts]
    model = _get_sentence_transformer(model_name)
    if model is None:
        return [], model_name

    batch_size = max(1, int(get_settings().embedding_batch_size or 32))
    try:
        vectors = model.encode(
            prepared,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    except Exception:
        return [], model_name
    return [vector.tolist() for vector in vectors], model_name


@lru_cache(maxsize=2)
def _get_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None
    try:
        return SentenceTransformer(model_name)
    except Exception:
        return None


def _prepare_embedding_text(text: str, *, input_type: str, model_name: str) -> str:
    normalized = " ".join(str(text or "").split()).strip()
    if not normalized:
        return normalized
    lowered_model = model_name.lower()
    if "e5" in lowered_model:
        prefix = "query: " if input_type == "query" else "passage: "
        if normalized.lower().startswith(("query: ", "passage: ")):
            return normalized
        return prefix + normalized
    return normalized


def retrieve_place_candidates(
    query: str,
    *,
    source_kind: str,
    top_k: int = 8,
    docs: list[dict[str, Any]] | None = None,
    place_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    hits = retrieve_chunk_hits(
        query=query,
        source_kind=source_kind,
        top_k=max(top_k * 5, 18),
        docs=docs,
    )
    if not hits:
        return []

    per_place: dict[str, dict[str, Any]] = {}
    for hit in hits:
        place_id = hit.place_id.strip()
        if not place_id:
            continue
        entry = per_place.setdefault(
            place_id,
            {
                "best_score": float("-inf"),
                "score_sum": 0.0,
                "count": 0,
                "snippets": [],
            },
        )
        entry["best_score"] = max(float(entry["best_score"]), float(hit.score))
        entry["score_sum"] += float(hit.score)
        entry["count"] += 1
        snippets = entry["snippets"]
        if len(snippets) < 3 and hit.document_text not in snippets:
            snippets.append(hit.document_text)

    ranked_ids = sorted(
        per_place.items(),
        key=lambda item: (
            float(item[1]["best_score"]),
            float(item[1]["score_sum"]) / max(1, int(item[1]["count"])),
        ),
        reverse=True,
    )[:top_k]
    if not ranked_ids:
        return []

    place_lookup = place_lookup or _load_unified_places_by_id()
    raw_scores = [float(payload["best_score"]) for _, payload in ranked_ids]
    max_score = max(raw_scores) if raw_scores else 0.0
    min_score = min(raw_scores) if raw_scores else 0.0

    rows: list[dict[str, Any]] = []
    for place_id, payload in ranked_ids:
        base_place = place_lookup.get(place_id)
        if not base_place:
            continue
        row = dict(base_place)
        best_score = float(payload["best_score"])
        if max_score > min_score:
            normalized = (best_score - min_score) / (max_score - min_score)
        else:
            normalized = 1.0 if best_score > 0 else 0.0
        row["retrieval_relevance"] = round(max(0.0, min(1.0, normalized)), 4)
        row["vector_score"] = round(best_score, 4)
        row["rag_snippets"] = list(payload["snippets"])[:2]
        row["retrieval_tier"] = "vector_rag"
        rows.append(row)
    return rows


def retrieve_chunk_hits(
    query: str,
    *,
    source_kind: str | None = None,
    top_k: int = 8,
    docs: list[dict[str, Any]] | None = None,
) -> list[ChunkHit]:
    if not str(query or "").strip():
        return []

    query_vector = _embed_query(query)
    category = _source_kind_to_category(source_kind)

    # Use Chroma if available and no in-memory docs override provided.
    if docs is None and query_vector:
        chroma_hits = _retrieve_from_chroma(query_vector, category=category, top_k=top_k)
        if chroma_hits:
            return chroma_hits

    # Fallback: in-memory search over provided docs or SQLite chunks.
    active_docs = _filter_documents(docs or _load_rag_documents(), source_kind=source_kind)
    if not active_docs:
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    for doc in active_docs:
        score = _document_score(query=query, query_vector=query_vector, doc=doc)
        if score <= 0:
            continue
        scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)

    hits: list[ChunkHit] = []
    for score, doc in scored[:top_k]:
        hits.append(
            ChunkHit(
                doc_id=str(doc.get("doc_id") or ""),
                place_id=str(doc.get("place_id") or ""),
                title=str(doc.get("title") or ""),
                category=str(doc.get("category") or ""),
                city=str(doc.get("city") or ""),
                document_text=str(doc.get("document_text") or ""),
                metadata=dict(doc.get("metadata") or {}),
                score=float(score),
            )
        )
    return hits


def _retrieve_from_chroma(
    query_vector: list[float],
    *,
    category: str | None,
    top_k: int,
) -> list[ChunkHit]:
    try:
        from app.places.chroma import search_chunks
    except Exception:
        return []
    try:
        raw_hits = search_chunks(query_vector, category=category, top_k=top_k)
    except Exception:
        return []
    hits: list[ChunkHit] = []
    for h in raw_hits:
        hits.append(
            ChunkHit(
                doc_id=str(h.get("doc_id") or ""),
                place_id=str(h.get("place_id") or ""),
                title=str(h.get("title") or ""),
                category=str(h.get("category") or ""),
                city=str(h.get("city") or ""),
                document_text=str(h.get("document_text") or ""),
                metadata=dict(h.get("metadata") or {}),
                score=float(h.get("score") or 0.0),
            )
        )
    return hits


def _source_kind_to_category(source_kind: str | None) -> str | None:
    mapping = {
        "destinations": "destination",
        "entertainment": "entertainment",
        "restaurants": "restaurant",
        "accommodations": "accommodation",
    }
    return mapping.get(str(source_kind or "").strip().lower())


def format_chunk_context(chunks: list[ChunkHit], *, limit: int = 6) -> str:
    if not chunks:
        return ""
    lines = ["## Vector RAG snippets"]
    for hit in chunks[:limit]:
        lines.extend(
            [
                f"- {hit.title} ({hit.category}, {hit.city})",
                f"  snippet: {_truncate_text(hit.document_text, limit=320)}",
                f"  similarity_score: {hit.score:.3f}",
            ]
        )
    return "\n".join(lines).strip()


def _document_score(query: str, query_vector: list[float] | None, doc: dict[str, Any]) -> float:
    doc_vector = doc.get("embedding")
    if query_vector and isinstance(doc_vector, list) and doc_vector:
        return max(0.0, _cosine_similarity(query_vector, [float(v) for v in doc_vector]))
    return _lexical_score(query, str(doc.get("document_text") or ""))


@lru_cache(maxsize=128)
def _embed_query(query: str) -> list[float] | None:
    vectors, _ = embed_texts([query], input_type="query")
    if not vectors:
        return None
    return vectors[0]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a <= 0 or mag_b <= 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _lexical_score(query: str, document_text: str) -> float:
    qf = fold_text(query)
    df = fold_text(document_text)
    if not qf or not df:
        return 0.0
    terms = [term for term in qf.split() if len(term) >= 2]
    if not terms:
        return 0.0
    hits = sum(1 for term in terms if term in df)
    if hits <= 0:
        return 0.0
    phrase_bonus = 0.15 if qf in df else 0.0
    return min(1.0, hits / max(len(terms), 1) + phrase_bonus)


def _filter_documents(
    docs: list[dict[str, Any]],
    *,
    source_kind: str | None,
) -> list[dict[str, Any]]:
    wanted_categories = _SOURCE_KIND_CATEGORY_MAP.get(str(source_kind or "").strip().lower())
    if not wanted_categories:
        return docs
    return [
        doc for doc in docs
        if str(doc.get("category") or "").strip().lower() in wanted_categories
    ]


@lru_cache(maxsize=1)
def _load_rag_documents() -> list[dict[str, Any]]:
    db_docs = list_place_chunks()
    if db_docs:
        return [dict(item) for item in db_docs if isinstance(item, dict)]
    if not RAG_JSON.exists():
        return []
    try:
        payload = json.loads(RAG_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


@lru_cache(maxsize=1)
def _load_unified_places_by_id() -> dict[str, dict[str, Any]]:
    db_lookup = load_places_lookup()
    if db_lookup:
        return {key: dict(value) for key, value in db_lookup.items()}
    out: dict[str, dict[str, Any]] = {}
    payload: list[dict[str, Any]] = []
    if UNIFIED_JSON.exists():
        try:
            raw = json.loads(UNIFIED_JSON.read_text(encoding="utf-8"))
        except Exception:
            raw = []
        if isinstance(raw, list):
            payload.extend([item for item in raw if isinstance(item, dict)])
    for item in payload:
        if not isinstance(item, dict):
            continue
        enriched = enrich_place_record(item)
        place_id = str(enriched.get("place_id") or "").strip()
        if not place_id:
            continue
        existing = out.get(place_id)
        if existing is None:
            out[place_id] = enriched
            continue
        if _record_richness(enriched) > _record_richness(existing):
            out[place_id] = enriched
    return out


def _record_richness(place: dict[str, Any]) -> int:
    score = 0
    for field in ("description", "detail_content", "list_snippet", "address", "lat", "lon"):
        if place.get(field) not in (None, "", [], {}):
            score += 1
    return score


def _truncate_text(text: str, limit: int) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _category_label(category: str) -> str:
    normalized = str(category or "").strip().lower()
    labels = {
        "destination": "diem den",
        "entertainment": "giai tri",
        "restaurant": "am thuc",
        "accommodation": "luu tru",
    }
    return labels.get(normalized, normalized or "dia diem")


def _planner_role_label(role: str) -> str:
    normalized = str(role or "").strip().lower()
    labels = {
        "tourism": "tham quan",
        "entertainment": "giai tri",
        "restaurant": "an uong",
        "stay": "luu tru",
    }
    return labels.get(normalized, normalized or "khong ro")
