from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.settings import get_settings

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ChromaStatus:
    ok: bool
    message: str = ""


@lru_cache(maxsize=1)
def get_chroma_client():
    import chromadb

    settings = get_settings()
    if settings.chroma_mode == "http":
        return chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
    persist_path = str(ROOT / settings.chroma_persist_dir)
    return chromadb.PersistentClient(path=persist_path)


def get_collection():
    settings = get_settings()
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def chroma_status() -> ChromaStatus:
    try:
        col = get_collection()
        count = col.count()
        return ChromaStatus(ok=True, message=f"Chroma ready — {count} chunks indexed.")
    except Exception as exc:
        return ChromaStatus(ok=False, message=str(exc))


def upsert_chunks(documents: list[dict[str, Any]]) -> int:
    if not documents:
        return 0

    ids: list[str] = []
    embeddings: list[list[float]] = []
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for doc in documents:
        doc_id = str(doc.get("doc_id") or "").strip()
        embedding = doc.get("embedding")
        text = str(doc.get("document_text") or "").strip()
        if not doc_id or not text:
            continue
        if not isinstance(embedding, list) or not embedding:
            continue

        meta: dict[str, Any] = {}
        raw_meta = doc.get("metadata") or {}
        for key, value in raw_meta.items():
            if isinstance(value, (str, int, float, bool)):
                meta[key] = value
            elif isinstance(value, list):
                meta[key] = ",".join(str(v) for v in value)
            else:
                meta[key] = str(value) if value is not None else ""
        meta["place_id"] = str(doc.get("place_id") or "")
        meta["title"] = str(doc.get("title") or "")
        meta["city"] = str(doc.get("city") or "")
        meta["category"] = str(doc.get("category") or "")
        meta["chunk_index"] = int(doc.get("chunk_index") or 0)

        ids.append(doc_id)
        embeddings.append([float(v) for v in embedding])
        texts.append(text)
        metadatas.append(meta)

    if not ids:
        return 0

    collection = get_collection()
    batch_size = 512
    inserted = 0
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
        inserted += len(ids[start:end])
    return inserted


def search_chunks(
    query_embedding: list[float],
    *,
    category: str | None = None,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    if not query_embedding:
        return []

    collection = get_collection()
    where: dict[str, Any] | None = None
    if category:
        where = {"category": {"$eq": category}}

    kwargs: dict[str, Any] = {
        "query_embeddings": [[float(v) for v in query_embedding]],
        "n_results": min(top_k, max(1, collection.count())),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    try:
        results = collection.query(**kwargs)
    except Exception:
        return []

    hits: list[dict[str, Any]] = []
    ids = (results.get("ids") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]

    for doc_id, distance, text, meta in zip(ids, distances, documents, metadatas):
        score = max(0.0, 1.0 - float(distance))
        hits.append(
            {
                "doc_id": doc_id,
                "document_text": text,
                "score": round(score, 4),
                "place_id": str((meta or {}).get("place_id") or ""),
                "title": str((meta or {}).get("title") or ""),
                "category": str((meta or {}).get("category") or ""),
                "city": str((meta or {}).get("city") or ""),
                "metadata": dict(meta or {}),
            }
        )
    return hits


def delete_collection() -> None:
    settings = get_settings()
    client = get_chroma_client()
    try:
        client.delete_collection(settings.chroma_collection_name)
    except Exception:
        pass
    get_collection.cache_clear() if hasattr(get_collection, "cache_clear") else None
