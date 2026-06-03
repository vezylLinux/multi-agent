from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.settings import get_settings
from app.core.database import init_db
from app.places.metadata import enrich_place_record
from app.places.repository import list_places, replace_place_chunks
from app.places.vector_rag import attach_embeddings, build_chunk_documents

UNIFIED_JSON = ROOT / "data" / "processed" / "unified_places.json"
RAG_DIR = ROOT / "data" / "rag"
RAG_JSON = RAG_DIR / "rag_documents.json"
RAG_JSONL = RAG_DIR / "rag_documents.jsonl"


def main() -> None:
    RAG_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    settings = get_settings()
    places = _load_places()
    documents = build_chunk_documents(places)
    documents, embedding_model = attach_embeddings(documents)
    RAG_JSON.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
    RAG_JSONL.write_text(
        "\n".join(json.dumps(doc, ensure_ascii=False) for doc in documents),
        encoding="utf-8",
    )
    replace_place_chunks(documents)
    with_embeddings = sum(
        1 for doc in documents if isinstance(doc.get("embedding"), list) and doc.get("embedding")
    )
    print(f"Wrote {len(documents)} chunked RAG documents to {RAG_JSON}")
    print(f"Wrote {len(documents)} chunked RAG documents to {RAG_JSONL}")
    print(f"Synced {len(documents)} chunked RAG documents into SQLite")
    print("Run 'python scripts/ingest_to_chroma.py' to sync chunks into Chroma.")
    if with_embeddings:
        print(f"Embedded {with_embeddings}/{len(documents)} chunks with model: {embedding_model}")
    else:
        provider = str(settings.embedding_provider or "sentence_transformers").strip() or "sentence_transformers"
        print(
            "No embeddings attached. "
            f"Current embedding provider: {provider}, model: {embedding_model}."
        )


def _load_places() -> list[dict[str, Any]]:
    pg_places = list_places()
    if pg_places:
        return [dict(item) for item in pg_places if isinstance(item, dict)]
    if not UNIFIED_JSON.exists():
        raise FileNotFoundError(
            f"Missing unified catalog at {UNIFIED_JSON}. Run scripts/preprocess.py first."
        )
    payload = json.loads(UNIFIED_JSON.read_text(encoding="utf-8"))
    places = [enrich_place_record(item) for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []
    seen = {str(place.get("place_id") or "").strip() for place in places if str(place.get("place_id") or "").strip()}
    return places


if __name__ == "__main__":
    main()
