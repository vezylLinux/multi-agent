"""
Sync processed place chunks from SQLite into the Chroma vector collection.

Usage:
    python scripts/ingest_to_chroma.py [--recreate]

Options:
    --recreate   Drop and recreate the Chroma collection before ingesting.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.places.chroma import delete_collection, get_collection, upsert_chunks
from app.places.repository import list_place_chunks, list_places
from app.places.vector_rag import attach_embeddings, build_chunk_documents


def main(*, recreate: bool = False) -> None:
    if recreate:
        print("Dropping existing Chroma collection...")
        delete_collection()

    print("Loading place chunks from SQLite...")
    db_chunks = list_place_chunks()

    if db_chunks:
        documents = [dict(item) for item in db_chunks if isinstance(item, dict)]
        # Chunks from DB may already have embeddings stored as lists in embedding_json.
        # Re-embed any that are missing embeddings.
        missing_embed = [d for d in documents if not isinstance(d.get("embedding"), list)]
        if missing_embed:
            print(f"Re-embedding {len(missing_embed)} chunks without embeddings...")
            enriched, model = attach_embeddings(missing_embed)
            embed_map = {d["doc_id"]: d for d in enriched if d.get("embedding")}
            for i, doc in enumerate(documents):
                if doc.get("doc_id") in embed_map:
                    documents[i] = embed_map[doc["doc_id"]]
    else:
        print("No chunks in SQLite. Building from places...")
        places = list_places()
        if not places:
            print("No places found in SQLite. Run scripts/preprocess.py first.")
            sys.exit(1)
        raw_docs = build_chunk_documents([dict(p) for p in places])
        print(f"Built {len(raw_docs)} chunks from {len(places)} places. Embedding...")
        documents, model = attach_embeddings(raw_docs)
        print(f"Embedding model: {model}")

    embeddable = [d for d in documents if isinstance(d.get("embedding"), list)]
    print(f"Upserting {len(embeddable)} chunks into Chroma...")
    count = upsert_chunks(embeddable)
    col = get_collection()
    print(f"Done. Chroma collection now has {col.count()} chunks (upserted {count}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collection")
    args = parser.parse_args()
    main(recreate=args.recreate)
