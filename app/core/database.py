from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator

from app.core.settings import get_settings

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS principals (
        id TEXT PRIMARY KEY,
        type TEXT NOT NULL CHECK (type IN ('anonymous', 'user')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS anonymous_sessions (
        id TEXT PRIMARY KEY,
        principal_id TEXT NOT NULL,
        session_key TEXT NOT NULL UNIQUE,
        first_seen_at TEXT NOT NULL,
        last_seen_at TEXT NOT NULL,
        user_agent TEXT,
        ip_hash TEXT,
        FOREIGN KEY (principal_id) REFERENCES principals(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        principal_id TEXT NOT NULL,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (principal_id) REFERENCES principals(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        metadata_json TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS plans (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        principal_id TEXT NOT NULL,
        city TEXT,
        days INTEGER,
        structured_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id),
        FOREIGN KEY (principal_id) REFERENCES principals(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS places (
        place_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        city TEXT,
        city_key TEXT,
        district TEXT,
        ward TEXT,
        address TEXT,
        description TEXT,
        detail_content TEXT,
        list_snippet TEXT,
        source TEXT,
        source_category_code TEXT,
        destination_type TEXT,
        item_id TEXT,
        detail_url TEXT,
        website TEXT,
        phone TEXT,
        planner_role TEXT,
        primary_area_key TEXT,
        admin_area_keys_json TEXT,
        intent_tags_json TEXT,
        density_bucket TEXT,
        verification_status TEXT,
        lat REAL,
        lon REAL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS place_chunks (
        doc_id TEXT PRIMARY KEY,
        place_id TEXT NOT NULL,
        title TEXT,
        city TEXT,
        category TEXT,
        chunk_index INTEGER NOT NULL,
        document_text TEXT NOT NULL,
        metadata_json TEXT,
        embedding_json TEXT,
        embedding_model TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (place_id) REFERENCES places(place_id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_anonymous_sessions_principal_id ON anonymous_sessions(principal_id)",
    "CREATE INDEX IF NOT EXISTS idx_conversations_principal_id ON conversations(principal_id)",
    "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)",
    "CREATE INDEX IF NOT EXISTS idx_plans_conversation_id ON plans(conversation_id)",
    "CREATE INDEX IF NOT EXISTS idx_plans_principal_id ON plans(principal_id)",
    "CREATE INDEX IF NOT EXISTS idx_places_category ON places(category)",
    "CREATE INDEX IF NOT EXISTS idx_places_city_key ON places(city_key)",
    "CREATE INDEX IF NOT EXISTS idx_places_primary_area_key ON places(primary_area_key)",
    "CREATE INDEX IF NOT EXISTS idx_places_source ON places(source)",
    "CREATE INDEX IF NOT EXISTS idx_place_chunks_place_id ON place_chunks(place_id)",
    "CREATE INDEX IF NOT EXISTS idx_place_chunks_category ON place_chunks(category)",
)


def _db_path_from_url(url: str) -> str:
    if url.startswith("sqlite:///"):
        return url[len("sqlite:///"):]
    if url.startswith("sqlite://"):
        return url[len("sqlite://"):]
    return url


def _get_db_path() -> str:
    return _db_path_from_url(get_settings().database_url)


def _dict_row_factory(cursor: sqlite3.Cursor, row: tuple) -> dict:
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def get_connection() -> sqlite3.Connection:
    db_path = _get_db_path()
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = _dict_row_factory
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def get_cursor(*, commit: bool = False) -> Iterator[sqlite3.Cursor]:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        try:
            yield cursor
            if commit:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
    finally:
        conn.close()


def init_db() -> None:
    try:
        with get_cursor(commit=True) as cursor:
            for statement in _SCHEMA_STATEMENTS:
                cursor.execute(statement)
        print("SQLite database initialized successfully")
    except Exception as exc:
        print(f"Database initialization failed: {exc}")
        raise
