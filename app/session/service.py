from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from fastapi import Request, Response

from app.core.settings import get_settings
from app.core.database import get_cursor
from app.core.security import generate_session_key, hash_ip, set_session_cookie


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class PrincipalContext:
    principal_id: str
    principal_type: Literal["anonymous", "user"]
    session_id: str
    session_key: str
    first_seen_at: str
    last_seen_at: str


class SessionService:
    def get_or_create_principal(self, request: Request, response: Response) -> PrincipalContext:
        session_key = request.cookies.get(get_settings().session_cookie_name)
        if session_key:
            existing = self._find_by_session_key(session_key)
            if existing:
                refreshed = self._touch_session(existing.session_id)
                return PrincipalContext(
                    principal_id=existing.principal_id,
                    principal_type=existing.principal_type,
                    session_id=existing.session_id,
                    session_key=existing.session_key,
                    first_seen_at=existing.first_seen_at,
                    last_seen_at=refreshed,
                )

        created = self._create_anonymous_session(request)
        set_session_cookie(response, created.session_key)
        return created

    def _find_by_session_key(self, session_key: str) -> PrincipalContext | None:
        with get_cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    s.id AS session_id,
                    s.principal_id AS principal_id,
                    s.session_key AS session_key,
                    s.first_seen_at AS first_seen_at,
                    s.last_seen_at AS last_seen_at,
                    p.type AS principal_type
                FROM anonymous_sessions s
                JOIN principals p ON p.id = s.principal_id
                WHERE s.session_key = ?
                """,
                (session_key,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return PrincipalContext(
            principal_id=row["principal_id"],
            principal_type=row["principal_type"],
            session_id=row["session_id"],
            session_key=row["session_key"],
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
        )

    def _touch_session(self, session_id: str) -> str:
        now = _utc_now()
        with get_cursor(commit=True) as cursor:
            cursor.execute(
                "UPDATE anonymous_sessions SET last_seen_at = ? WHERE id = ?",
                (now, session_id),
            )
        return now

    def _create_anonymous_session(self, request: Request) -> PrincipalContext:
        principal_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        session_key = generate_session_key()
        now = _utc_now()
        user_agent = request.headers.get("user-agent")
        client_host = request.client.host if request.client else None
        ip_digest = hash_ip(client_host)
        with get_cursor(commit=True) as cursor:
            cursor.execute(
                """
                INSERT INTO principals (id, type, created_at, updated_at)
                VALUES (?, 'anonymous', ?, ?)
                """,
                (principal_id, now, now),
            )
            cursor.execute(
                """
                INSERT INTO anonymous_sessions (
                    id,
                    principal_id,
                    session_key,
                    first_seen_at,
                    last_seen_at,
                    user_agent,
                    ip_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, principal_id, session_key, now, now, user_agent, ip_digest),
            )
        return PrincipalContext(
            principal_id=principal_id,
            principal_type="anonymous",
            session_id=session_id,
            session_key=session_key,
            first_seen_at=now,
            last_seen_at=now,
        )
