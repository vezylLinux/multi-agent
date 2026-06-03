from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, status

from app.core.database import get_cursor
from app.session.schemas import PlanPayload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PlanService:
    def save_plan(
        self,
        *,
        principal_id: str,
        conversation_id: str,
        city: str | None,
        days: int | None,
        structured_json: dict[str, Any],
    ) -> PlanPayload:
        self._ensure_conversation_owner(principal_id, conversation_id)
        now = _utc_now()
        plan_id = str(uuid.uuid4())
        with get_cursor(commit=True) as cursor:
            cursor.execute(
                """
                INSERT INTO plans (
                    id,
                    conversation_id,
                    principal_id,
                    city,
                    days,
                    structured_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan_id,
                    conversation_id,
                    principal_id,
                    city,
                    days,
                    json.dumps(structured_json, ensure_ascii=True),
                    now,
                    now,
                ),
            )
        return PlanPayload(
            id=plan_id,
            conversation_id=conversation_id,
            principal_id=principal_id,
            city=city,
            days=days,
            structured_json=structured_json,
            created_at=datetime.fromisoformat(now),
            updated_at=datetime.fromisoformat(now),
        )

    def get_plan(self, *, principal_id: str, plan_id: str) -> PlanPayload:
        with get_cursor() as cursor:
            cursor.execute(
                """
                SELECT id, conversation_id, principal_id, city, days, structured_json, created_at, updated_at
                FROM plans
                WHERE id = ? AND principal_id = ?
                """,
                (plan_id, principal_id),
            )
            row = cursor.fetchone()
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Plan not found for current principal.",
            )
        return PlanPayload(
            id=row["id"],
            conversation_id=row["conversation_id"],
            principal_id=row["principal_id"],
            city=row["city"],
            days=row["days"],
            structured_json=json.loads(row["structured_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _ensure_conversation_owner(self, principal_id: str, conversation_id: str) -> None:
        with get_cursor() as cursor:
            cursor.execute(
                "SELECT id FROM conversations WHERE id = ? AND principal_id = ?",
                (conversation_id, principal_id),
            )
            row = cursor.fetchone()
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found for current principal.",
            )
