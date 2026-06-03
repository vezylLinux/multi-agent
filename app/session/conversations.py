from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, status

from app.core.database import get_cursor
from app.session.schemas import ConversationDetail, ConversationSummary, MessagePayload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_title_from_message(message: str) -> str:
    text = " ".join((message or "").strip().split())
    if not text:
        return "New conversation"
    return text[:77] + "..." if len(text) > 80 else text


class ConversationService:
    def get_or_create_conversation(
        self,
        *,
        principal_id: str,
        conversation_id: str | None,
        initial_message: str,
    ) -> ConversationSummary:
        if conversation_id:
            existing = self.get_conversation_summary(principal_id, conversation_id)
            if not existing:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found for current principal.",
                )
            return existing

        now = _utc_now()
        created_id = str(uuid.uuid4())
        title = _build_title_from_message(initial_message)
        with get_cursor(commit=True) as cursor:
            cursor.execute(
                """
                INSERT INTO conversations (id, principal_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (created_id, principal_id, title, now, now),
            )
        return self._fetch_conversation_summary(principal_id, created_id)

    def append_message(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> MessagePayload:
        created_at = _utc_now()
        message_id = str(uuid.uuid4())
        metadata_json = json.dumps(metadata, ensure_ascii=True) if metadata is not None else None
        with get_cursor(commit=True) as cursor:
            cursor.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (message_id, conversation_id, role, content, metadata_json, created_at),
            )
            cursor.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (created_at, conversation_id),
            )
        return MessagePayload(
            id=message_id,
            role=role,
            content=content,
            metadata=metadata,
            created_at=datetime.fromisoformat(created_at),
        )

    def list_conversations(self, principal_id: str) -> list[ConversationSummary]:
        with get_cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    c.id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    (
                        SELECT m.content
                        FROM messages m
                        WHERE m.conversation_id = c.id
                        ORDER BY m.created_at DESC
                        LIMIT 1
                    ) AS latest_message_preview,
                    (
                        SELECT COUNT(*)
                        FROM messages m
                        WHERE m.conversation_id = c.id
                    ) AS message_count
                FROM conversations c
                WHERE c.principal_id = ?
                ORDER BY c.updated_at DESC
                """,
                (principal_id,),
            )
            rows = cursor.fetchall()
        return [self._row_to_conversation_summary(row) for row in rows]

    def get_conversation_summary(self, principal_id: str, conversation_id: str) -> ConversationSummary | None:
        return self._fetch_conversation_summary(principal_id, conversation_id)

    def get_conversation_detail(self, principal_id: str, conversation_id: str) -> ConversationDetail:
        summary = self._fetch_conversation_summary(principal_id, conversation_id)
        if not summary:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found for current principal.",
            )
        with get_cursor() as cursor:
            cursor.execute(
                """
                SELECT id, role, content, metadata_json, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            )
            rows = cursor.fetchall()
        messages = [self._row_to_message(row) for row in rows]
        return ConversationDetail(
            **summary.model_dump(),
            principal_id=principal_id,
            messages=messages,
        )

    def delete_conversation(self, principal_id: str, conversation_id: str) -> bool:
        summary = self._fetch_conversation_summary(principal_id, conversation_id)
        if not summary:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found for current principal.",
            )
        with get_cursor(commit=True) as cursor:
            cursor.execute(
                "DELETE FROM plans WHERE conversation_id = ? AND principal_id = ?",
                (conversation_id, principal_id),
            )
            cursor.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            cursor.execute(
                "DELETE FROM conversations WHERE id = ? AND principal_id = ?",
                (conversation_id, principal_id),
            )
        return True

    def delete_all_conversations(self, principal_id: str) -> int:
        with get_cursor(commit=True) as cursor:
            cursor.execute(
                "SELECT id FROM conversations WHERE principal_id = ?",
                (principal_id,),
            )
            conversation_ids = [str(row["id"]) for row in cursor.fetchall()]
            if not conversation_ids:
                return 0
            cursor.execute(
                "DELETE FROM plans WHERE principal_id = ?",
                (principal_id,),
            )
            placeholders = ",".join(["?" for _ in conversation_ids])
            cursor.execute(
                f"DELETE FROM messages WHERE conversation_id IN ({placeholders})",
                conversation_ids,
            )
            cursor.execute(
                "DELETE FROM conversations WHERE principal_id = ?",
                (principal_id,),
            )
        return len(conversation_ids)

    def build_effective_user_message(
        self,
        *,
        principal_id: str,
        conversation_id: str | None,
        current_message: str,
        max_user_messages: int = 8,
    ) -> str:
        current_text = " ".join((current_message or "").strip().split())
        if not conversation_id:
            return current_text

        detail = self.get_conversation_detail(principal_id, conversation_id)
        prior_user_messages = [
            " ".join((message.content or "").strip().split())
            for message in detail.messages
            if message.role == "user" and str(message.content or "").strip()
        ]
        if max_user_messages > 0:
            prior_user_messages = prior_user_messages[-max_user_messages:]

        merged: list[str] = []
        for item in [*prior_user_messages, current_text]:
            if not item:
                continue
            if merged and merged[-1] == item:
                continue
            merged.append(item)
        return "\n".join(merged)

    def _fetch_conversation_summary(self, principal_id: str, conversation_id: str) -> ConversationSummary | None:
        with get_cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    c.id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    (
                        SELECT m.content
                        FROM messages m
                        WHERE m.conversation_id = c.id
                        ORDER BY m.created_at DESC
                        LIMIT 1
                    ) AS latest_message_preview,
                    (
                        SELECT COUNT(*)
                        FROM messages m
                        WHERE m.conversation_id = c.id
                    ) AS message_count
                FROM conversations c
                WHERE c.id = ? AND c.principal_id = ?
                """,
                (conversation_id, principal_id),
            )
            row = cursor.fetchone()
        return self._row_to_conversation_summary(row) if row else None

    def _row_to_conversation_summary(self, row: Any) -> ConversationSummary:
        preview = row["latest_message_preview"]
        if preview and len(preview) > 140:
            preview = preview[:137] + "..."
        return ConversationSummary(
            id=row["id"],
            title=row["title"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            latest_message_preview=preview,
            message_count=int(row["message_count"] or 0),
        )

    def _row_to_message(self, row: Any) -> MessagePayload:
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else None
        return MessagePayload(
            id=row["id"],
            role=row["role"],
            content=row["content"],
            metadata=metadata,
            created_at=datetime.fromisoformat(row["created_at"]),
        )
