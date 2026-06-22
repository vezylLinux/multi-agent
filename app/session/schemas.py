from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class PrincipalPayload(BaseModel):
    id: str
    type: Literal["anonymous", "user"]
    display_name: str | None = None


class SessionInfoResponse(BaseModel):
    principal: PrincipalPayload
    session_started_at: datetime
    last_seen_at: datetime


class MessagePayload(BaseModel):
    id: str
    role: str
    content: str
    metadata: dict[str, Any] | None = None
    created_at: datetime


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    latest_message_preview: str | None = None
    message_count: int = 0


class ConversationDetail(ConversationSummary):
    principal_id: str
    messages: list[MessagePayload] = Field(default_factory=list)


class PlanSaveRequest(BaseModel):
    conversation_id: str
    city: str | None = None
    days: int | None = Field(default=None, ge=1)
    structured_json: dict[str, Any]


class PlanPayload(BaseModel):
    id: str
    conversation_id: str
    principal_id: str
    city: str | None = None
    days: int | None = None
    structured_json: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")


class ChatSendRequest(ChatRequest):
    conversation_id: str | None = Field(
        default=None,
        description="Existing conversation id for follow-up messages.",
    )
    top_k: int = Field(default=5, ge=1, le=50)
    with_plan: bool = True
    category: str | None = None


class DebugStep(BaseModel):
    key: str
    title: str
    status: str
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str | None = None
    plan_id: str | None = None
    conversation_stage: str = "planning"
    collected_info: dict[str, Any] | None = None
    missing_fields: list[str] | None = None
    follow_up_questions: list[str] | None = None
    trace: list[str] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    plan: str | None = None
    stay_plan: dict[str, Any] | None = None
    stay_recommendations: list[dict[str, Any]] | None = None
    plan_validation: dict[str, Any] | None = None
    research: str | None = None
    coordinator_plan: str | None = None
    transport: list[str] | None = None
    recommended_hotel: dict[str, Any] | None = None
    mobility_plan: dict[str, Any] | None = None
    verified_places: list[dict[str, Any]] | None = None
    route_plan: list[dict[str, Any]] | None = None
    grounding: dict[str, Any] | None = None
    debug_steps: list[DebugStep] = Field(default_factory=list)
