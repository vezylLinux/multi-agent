from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Response, status

from app.core.dependencies import get_current_principal
from app.session.schemas import (
    ConversationDetail,
    ConversationSummary,
    PlanPayload,
    PlanSaveRequest,
    PrincipalPayload,
    SessionInfoResponse,
)
from app.session.conversations import ConversationService
from app.session.plans import PlanService
from app.session.service import PrincipalContext

session_router = APIRouter()
conversations_router = APIRouter()
plan_router = APIRouter()


def get_conversation_service() -> ConversationService:
    return ConversationService()


def get_plan_service() -> PlanService:
    return PlanService()


def _to_session_response(principal: PrincipalContext) -> SessionInfoResponse:
    return SessionInfoResponse(
        principal=PrincipalPayload(
            id=principal.principal_id,
            type=principal.principal_type,
        ),
        session_started_at=datetime.fromisoformat(principal.first_seen_at),
        last_seen_at=datetime.fromisoformat(principal.last_seen_at),
    )


@session_router.post("/init", response_model=SessionInfoResponse)
def init_session(
    principal: PrincipalContext = Depends(get_current_principal),
) -> SessionInfoResponse:
    return _to_session_response(principal)


@session_router.get("/me", response_model=SessionInfoResponse)
def read_session(
    principal: PrincipalContext = Depends(get_current_principal),
) -> SessionInfoResponse:
    return _to_session_response(principal)


@conversations_router.get("", response_model=list[ConversationSummary])
def list_conversations(
    principal: PrincipalContext = Depends(get_current_principal),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> list[ConversationSummary]:
    return conversation_service.list_conversations(principal.principal_id)


@conversations_router.get("/{conversation_id}", response_model=ConversationDetail)
def read_conversation(
    conversation_id: str,
    principal: PrincipalContext = Depends(get_current_principal),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationDetail:
    return conversation_service.get_conversation_detail(principal.principal_id, conversation_id)


@conversations_router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: str,
    principal: PrincipalContext = Depends(get_current_principal),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> Response:
    conversation_service.delete_conversation(principal.principal_id, conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@conversations_router.delete("", response_model=dict[str, int])
def delete_all_conversations(
    principal: PrincipalContext = Depends(get_current_principal),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> dict[str, int]:
    deleted_count = conversation_service.delete_all_conversations(principal.principal_id)
    return {"deleted_conversations": deleted_count}


@plan_router.post("/save", response_model=PlanPayload)
def save_plan(
    request: PlanSaveRequest,
    principal: PrincipalContext = Depends(get_current_principal),
    plan_service: PlanService = Depends(get_plan_service),
) -> PlanPayload:
    return plan_service.save_plan(
        principal_id=principal.principal_id,
        conversation_id=request.conversation_id,
        city=request.city,
        days=request.days,
        structured_json=request.structured_json,
    )


@plan_router.get("/{plan_id}", response_model=PlanPayload)
def read_plan(
    plan_id: str,
    principal: PrincipalContext = Depends(get_current_principal),
    plan_service: PlanService = Depends(get_plan_service),
) -> PlanPayload:
    return plan_service.get_plan(principal_id=principal.principal_id, plan_id=plan_id)
