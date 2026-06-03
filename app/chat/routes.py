from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.core.dependencies import get_current_principal
from app.graph.build_graph import run_travel_graph, stream_travel_graph
from app.session.schemas import ChatRequest, ChatResponse, ChatSendRequest
from app.session.conversations import ConversationService
from app.session.plans import PlanService
from app.session.service import PrincipalContext

router = APIRouter()


def get_conversation_service() -> ConversationService:
    return ConversationService()


def get_plan_service() -> PlanService:
    return PlanService()


def _execute_chat(
    request: ChatSendRequest,
    principal: PrincipalContext,
    conversation_service: ConversationService,
    plan_service: PlanService,
) -> ChatResponse:
    request_started = perf_counter()

    step_started = perf_counter()
    conversation = conversation_service.get_or_create_conversation(
        principal_id=principal.principal_id,
        conversation_id=request.conversation_id,
        initial_message=request.message,
    )
    conversation_lookup_ms = round((perf_counter() - step_started) * 1000, 1)

    step_started = perf_counter()
    effective_message = conversation_service.build_effective_user_message(
        principal_id=principal.principal_id,
        conversation_id=conversation.id,
        current_message=request.message,
    )
    effective_message_ms = round((perf_counter() - step_started) * 1000, 1)

    step_started = perf_counter()
    response = run_travel_graph(
        effective_message,
        top_k=request.top_k,
        with_plan=request.with_plan,
        category=request.category,
    )
    graph_ms = round((perf_counter() - step_started) * 1000, 1)

    step_started = perf_counter()
    conversation_service.append_message(
        conversation_id=conversation.id,
        role="user",
        content=request.message,
        metadata={
            "top_k": request.top_k,
            "with_plan": request.with_plan,
            "category": request.category,
            "effective_message": effective_message,
        },
    )
    save_user_message_ms = round((perf_counter() - step_started) * 1000, 1)
    plan_payload = None
    save_plan_ms = 0.0
    if response.plan:
        collected_info = response.collected_info or {}
        raw_days = collected_info.get("days")
        if isinstance(raw_days, int):
            days = raw_days
        elif isinstance(raw_days, str) and raw_days.isdigit():
            days = int(raw_days)
        else:
            days = None
        step_started = perf_counter()
        plan_payload = plan_service.save_plan(
            principal_id=principal.principal_id,
            conversation_id=conversation.id,
            city=str(collected_info.get("destination") or "") or None,
            days=days,
            structured_json=response.model_dump(),
        )
        save_plan_ms = round((perf_counter() - step_started) * 1000, 1)
    response.conversation_id = conversation.id
    response.plan_id = plan_payload.id if plan_payload else None
    step_started = perf_counter()
    conversation_service.append_message(
        conversation_id=conversation.id,
        role="assistant",
        content=response.answer,
        metadata={
            **response.model_dump(),
            "effective_message": effective_message,
        },
    )
    save_assistant_message_ms = round((perf_counter() - step_started) * 1000, 1)
    total_request_ms = round((perf_counter() - request_started) * 1000, 1)
    response.grounding = {
        **(response.grounding or {}),
        "request_timing_ms": {
            "conversation_lookup_ms": conversation_lookup_ms,
            "effective_message_ms": effective_message_ms,
            "graph_ms": graph_ms,
            "save_user_message_ms": save_user_message_ms,
            "save_plan_ms": save_plan_ms,
            "save_assistant_message_ms": save_assistant_message_ms,
            "total_request_ms": total_request_ms,
        },
    }
    return response


@router.post("", response_model=ChatResponse, include_in_schema=False)
def chat_legacy(
    request: ChatRequest,
    principal: PrincipalContext = Depends(get_current_principal),
    conversation_service: ConversationService = Depends(get_conversation_service),
    plan_service: PlanService = Depends(get_plan_service),
) -> ChatResponse:
    return _execute_chat(
        ChatSendRequest(message=request.message),
        principal,
        conversation_service,
        plan_service,
    )


@router.post("/send", response_model=ChatResponse)
def chat_send(
    request: ChatSendRequest,
    principal: PrincipalContext = Depends(get_current_principal),
    conversation_service: ConversationService = Depends(get_conversation_service),
    plan_service: PlanService = Depends(get_plan_service),
) -> ChatResponse:
    return _execute_chat(request, principal, conversation_service, plan_service)


@router.post("/stream")
async def chat_stream(
    request: ChatSendRequest,
    principal: PrincipalContext = Depends(get_current_principal),
    conversation_service: ConversationService = Depends(get_conversation_service),
    plan_service: PlanService = Depends(get_plan_service),
) -> StreamingResponse:
    conversation = conversation_service.get_or_create_conversation(
        principal_id=principal.principal_id,
        conversation_id=request.conversation_id,
        initial_message=request.message,
    )
    effective_message = conversation_service.build_effective_user_message(
        principal_id=principal.principal_id,
        conversation_id=conversation.id,
        current_message=request.message,
    )

    async def _generate():
        final_response: ChatResponse | None = None
        import json

        async for event_str in stream_travel_graph(
            effective_message,
            top_k=request.top_k,
            with_plan=request.with_plan,
            category=request.category,
        ):
            yield event_str
            if event_str.startswith("data: "):
                try:
                    event = json.loads(event_str[6:])
                    if event.get("type") == "done":
                        payload = event.get("response") or {}
                        final_response = ChatResponse(**payload)
                except Exception:
                    pass

        if final_response is None:
            return

        final_response.conversation_id = conversation.id
        conversation_service.append_message(
            conversation_id=conversation.id,
            role="user",
            content=request.message,
            metadata={},
        )

        if final_response.plan:
            collected_info = final_response.collected_info or {}
            raw_days = collected_info.get("days")
            if isinstance(raw_days, int):
                days = raw_days
            elif isinstance(raw_days, str) and raw_days.isdigit():
                days = int(raw_days)
            else:
                days = None
            plan_payload = plan_service.save_plan(
                principal_id=principal.principal_id,
                conversation_id=conversation.id,
                city=str(collected_info.get("destination") or "") or None,
                days=days,
                structured_json=final_response.model_dump(),
            )
            final_response.plan_id = plan_payload.id if plan_payload else None

        conversation_service.append_message(
            conversation_id=conversation.id,
            role="assistant",
            content=final_response.answer,
            metadata={**final_response.model_dump()},
        )

        import json as _json
        yield f"data: {_json.dumps({'type': 'saved', 'conversation_id': conversation.id, 'plan_id': final_response.plan_id}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
