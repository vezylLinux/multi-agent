from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from typing import AsyncIterator

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    clarify_response_node,
    intake_node,
    planning_node,
    response_node,
    retrieval_node,
    route_after_intake,
    route_after_validation,
    validator_node,
)
from app.graph.state import TravelGraphState
from app.session.schemas import ChatResponse


def build_travel_graph():
    workflow = StateGraph(TravelGraphState)

    workflow.add_node("intake", intake_node)
    workflow.add_node("clarify_response", clarify_response_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("response", response_node)

    workflow.add_edge(START, "intake")
    workflow.add_conditional_edges(
        "intake",
        route_after_intake,
        {
            "clarify": "clarify_response",
            "planning": "retrieval",
        },
    )
    workflow.add_edge("clarify_response", END)
    workflow.add_edge("retrieval", "planning")
    workflow.add_edge("planning", "validator")
    workflow.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "planning": "planning",
            "response": "response",
        },
    )
    workflow.add_edge("response", END)

    return workflow.compile()


@lru_cache
def get_travel_graph():
    return build_travel_graph()


def run_travel_graph(
    message: str,
    *,
    top_k: int = 5,
    with_plan: bool = True,
    category: str | None = None,
) -> ChatResponse:
    state = get_travel_graph().invoke(
        {
            "message": message,
            "top_k": top_k,
            "with_plan": with_plan,
            "category": category,
            "trace": [],
        }
    )
    return ChatResponse(**state["response_payload"])


_NODE_LABELS: dict[str, str] = {
    "intake": "Intake Agent",
    "retrieval": "Retrieval Agent",
    "planning": "Planning Agent",
    "validator": "Validator Agent",
    "clarify_response": "Clarifying",
    "response": "Building Response",
}


def _node_done_text(node: str, update: dict) -> str:
    if node == "intake":
        info = update.get("collected_info") or {}
        dest = str(info.get("destination") or "")
        days = str(info.get("days") or "")
        interests = str(info.get("interests") or "")
        parts = [p for p in [dest, f"{days} days" if days else "", interests] if p]
        return " · ".join(parts) or "Complete"
    if node == "retrieval":
        places = update.get("places") or []
        n = len(places) if isinstance(places, list) else 0
        hotel = update.get("recommended_hotel")
        hotel_name = hotel.get("name", "") if isinstance(hotel, dict) else ""
        suffix = f", hotel: {hotel_name}" if hotel_name else ""
        return f"Retrieved {n} places{suffix}"
    if node == "planning":
        return "Itinerary built"
    if node == "validator":
        return "Replanning..." if update.get("needs_replan") else "Plan validated"
    if node == "clarify_response":
        return "Follow-up questions ready"
    if node == "response":
        return "Response ready"
    return "Complete"


async def stream_travel_graph(
    message: str,
    *,
    top_k: int = 5,
    with_plan: bool = True,
    category: str | None = None,
) -> AsyncIterator[str]:
    def _sse(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    graph = get_travel_graph()
    final_payload: dict | None = None

    async for chunk in graph.astream(
        {
            "message": message,
            "top_k": top_k,
            "with_plan": with_plan,
            "category": category,
            "trace": [],
        },
        stream_mode="updates",
    ):
        for node_name, state_update in chunk.items():
            if not isinstance(state_update, dict):
                continue

            label = _NODE_LABELS.get(node_name, node_name)
            done_text = _node_done_text(node_name, state_update)

            yield _sse({"type": "node_done", "node": node_name, "label": label, "text": done_text})

            if node_name == "planning":
                plan_text = str(state_update.get("plan") or "")
                for line in plan_text.splitlines(keepends=True):
                    yield _sse({"type": "text_chunk", "node": "planning", "text": line})
                    await asyncio.sleep(0.008)

            if "response_payload" in state_update:
                final_payload = state_update["response_payload"]

    if final_payload is None:
        yield _sse({"type": "error", "message": "No response payload generated"})
        return

    yield _sse({"type": "done", "response": final_payload})
