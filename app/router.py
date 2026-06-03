from fastapi import APIRouter

from app.chat.routes import router as chat_router
from app.itinerary.routes import router as route_router
from app.session.routes import conversations_router, plan_router, session_router

router = APIRouter()
router.include_router(session_router, prefix="/session", tags=["session"])
router.include_router(chat_router, prefix="/chat", tags=["chat"])
router.include_router(conversations_router, prefix="/conversations", tags=["conversations"])
router.include_router(plan_router, prefix="/plans", tags=["plans"])
router.include_router(route_router, prefix="/route", tags=["route"])
