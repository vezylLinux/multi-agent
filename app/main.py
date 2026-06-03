from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.router import router as api_router
from app.core.settings import get_settings
from app.core.database import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


settings = get_settings()
app = FastAPI(title="Multi Agent Travel API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix="/api")


@app.get("/")
def root() -> dict:
    return {
        "message": "Multi Agent Travel API is running.",
        "endpoints": {
            "health": "/health",
            "chat": "/api/chat/send",
            "session": "/api/session/me",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}
