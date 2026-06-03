from __future__ import annotations

from fastapi import Depends, Request, Response

from app.session.service import PrincipalContext, SessionService


def get_session_service() -> SessionService:
    return SessionService()


def get_current_principal(
    request: Request,
    response: Response,
    session_service: SessionService = Depends(get_session_service),
) -> PrincipalContext:
    return session_service.get_or_create_principal(request, response)
