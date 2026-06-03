from __future__ import annotations

import hashlib
import secrets
from typing import Literal

from fastapi import Response

from app.core.settings import get_settings

CookieSameSite = Literal["lax", "strict", "none"]


def generate_session_key() -> str:
    return secrets.token_urlsafe(32)


def hash_ip(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def set_session_cookie(response: Response, session_key: str) -> None:
    settings = get_settings()
    same_site = settings.session_cookie_samesite.lower()
    response.set_cookie(
        key=settings.session_cookie_name,
        value=session_key,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite=_normalize_samesite(same_site),
        max_age=settings.session_cookie_max_age_seconds,
    )


def _normalize_samesite(value: str) -> CookieSameSite:
    normalized = value.lower()
    if normalized not in {"lax", "strict", "none"}:
        return "lax"
    return normalized  # type: ignore[return-value]
