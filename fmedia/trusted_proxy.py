"""
fmedia/trusted_proxy.py — real visitor IP behind the Next.js proxy
-------------------------------------------------------------------
The public domain points at the Next.js service, which forwards what has not been migrated to
this app over a private network. Without help, every request would appear to come from
Next's own IP: per-IP rate limits (contact form, search) and the admin login lockout would all
share one bucket — one abuser could lock the real admin out.

A second, explicit opt-in (TRUST_X_REAL_IP=1) trusts `X-Real-IP`: on some platforms the edge
overwrites any client-supplied value with the real address, so FastAPI's own per-IP limits stop
sharing one bucket among all visitors. Never enable it where the app is reachable without that edge.

Next adds `X-Client-IP` (the visitor's address) plus `X-Internal-Token` (shared secret,
INTERNAL_API_TOKEN). This middleware rewrites the ASGI client address **only** when the secret
matches, so existing code that reads `request.client.host` keeps working. Without a valid secret
the header is ignored: a visitor cannot spoof their IP to dodge a limit.
"""
from __future__ import annotations

import hmac
import ipaddress
import os

from starlette.types import ASGIApp, Receive, Scope, Send


class TrustedClientIPMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            ip = self._trusted_ip(scope)
            if ip:
                scope = {**scope, "client": (ip, 0)}
        await self.app(scope, receive, send)

    @staticmethod
    def _trusted_ip(scope: Scope) -> str | None:
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        expected = os.getenv("INTERNAL_API_TOKEN", "")
        token = headers.get("x-internal-token", "")
        if expected and token and hmac.compare_digest(token, expected):
            return _valid_ip(headers.get("x-client-ip"))
        if os.getenv("TRUST_X_REAL_IP") == "1":
            return _valid_ip(headers.get("x-real-ip"))
        return None


def _valid_ip(value: str | None) -> str | None:
    try:
        return str(ipaddress.ip_address((value or "").strip()))
    except ValueError:
        return None
