"""Per-customer access codes for the hosted test service.

Each customer gets a code (KYC_ACCESS_CODES = {"customer": "code", ...}, set as
a server secret). With KYC_REQUIRE_ACCESS=true every /v1 call must carry one
in the `X-Access-Code` header; the matching customer name is put on the
request, and verification cases are only visible to the customer that opened
them. Revoking a customer = removing their entry and restarting.

This is an access gate for a test deployment, not an identity system: no
accounts, no expiry, no rate limits. Production needs real authentication
(specs/00-platform.md §8).
"""

from __future__ import annotations

import hmac

from fastapi import Request
from fastapi.responses import JSONResponse

from kyc.core.config import Settings

HEADER = "X-Access-Code"
LOCAL = "local"  # the customer name when access control is off


def customer_for(code: str | None, settings: Settings) -> str | None:
    if not code:
        return None
    for customer, expected in settings.access_codes.items():
        # Constant-time compare so a code cannot be guessed character by character.
        if hmac.compare_digest(code.encode(), expected.encode()):
            return customer
    return None


def install(app, settings: Settings) -> None:
    @app.middleware("http")
    async def require_access(request: Request, call_next):
        request.state.customer = LOCAL
        if settings.require_access and request.url.path.startswith("/v1/"):
            customer = customer_for(request.headers.get(HEADER), settings)
            if customer is None:
                return JSONResponse(status_code=401, content={
                    "error": "access_denied", "message": "A valid access code is required",
                    "message_fa": "کد دسترسی معتبر نیست"})
            request.state.customer = customer
        return await call_next(request)

    @app.get("/v1/access/whoami", tags=["ops"], summary="Which customer this access code belongs to")
    def whoami(request: Request) -> dict:
        return {"customer": request.state.customer}


def current_customer(request: Request) -> str:
    """FastAPI dependency: the customer of this request."""
    return getattr(request.state, "customer", LOCAL)
