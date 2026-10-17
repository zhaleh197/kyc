"""The customer-facing web app (/) and its assets. Public: the page holds no
data - every /v1 call it makes carries the customer's access code."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse

STATIC = Path(__file__).parent / "static"

router = APIRouter(include_in_schema=False)


@router.get("/", response_class=HTMLResponse)
def app_page() -> str:
    return (STATIC / "app.html").read_text(encoding="utf-8")


@router.get("/assets/logo.svg")
def logo() -> FileResponse:
    # Replace kyc/api/static/logo.svg with the real Ekam logo.
    return FileResponse(STATIC / "logo.svg", media_type="image/svg+xml")
