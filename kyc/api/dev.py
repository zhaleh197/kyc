"""Development-only pages. Mounted only when KYC_DEBUG=true.

The face-capture test page drives the real /v1/face-capture API from a
browser webcam, shows the per-frame measurements, and can save labelled
frames to the developer's own disk for threshold calibration
(scripts/calibrate_face_quality.py). Nothing here stores images server-side.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

STATIC = Path(__file__).parent / "static"

router = APIRouter(prefix="/dev", tags=["dev"], include_in_schema=False)


@router.get("/face-capture", response_class=HTMLResponse)
def face_capture_page() -> str:
    return (STATIC / "face_capture.html").read_text(encoding="utf-8")
