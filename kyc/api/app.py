"""FastAPI application.

One process, one app, routers per module. This is the "modular monolith"
shape: every module keeps its own package, schemas and models, but they share
a process so a card image is decoded once and a model is loaded once.
Splitting a router into its own service later is a deployment change, not a
rewrite.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from kyc.core.config import get_settings
from kyc.core.errors import KycError
from kyc.modules.face_capture.router import router as face_capture_router
from kyc.modules.face_match.router import router as face_match_router
from kyc.modules.ocr.router import router as ocr_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("kyc")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=(
            "Identity verification services: document OCR, face quality, face matching, "
            "liveness, anti-spoofing and AI forensics."
        ),
    )

    # Empty origins means same-origin only. The prototypes this replaces all
    # shipped allow_origins=["*"], which is fine for a demo and wrong for a
    # service that receives identity documents.
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.exception_handler(KycError)
    async def handle_kyc_error(_: Request, exc: KycError) -> JSONResponse:
        if exc.http_status >= 500:
            logger.exception("module error: %s", exc.code)
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    @app.get("/health", tags=["ops"])
    def health() -> dict:
        return {"status": "ok", "modules": ["ocr", "face_quality", "face_match"]}

    app.include_router(ocr_router)
    app.include_router(face_capture_router)
    app.include_router(face_match_router)
    if settings.debug:
        from kyc.api.dev import router as dev_router

        app.include_router(dev_router)
    return app


app = create_app()
