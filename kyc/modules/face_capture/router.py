"""HTTP surface for face capture.

There is deliberately no endpoint that accepts a selfie file and returns a
capture_id: a selfie only exists if it came through a capture session
(specs/02-face-quality.md §2). `/v1/face-quality/check` judges an image but
never produces a capture_id, so its output cannot reach face match.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from kyc.core.config import get_settings
from kyc.core.imaging import decode_base64_image, decode_image
from kyc.core.schemas import ModuleResult

from .pipeline import FaceCapturePipeline

router = APIRouter(tags=["face-capture"])

_pipeline: FaceCapturePipeline | None = None


def get_pipeline() -> FaceCapturePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = FaceCapturePipeline(get_settings())
    return _pipeline


class FrameBase64(BaseModel):
    image_base64: str


class SessionCreated(BaseModel):
    session_id: str
    expires_in_s: int


@router.post("/v1/face-capture/sessions", response_model=SessionCreated, status_code=201)
def create_session(pipeline: FaceCapturePipeline = Depends(get_pipeline)) -> SessionCreated:
    session = pipeline.start()
    return SessionCreated(session_id=session.session_id, expires_in_s=pipeline.settings.face_quality.session_ttl_s)


@router.post("/v1/face-capture/sessions/{session_id}/frames", summary="Submit one camera frame")
async def submit_frame(
    session_id: str,
    file: UploadFile = File(..., description="One JPEG/PNG camera frame"),
    pipeline: FaceCapturePipeline = Depends(get_pipeline),
) -> dict:
    data = await file.read()
    _guard_size(len(data))
    return pipeline.submit_frame(session_id, decode_image(data))


@router.post("/v1/face-capture/sessions/{session_id}/frames/base64", summary="Submit one camera frame (base64)")
def submit_frame_base64(
    session_id: str,
    payload: FrameBase64,
    pipeline: FaceCapturePipeline = Depends(get_pipeline),
) -> dict:
    return pipeline.submit_frame(session_id, decode_base64_image(payload.image_base64))


@router.get("/v1/face-capture/sessions/{session_id}/result", response_model=ModuleResult)
def session_result(
    session_id: str,
    include_selfie: bool = Query(False, description="Include the captured selfie as base64 JPEG"),
    pipeline: FaceCapturePipeline = Depends(get_pipeline),
) -> ModuleResult:
    return pipeline.result(session_id, include_selfie=include_selfie)


@router.post("/v1/face-quality/check", response_model=ModuleResult, summary="Judge one image (no capture)")
async def check_image(
    file: UploadFile = File(...),
    pipeline: FaceCapturePipeline = Depends(get_pipeline),
) -> ModuleResult:
    data = await file.read()
    _guard_size(len(data))
    return pipeline.check(decode_image(data))


def _guard_size(size: int) -> None:
    limit = get_settings().max_upload_bytes
    if size > limit:
        raise HTTPException(
            status_code=413,
            detail={"error": "payload_too_large", "message": f"Upload exceeds {limit} bytes", "size": size},
        )
