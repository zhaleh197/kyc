"""HTTP surface for face capture.

There is deliberately no endpoint that accepts a selfie file and returns a
capture_id: a selfie only exists if it came through a capture session
(specs/02-face-quality.md §2). The still photo is accepted only inside a
session whose video stream has already qualified, and only if it shows the
same person as that stream. `/v1/face-quality/check` judges an image but
never produces a capture_id, so its output cannot reach face match.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from kyc.core.config import get_settings
from kyc.core.errors import InvalidInput
from kyc.core.imaging import decode_base64_image, decode_image
from kyc.core.schemas import ModuleResult

from . import quality
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


class SessionOptions(BaseModel):
    # Dev page only: honoured with KYC_DEBUG=true, refused (403) otherwise.
    # {"group": "off" | "info" | "block"}
    check_modes: dict[str, str] = {}


class SessionCreated(BaseModel):
    session_id: str
    expires_in_s: int


@router.post("/v1/face-capture/sessions", response_model=SessionCreated, status_code=201)
def create_session(
    options: SessionOptions | None = None, pipeline: FaceCapturePipeline = Depends(get_pipeline)
) -> SessionCreated:
    session = pipeline.start(options.check_modes if options else None)
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


@router.post("/v1/face-capture/sessions/{session_id}/still", summary="Submit the still photo (the selfie)")
async def submit_still(
    session_id: str,
    file: UploadFile = File(..., description="Full-resolution photo from the same camera"),
    pipeline: FaceCapturePipeline = Depends(get_pipeline),
) -> dict:
    """Accepted only once the stream has qualified (state READY_FOR_STILL)."""
    data = await file.read()
    _guard_size(len(data))
    return pipeline.submit_still(session_id, decode_image(data))


@router.post("/v1/face-capture/sessions/{session_id}/still/base64", summary="Submit the still photo (base64)")
def submit_still_base64(
    session_id: str,
    payload: FrameBase64,
    pipeline: FaceCapturePipeline = Depends(get_pipeline),
) -> dict:
    return pipeline.submit_still(session_id, decode_base64_image(payload.image_base64))


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
    check_modes: str | None = Form(
        None, description='Debug only: JSON {"group": "off" | "info" | "block"}; refused (403) otherwise'
    ),
    pipeline: FaceCapturePipeline = Depends(get_pipeline),
) -> ModuleResult:
    modes = _parse_modes(check_modes)
    data = await file.read()
    _guard_size(len(data))
    return pipeline.check(decode_image(data), modes)


class CheckModes(BaseModel):
    adjustable: bool  # per-request check_modes are honoured (KYC_DEBUG / KYC_ADJUSTABLE_CHECKS)
    modes: dict[str, str]  # each group's mode on this server
    groups: dict[str, list[str]]  # group -> reason codes it produces


@router.get("/v1/face-quality/check-modes", response_model=CheckModes, summary="The selfie checks and their modes")
def check_modes(pipeline: FaceCapturePipeline = Depends(get_pipeline)) -> CheckModes:
    return CheckModes(
        adjustable=pipeline.settings.checks_adjustable,
        modes=pipeline.validate_modes(None),
        groups={g: list(codes) for g, codes in quality.CHECK_GROUPS.items()},
    )


def _parse_modes(raw: str | None) -> dict[str, str] | None:
    if not raw:
        return None
    try:
        modes = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidInput("check_modes is not valid JSON") from exc
    if not isinstance(modes, dict) or not all(isinstance(v, str) for v in modes.values()):
        raise InvalidInput('check_modes must be a JSON object of {"group": "off" | "info" | "block"}')
    return modes


def _guard_size(size: int) -> None:
    limit = get_settings().max_upload_bytes
    if size > limit:
        raise HTTPException(
            status_code=413,
            detail={"error": "payload_too_large", "message": f"Upload exceeds {limit} bytes", "size": size},
        )
