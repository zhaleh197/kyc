"""HTTP surface for face match.

The reference is the ID-card portrait (OCR returns it as `portrait_base64`).
The probe is a `capture_id` from face capture - never an uploaded selfie.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from kyc.core.config import get_settings
from kyc.core.imaging import decode_base64_image, decode_image
from kyc.core.schemas import ModuleResult
from kyc.modules.face_capture.router import get_pipeline as get_capture_pipeline

from .pipeline import FaceMatchPipeline

router = APIRouter(prefix="/v1/face-match", tags=["face-match"])

_pipeline: FaceMatchPipeline | None = None


def get_pipeline() -> FaceMatchPipeline:
    global _pipeline
    if _pipeline is None:
        # Looked up per call, so the face-capture pipeline can be swapped
        # (tests, another deployment) without rebuilding this one.
        _pipeline = FaceMatchPipeline(
            get_settings(), capture_lookup=lambda cid: get_capture_pipeline().get_capture(cid)
        )
    return _pipeline


class VerifyBase64(BaseModel):
    reference_base64: str
    capture_id: str


@router.post("/verify", response_model=ModuleResult, summary="ID card portrait vs captured selfie")
async def verify(
    reference: UploadFile = File(..., description="The ID card portrait (OCR's portrait crop)"),
    capture_id: str = Form(..., description="Selfie capture_id from /v1/face-capture"),
    pipeline: FaceMatchPipeline = Depends(get_pipeline),
) -> ModuleResult:
    data = await reference.read()
    limit = get_settings().max_upload_bytes
    if len(data) > limit:
        raise HTTPException(
            status_code=413,
            detail={"error": "payload_too_large", "message": f"Upload exceeds {limit} bytes", "size": len(data)},
        )
    return pipeline.verify(decode_image(data), capture_id)


@router.post("/verify/base64", response_model=ModuleResult, summary="ID card portrait vs captured selfie (base64)")
def verify_base64(payload: VerifyBase64, pipeline: FaceMatchPipeline = Depends(get_pipeline)) -> ModuleResult:
    return pipeline.verify(decode_base64_image(payload.reference_base64), payload.capture_id)
