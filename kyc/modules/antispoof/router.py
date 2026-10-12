"""HTTP surface for passive anti-spoofing. Input is a capture_id only."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from kyc.core.config import get_settings
from kyc.core.schemas import ModuleResult
from kyc.modules.face_capture.router import get_pipeline as get_capture_pipeline

from .pipeline import AntiSpoofPipeline

router = APIRouter(prefix="/v1/antispoof", tags=["antispoof"])

_pipeline: AntiSpoofPipeline | None = None


def get_pipeline() -> AntiSpoofPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AntiSpoofPipeline(
            get_settings(), capture_lookup=lambda cid: get_capture_pipeline().get_capture(cid)
        )
    return _pipeline


class VerifyRequest(BaseModel):
    capture_id: str


@router.post("/verify", response_model=ModuleResult, summary="Is the captured selfie a live face?")
def verify(payload: VerifyRequest, pipeline: AntiSpoofPipeline = Depends(get_pipeline)) -> ModuleResult:
    return pipeline.verify(payload.capture_id)
