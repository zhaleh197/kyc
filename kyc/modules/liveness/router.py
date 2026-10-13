"""HTTP surface for active liveness. A session is bound to a capture_id."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from kyc.core.config import get_settings
from kyc.core.imaging import decode_base64_image, decode_image
from kyc.core.schemas import ModuleResult
from kyc.modules.face_capture.router import get_pipeline as get_capture_pipeline

from .pipeline import LivenessPipeline
from .tasks import INSTRUCTIONS_EN, INSTRUCTIONS_FA

router = APIRouter(prefix="/v1/liveness", tags=["liveness"])

_pipeline: LivenessPipeline | None = None


def get_pipeline() -> LivenessPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = LivenessPipeline(
            get_settings(), capture_lookup=lambda cid: get_capture_pipeline().get_capture(cid)
        )
    return _pipeline


class StartRequest(BaseModel):
    capture_id: str


class FrameBase64(BaseModel):
    image_base64: str


@router.post("/sessions", status_code=201, summary="Start liveness for a captured selfie")
def start(payload: StartRequest, pipeline: LivenessPipeline = Depends(get_pipeline)) -> dict:
    session = pipeline.start(payload.capture_id)
    cfg = pipeline.settings.liveness
    return {
        "session_id": session.session_id,
        "tasks": [{"task": t.name, "instruction_fa": INSTRUCTIONS_FA[t.name], "instruction_en": INSTRUCTIONS_EN[t.name]}
                  for t in session.tasks],
        "task_time_limit_s": cfg.task_time_limit_s,
        "first_instruction_fa": INSTRUCTIONS_FA["BASELINE"],
    }


@router.post("/sessions/{session_id}/frames", summary="Submit one camera frame")
async def frame(
    session_id: str, file: UploadFile = File(...), pipeline: LivenessPipeline = Depends(get_pipeline)
) -> dict:
    data = await file.read()
    limit = get_settings().max_upload_bytes
    if len(data) > limit:
        raise HTTPException(status_code=413, detail={"error": "payload_too_large", "size": len(data)})
    return pipeline.submit_frame(session_id, decode_image(data))


@router.post("/sessions/{session_id}/frames/base64", summary="Submit one camera frame (base64)")
def frame_base64(session_id: str, payload: FrameBase64, pipeline: LivenessPipeline = Depends(get_pipeline)) -> dict:
    return pipeline.submit_frame(session_id, decode_base64_image(payload.image_base64))


@router.get("/sessions/{session_id}/result", response_model=ModuleResult)
def result(session_id: str, pipeline: LivenessPipeline = Depends(get_pipeline)) -> ModuleResult:
    return pipeline.result(session_id)
