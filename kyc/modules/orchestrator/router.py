"""HTTP surface for cases - the endpoint a client integrates against.

    POST /v1/cases                         -> case_id
    POST /v1/cases/{id}/document           card photo -> OCR (stored on the case)
    POST /v1/cases/{id}/capture            -> capture session id; stream frames and
                                              the still to /v1/face-capture/sessions/...
    POST /v1/cases/{id}/liveness           -> liveness session id + tasks; stream
                                              frames to /v1/liveness/sessions/...
    POST /v1/cases/{id}/face-match         -> card portrait vs selfie now (reused by finalize)
    POST /v1/cases/{id}/finalize           -> verdict
    GET  /v1/cases/{id}                    -> state, or the verdict once finalized
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from kyc.core.config import get_settings
from kyc.core.schemas import ModuleResult
from kyc.modules.antispoof.router import get_pipeline as antispoof_pipeline
from kyc.modules.face_capture.router import get_pipeline as capture_pipeline
from kyc.modules.face_match.router import get_pipeline as match_pipeline
from kyc.modules.liveness.router import get_pipeline as liveness_pipeline
from kyc.modules.liveness.tasks import INSTRUCTIONS_EN, INSTRUCTIONS_FA
from kyc.modules.ocr.router import get_pipeline as ocr_pipeline

from .service import CaseService

router = APIRouter(prefix="/v1/cases", tags=["cases"])

_service: CaseService | None = None


def get_service() -> CaseService:
    global _service
    if _service is None:
        _service = CaseService(
            get_settings(), ocr=ocr_pipeline(), capture=capture_pipeline(), antispoof=antispoof_pipeline(),
            face_match=match_pipeline(), liveness=liveness_pipeline(),
        )
    return _service


@router.post("", status_code=201, summary="Open a verification case")
def create(service: CaseService = Depends(get_service)) -> dict:
    case = service.create()
    cfg = service.settings.orchestrator
    return {"case_id": case.case_id, "expires_in_s": cfg.case_ttl_s, "require_liveness": cfg.require_liveness,
            "policy_version": cfg.policy_version}


@router.post("/{case_id}/document", response_model=ModuleResult, summary="ID card photo -> OCR")
async def document(
    case_id: str,
    file: UploadFile = File(...),
    profile: str | None = Form(None),
    service: CaseService = Depends(get_service),
) -> ModuleResult:
    data = await file.read()
    limit = get_settings().max_upload_bytes
    if len(data) > limit:
        raise HTTPException(status_code=413, detail={"error": "payload_too_large", "size": len(data)})
    return service.submit_document(case_id, data, profile)


class CaptureOptions(BaseModel):
    # Dev page only - face capture refuses per-session modes outside debug.
    check_modes: dict[str, str] = {}


@router.post("/{case_id}/capture", summary="Start the selfie capture session for this case")
def capture(case_id: str, options: CaptureOptions | None = None, service: CaseService = Depends(get_service)) -> dict:
    return {"capture_session_id": service.start_capture(case_id, options.check_modes if options else None)}


@router.post("/{case_id}/liveness", summary="Start liveness on this case's selfie")
def liveness(case_id: str, service: CaseService = Depends(get_service)) -> dict:
    session = service.start_liveness(case_id)
    return {
        "liveness_session_id": session.session_id,
        "tasks": [{"task": t.name, "instruction_fa": INSTRUCTIONS_FA[t.name], "instruction_en": INSTRUCTIONS_EN[t.name]}
                  for t in session.tasks],
        "first_instruction_fa": INSTRUCTIONS_FA["BASELINE"],
    }


@router.post("/{case_id}/face-match", response_model=ModuleResult, summary="Card portrait vs selfie, now")
def face_match(case_id: str, service: CaseService = Depends(get_service)) -> ModuleResult:
    return service.match_now(case_id)


@router.post("/{case_id}/finalize", summary="Combine every step into the verdict")
def finalize(case_id: str, service: CaseService = Depends(get_service)) -> dict:
    return service.finalize(case_id)


@router.get("/{case_id}", summary="Case state, or its verdict once finalized")
def view(case_id: str, service: CaseService = Depends(get_service)) -> dict:
    return service.view(case_id)
