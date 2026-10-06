"""HTTP surface for the OCR module.

Endpoints are versioned (`/v1`) from day one: an identity pipeline is
integrated by other teams, and changing a field name later without a version
break is not an option.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from kyc.core.config import get_settings
from kyc.core.imaging import decode_base64_image
from kyc.core.schemas import ModuleResult

from .pipeline import DocumentOcrPipeline
from .profiles import list_profiles
from .schemas import Base64OcrRequest

router = APIRouter(prefix="/v1/ocr", tags=["ocr"])

_pipeline: DocumentOcrPipeline | None = None


def get_pipeline() -> DocumentOcrPipeline:
    """Injected as a dependency so a deployment (or a test) can substitute a
    pipeline built with different models without touching the routes."""
    global _pipeline
    if _pipeline is None:
        _pipeline = DocumentOcrPipeline(get_settings())
    return _pipeline


@router.get("/profiles", summary="Supported document types")
def get_profiles() -> list[dict]:
    return [
        {
            "id": profile.id,
            "country": profile.country,
            "name_en": profile.name_en,
            "name_fa": profile.name_fa,
            "fields": [
                {
                    "key": f.key,
                    "kind": f.kind.value,
                    "label_fa": f.label_fa,
                    "label_en": f.label_en,
                    "required": f.required,
                }
                for f in profile.fields
            ],
        }
        for profile in list_profiles()
    ]


@router.post("/document", response_model=ModuleResult, summary="Read an identity document")
async def read_document(
    file: UploadFile = File(..., description="Photo or scan of the document"),
    profile: str | None = Form(None, description="Document profile id; defaults to the Iranian national card"),
    return_portrait: bool = Form(False, description="Include the cropped photo, base64 JPEG"),
    return_card: bool = Form(False, description="Include the rectified card, base64 JPEG"),
    pipeline: DocumentOcrPipeline = Depends(get_pipeline),
) -> ModuleResult:
    data = await file.read()
    _guard_size(len(data))
    return pipeline.run_bytes(
        data,
        profile,
        return_portrait=return_portrait,
        return_card=return_card,
    )


@router.post("/document/base64", response_model=ModuleResult, summary="Read an identity document (base64)")
def read_document_base64(
    payload: Base64OcrRequest,
    pipeline: DocumentOcrPipeline = Depends(get_pipeline),
) -> ModuleResult:
    image = decode_base64_image(payload.image_base64)
    return pipeline.run(
        image,
        payload.profile,
        return_portrait=payload.return_portrait,
        return_card=payload.return_card,
    )


def _guard_size(size: int) -> None:
    limit = get_settings().max_upload_bytes
    if size > limit:
        raise HTTPException(
            status_code=413,
            detail={"error": "payload_too_large", "message": f"Upload exceeds {limit} bytes", "size": size},
        )
