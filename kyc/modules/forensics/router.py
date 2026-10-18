"""HTTP surface for document forensics (06a).

The card is read by OCR first: its detector supplies the field boxes, mapped
back onto the photo exactly as submitted. Selfie forensics (06b) is not
built yet - liveness (05) and anti-spoof (04) cover the selfie for now.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile

from kyc.core.config import get_settings
from kyc.core.schemas import ModuleResult
from kyc.modules.ocr.pipeline import DocumentOcrPipeline
from kyc.modules.ocr.router import _guard_size
from kyc.modules.ocr.router import get_pipeline as get_ocr_pipeline

from .pipeline import DocumentForensicsPipeline

router = APIRouter(prefix="/v1/forensics", tags=["forensics"])

_pipeline: DocumentForensicsPipeline | None = None


def get_pipeline() -> DocumentForensicsPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DocumentForensicsPipeline(get_settings())
    return _pipeline


@router.post("/document", response_model=ModuleResult, summary="Signs that a card photo was edited")
async def check_document(
    file: UploadFile = File(..., description="The card photo as captured - not a screenshot or a re-save"),
    profile: str | None = Form(None),
    ocr: DocumentOcrPipeline = Depends(get_ocr_pipeline),
    pipeline: DocumentForensicsPipeline = Depends(get_pipeline),
) -> ModuleResult:
    data = await file.read()
    _guard_size(len(data))
    read = ocr.run_bytes(data, profile)
    return pipeline.run(data, read.data.get("source_boxes", {}))
