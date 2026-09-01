"""API-facing shapes for the OCR module."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FieldResult(BaseModel):
    """One extracted field, with everything needed to audit the extraction.

    `raw` is kept alongside `value` on purpose: when a national id fails its
    checksum, an operator needs to see exactly what the recogniser emitted
    before normalisation to tell a misread from a forgery.
    """

    key: str
    label_fa: str = ""
    label_en: str = ""
    kind: str
    raw: str = ""
    value: str | None = None
    confidence: float = 0.0
    low_confidence: bool = False
    box: list[int] | None = Field(None, description="x1, y1, x2, y2 on the rectified card")
    detection_confidence: float | None = None
    backend: str | None = None


class DocumentOcrData(BaseModel):
    profile: str
    country: str
    fields: dict[str, FieldResult] = Field(default_factory=dict)
    values: dict[str, str | None] = Field(
        default_factory=dict, description="Flat key -> value map for callers that want just the data"
    )
    derived: dict[str, Any] = Field(
        default_factory=dict, description="Computed extras: gregorian dates, age, checksum verdict"
    )
    missing_fields: list[str] = Field(default_factory=list)
    quality: dict[str, float] = Field(default_factory=dict)
    portrait_base64: str | None = Field(None, description="Cropped photo, only when requested")
    card_base64: str | None = Field(None, description="Rectified card, only when requested")


class OcrRequestOptions(BaseModel):
    profile: str | None = None
    return_portrait: bool = False
    return_card: bool = False


class Base64OcrRequest(OcrRequestOptions):
    image_base64: str
