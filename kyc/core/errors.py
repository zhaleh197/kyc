"""Typed errors shared by every KYC module.

Every module raises one of these instead of leaking library-specific
exceptions, so the API layer can map them to stable HTTP codes and the
orchestrator can decide whether a failure is the user's fault (retry with a
better photo) or ours (model missing, backend down).
"""

from __future__ import annotations


class KycError(Exception):
    """Base class. `code` is a stable machine-readable string."""

    code = "kyc_error"
    http_status = 500

    def __init__(self, message: str, *, code: str | None = None, details: dict | None = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"error": self.code, "message": self.message, "details": self.details}


class InvalidInput(KycError):
    """Caller sent something we cannot even decode."""

    code = "invalid_input"
    http_status = 400


class ImageQualityError(KycError):
    """Image decoded fine but is unusable (too small, too blurry, too dark).

    Distinct from InvalidInput because the correct client action is
    "take another photo", not "fix your request".
    """

    code = "image_quality"
    http_status = 422


class ModelNotAvailable(KycError):
    """A required model artifact is missing or failed to load."""

    code = "model_not_available"
    http_status = 503


class BackendNotInstalled(KycError):
    """An optional dependency for the selected backend is not installed."""

    code = "backend_not_installed"
    http_status = 503


class ProfileNotFound(KycError):
    code = "profile_not_found"
    http_status = 404


class CaptureNotFound(KycError):
    """A capture_id is unknown or its selfie has expired - take a new selfie.
    Shared by every module that works on a captured selfie."""

    code = "capture_not_found"
    http_status = 404
