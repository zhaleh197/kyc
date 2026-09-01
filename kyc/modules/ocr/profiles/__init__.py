"""Profile registry. Adding a country means adding a module and one entry here."""

from __future__ import annotations

from kyc.core.errors import ProfileNotFound

from .base import DocumentProfile, FieldKind, FieldSpec
from .iran import IR_NATIONAL_CARD_FRONT

_REGISTRY: dict[str, DocumentProfile] = {
    IR_NATIONAL_CARD_FRONT.id: IR_NATIONAL_CARD_FRONT,
}

DEFAULT_PROFILE_ID = IR_NATIONAL_CARD_FRONT.id


def register(profile: DocumentProfile) -> None:
    _REGISTRY[profile.id] = profile


def get_profile(profile_id: str | None) -> DocumentProfile:
    key = profile_id or DEFAULT_PROFILE_ID
    try:
        return _REGISTRY[key]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY))
        raise ProfileNotFound(f"Unknown document profile {key!r}. Available: {available}") from None


def list_profiles() -> list[DocumentProfile]:
    return sorted(_REGISTRY.values(), key=lambda p: p.id)


__all__ = [
    "DocumentProfile",
    "FieldKind",
    "FieldSpec",
    "IR_NATIONAL_CARD_FRONT",
    "DEFAULT_PROFILE_ID",
    "register",
    "get_profile",
    "list_profiles",
]
