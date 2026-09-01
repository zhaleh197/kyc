"""The contract every KYC module returns.

The whole system is built on one idea: a module never returns a bare bool.
It returns a `ModuleResult` carrying a decision, a score, and the *reasons*
behind it. The orchestrator aggregates those into a final verdict, and the
reasons become the audit trail an operator reads during manual review.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Decision(str, Enum):
    PASS = "pass"
    REVIEW = "review"
    FAIL = "fail"


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class Reason(BaseModel):
    """One machine-readable finding.

    `code` is stable and safe to branch on; the messages are for humans.
    Persian text is carried alongside English so the client does not have to
    maintain its own translation table.
    """

    code: str
    severity: Severity = Severity.INFO
    message_en: str = ""
    message_fa: str = ""
    field: str | None = None

    @classmethod
    def error(cls, code: str, en: str, fa: str, field: str | None = None) -> Reason:
        return cls(code=code, severity=Severity.ERROR, message_en=en, message_fa=fa, field=field)

    @classmethod
    def warn(cls, code: str, en: str, fa: str, field: str | None = None) -> Reason:
        return cls(code=code, severity=Severity.WARN, message_en=en, message_fa=fa, field=field)

    @classmethod
    def info(cls, code: str, en: str, fa: str, field: str | None = None) -> Reason:
        return cls(code=code, severity=Severity.INFO, message_en=en, message_fa=fa, field=field)


class ModuleResult(BaseModel):
    """Uniform envelope returned by ocr / face-quality / liveness / antispoof / forensic."""

    module: str
    version: str = "0.1.0"
    decision: Decision
    score: float = Field(ge=0.0, le=1.0, description="Confidence that this check passed")
    reasons: list[Reason] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict, description="Module-specific payload")
    elapsed_ms: float = 0.0

    @property
    def errors(self) -> list[Reason]:
        return [r for r in self.reasons if r.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Reason]:
        return [r for r in self.reasons if r.severity is Severity.WARN]


class Timer:
    """with Timer() as t: ...   -> t.ms"""

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        self.ms = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000.0
        return None
