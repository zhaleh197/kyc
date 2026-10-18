"""Combining module results into one verdict. Pure function, no I/O.

Rule-based, deliberately not a weighted sum: a weighted sum lets a strong OCR
score compensate for a failed liveness - exactly the wrong trade in KYC.
Scores only rank the manual-review queue.

    a required step missing or a module unavailable  -> manual_review (never approved)
    a module failed with a reason in reject_codes    -> rejected
    a module failed otherwise                        -> manual_review
    a module returned review                         -> manual_review
    a warning listed in reject_warning_codes         -> rejected
    everything passed                                -> approved
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from kyc.core.config import OrchestratorSettings
from kyc.core.schemas import Decision, ModuleResult, Severity

# The order steps appear in a verdict, and what each is called in it.
STEPS = ("document", "forensics", "selfie", "antispoof", "face_match", "liveness")
# Steps that may be switched off for a deployment: absent means skipped, not
# missing. Forensics is advisory (spec 06): it can only ask for a human look.
OPTIONAL_STEPS = ("forensics",)


class Verdict(str, Enum):
    APPROVED = "approved"
    MANUAL_REVIEW = "manual_review"
    REJECTED = "rejected"


UNAVAILABLE = "unavailable"  # the module raised (model missing, backend down)


@dataclass
class BlockingReason:
    step: str
    code: str
    message_fa: str
    message_en: str
    outcome: Verdict

    def as_dict(self) -> dict:
        return {"step": self.step, "code": self.code, "outcome": self.outcome.value,
                "message_fa": self.message_fa, "message_en": self.message_en}


@dataclass
class PolicyResult:
    verdict: Verdict
    blocking: list[BlockingReason] = field(default_factory=list)
    review_score: float = 0.0  # lower = look at it sooner; not a decision input


def decide(
    results: dict[str, ModuleResult | str | None], cfg: OrchestratorSettings, required: tuple[str, ...]
) -> PolicyResult:
    """`results` maps step -> ModuleResult, UNAVAILABLE, or None (not run)."""
    blocking: list[BlockingReason] = []
    for step in STEPS:
        result = results.get(step)
        if step not in required and result is None:
            continue
        if result is None:
            blocking.append(BlockingReason(step, "step_missing", "این مرحله انجام نشد", "Step not completed",
                                           Verdict.MANUAL_REVIEW))
            continue
        if result == UNAVAILABLE:
            blocking.append(BlockingReason(step, "module_unavailable", "سرویس این مرحله در دسترس نبود",
                                           "The module could not run", Verdict.MANUAL_REVIEW))
            continue
        for reason in result.reasons:
            if reason.severity is Severity.ERROR:
                outcome = Verdict.REJECTED if reason.code in cfg.reject_codes else Verdict.MANUAL_REVIEW
            elif reason.severity is Severity.WARN:
                outcome = Verdict.REJECTED if reason.code in cfg.reject_warning_codes else Verdict.MANUAL_REVIEW
            else:
                continue
            blocking.append(BlockingReason(step, reason.code, reason.message_fa, reason.message_en, outcome))
        if result.decision is not Decision.PASS and not any(b.step == step for b in blocking):
            # A non-pass with no blocking reason must still stop approval.
            outcome = Verdict.MANUAL_REVIEW
            blocking.append(BlockingReason(step, f"{step}_{result.decision.value}", "نتیجهٔ این مرحله قطعی نبود",
                                           f"Step result: {result.decision.value}", outcome))

    if any(b.outcome is Verdict.REJECTED for b in blocking):
        verdict = Verdict.REJECTED
    elif blocking:
        verdict = Verdict.MANUAL_REVIEW
    else:
        verdict = Verdict.APPROVED
    scores = [r.score for r in results.values() if isinstance(r, ModuleResult)]
    return PolicyResult(verdict, blocking, round(min(scores), 4) if scores else 0.0)
