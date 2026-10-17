"""One customer's verification as a case.

    POST case -> document (01 OCR) -> capture session (02) -> liveness (05)
              -> finalize: 04 anti-spoof + 03 face match on the captured
                 selfie, every result through policy.decide() -> verdict

The server holds every link: the case starts the capture session, so the
selfie it judges is the one this case captured; liveness starts from that
selfie; face match uses the portrait OCR cropped from this case's card.
Nothing a client sends can swap one of them for another.

Modules are reached through their public pipeline methods only, injected so
the suite runs with fakes. In memory per process, like the other session
stores; a shared store is the same seam later.
"""

from __future__ import annotations

import base64
import secrets
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from kyc.core.config import Settings, get_settings
from kyc.core.errors import CaptureNotFound, InvalidInput, KycError
from kyc.core.imaging import decode_image
from kyc.core.schemas import ModuleResult

from .policy import STEPS, UNAVAILABLE, decide


class CaseNotFound(KycError):
    code = "case_not_found"
    http_status = 404


class CaseConflict(KycError):
    """The case cannot take this step now (finalized, out of order, out of attempts)."""

    code = "case_conflict"
    http_status = 409


@dataclass
class Case:
    case_id: str
    expires_at: float
    created_at: float = field(default_factory=time.time)
    document: ModuleResult | None = None
    portrait_b64: str | None = None  # kept server-side for face match; never returned
    document_attempts: int = 0
    capture_session_id: str | None = None
    capture_attempts: int = 0
    liveness_session_id: str | None = None
    liveness_attempts: int = 0
    verdict: dict | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def _public(result: ModuleResult) -> ModuleResult:
    """A module result without images, for storing and showing."""
    data = {k: v for k, v in result.data.items() if not k.endswith("_base64")}
    return result.model_copy(update={"data": data})


class CaseService:
    def __init__(self, settings: Settings | None = None, *, ocr, capture, antispoof, face_match, liveness):
        self.settings = settings or get_settings()
        self.ocr, self.capture, self.antispoof = ocr, capture, antispoof
        self.face_match, self.liveness = face_match, liveness
        self._cases: dict[str, Case] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ cases

    def create(self) -> Case:
        cfg = self.settings.orchestrator
        case = Case(case_id=secrets.token_urlsafe(16), expires_at=time.time() + cfg.case_ttl_s)
        with self._lock:
            now = time.time()
            for cid in [c for c, v in self._cases.items() if now - v.expires_at > cfg.case_ttl_s]:
                del self._cases[cid]
            self._cases[case.case_id] = case
        return case

    def get(self, case_id: str) -> Case:
        with self._lock:
            case = self._cases.get(case_id)
        if case is None:
            raise CaseNotFound("Unknown or expired case")
        return case

    def _open(self, case: Case) -> None:
        if case.verdict is not None:
            raise CaseConflict("The case is already finalized", code="case_finalized")
        if time.time() >= case.expires_at:
            raise CaseConflict("The case has expired", code="case_expired")

    @staticmethod
    def _attempt(used: int, limit: int, what: str) -> None:
        if used >= limit:
            raise CaseConflict(f"No {what} attempts left ({limit})", code="attempts_exhausted",
                               details={"step": what, "limit": limit})

    # ------------------------------------------------------------------ steps

    def submit_document(self, case_id: str, data: bytes, profile: str | None = None) -> ModuleResult:
        cfg = self.settings.orchestrator
        case = self.get(case_id)
        with case.lock:
            self._open(case)
            self._attempt(case.document_attempts, cfg.max_document_attempts, "document")
            case.document_attempts += 1
            result = self.ocr.run_bytes(data, profile, return_portrait=True)
            case.document = _public(result)
            case.portrait_b64 = result.data.get("portrait_base64")
            return case.document

    def start_capture(self, case_id: str, check_modes: dict | None = None) -> str:
        """`check_modes` is passed to face capture, which honours it only in
        debug mode (dev page); in production it refuses with 403."""
        cfg = self.settings.orchestrator
        case = self.get(case_id)
        with case.lock:
            self._open(case)
            self._attempt(case.capture_attempts, cfg.max_capture_attempts, "selfie")
            case.capture_attempts += 1
            started = self.capture.start(check_modes=check_modes) if check_modes else self.capture.start()
            case.capture_session_id = started.session_id
            case.liveness_session_id = None  # a new selfie needs its own liveness
            return case.capture_session_id

    def start_liveness(self, case_id: str):
        cfg = self.settings.orchestrator
        case = self.get(case_id)
        with case.lock:
            self._open(case)
            capture_id = self._capture_id(case)
            if capture_id is None:
                raise CaseConflict("Take the selfie before liveness", code="selfie_missing")
            self._attempt(case.liveness_attempts, cfg.max_liveness_attempts, "liveness")
            case.liveness_attempts += 1
            session = self.liveness.start(capture_id)
            case.liveness_session_id = session.session_id
            return session

    def _selfie_result(self, case: Case) -> ModuleResult | None:
        if case.capture_session_id is None:
            return None
        try:
            return self.capture.result(case.capture_session_id)
        except KycError as exc:
            if exc.http_status >= 500:
                raise
            return None  # still capturing, or the session is gone

    def _capture_id(self, case: Case) -> str | None:
        selfie = self._selfie_result(case)
        return selfie.data.get("capture_id") if selfie is not None else None

    # ------------------------------------------------------------------ finalize

    def finalize(self, case_id: str) -> dict:
        cfg = self.settings.orchestrator
        case = self.get(case_id)
        with case.lock:
            if case.verdict is not None:
                return case.verdict  # idempotent
            self._open(case)
            selfie = self._selfie_result(case)
            capture_id = selfie.data.get("capture_id") if selfie is not None else None
            missing = [s for s, ok in (("document", case.document is not None), ("selfie", capture_id is not None))
                       if not ok]
            if cfg.require_liveness and case.liveness_session_id is None:
                missing.append("liveness")
            if missing:
                raise CaseConflict(f"Steps not done yet: {', '.join(missing)}", code="case_incomplete",
                                   details={"missing": missing})

            results: dict[str, ModuleResult | str | None] = {"document": case.document, "selfie": _public(selfie)}
            results["antispoof"] = self._run(lambda: self.antispoof.verify(capture_id))
            results["face_match"] = (
                self._run(lambda: self.face_match.verify(_decode_b64(case.portrait_b64), capture_id))
                if case.portrait_b64 else None
            )
            results["liveness"] = None
            if case.liveness_session_id is not None:
                results["liveness"] = self._run(lambda: self.liveness.result(case.liveness_session_id),
                                                not_finished=None)

            required = STEPS if cfg.require_liveness else tuple(s for s in STEPS if s != "liveness")
            policy = decide(results, cfg, required)
            document = case.document
            case.verdict = {
                "case_id": case.case_id,
                "verdict": policy.verdict.value,
                "policy_version": cfg.policy_version,
                "blocking_reasons": [b.as_dict() for b in policy.blocking],
                "review_score": policy.review_score,
                "values": document.data.get("values", {}) if document is not None else {},
                "derived": document.data.get("derived", {}) if document is not None else {},
                "steps": {s: _summary(r) for s, r in results.items()},
                "module_results": {s: r.model_dump() for s, r in results.items() if isinstance(r, ModuleResult)},
                "attempts": {"document": case.document_attempts, "selfie": case.capture_attempts,
                             "liveness": case.liveness_attempts},
                "decided_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            case.portrait_b64 = None  # not needed after the verdict
            return case.verdict

    @staticmethod
    def _run(call, not_finished=UNAVAILABLE):
        """A module result, UNAVAILABLE if the module could not run (5xx),
        or `not_finished` if its session is not done / gone (4xx)."""
        try:
            return _public(call())
        except CaptureNotFound:
            return None
        except KycError as exc:
            return UNAVAILABLE if exc.http_status >= 500 else not_finished

    def view(self, case_id: str) -> dict:
        case = self.get(case_id)
        with case.lock:
            if case.verdict is not None:
                return case.verdict
            return {
                "case_id": case.case_id,
                "verdict": None,
                "document": case.document.model_dump() if case.document else None,
                "capture_session_id": case.capture_session_id,
                "liveness_session_id": case.liveness_session_id,
                "attempts": {"document": case.document_attempts, "selfie": case.capture_attempts,
                             "liveness": case.liveness_attempts},
                "expires_in_s": max(0, round(case.expires_at - time.time())),
            }


def _summary(result) -> dict:
    if result is None:
        return {"status": "not_run"}
    if result == UNAVAILABLE:
        return {"status": UNAVAILABLE}
    return {"status": "done", "decision": result.decision.value, "score": result.score,
            "reasons": [r.code for r in result.reasons if r.severity.value != "info"]}


def _decode_b64(payload: str) -> np.ndarray:
    try:
        return decode_image(base64.b64decode(payload))
    except (ValueError, TypeError) as exc:
        raise InvalidInput("Stored card portrait could not be decoded") from exc
