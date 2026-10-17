"""Orchestrator (spec 07): the verdict policy and the case flow, with fake
modules - every module is reached through its public pipeline methods only."""

from __future__ import annotations

import base64
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from kyc.api.app import app
from kyc.core.config import OrchestratorSettings, Settings
from kyc.core.errors import ModelNotAvailable
from kyc.core.schemas import Decision, ModuleResult, Reason
from kyc.modules.orchestrator.policy import STEPS, UNAVAILABLE, Verdict, decide
from kyc.modules.orchestrator.router import get_service
from kyc.modules.orchestrator.service import CaseConflict, CaseService


def result(module, decision=Decision.PASS, *reasons, **data) -> ModuleResult:
    return ModuleResult(module=module, decision=decision, score=0.9 if decision is Decision.PASS else 0.3,
                        reasons=list(reasons), data=data)


def err(code):
    return Reason.error(code, code, code)


def warn(code):
    return Reason.warn(code, code, code)


ALL_PASS = {s: result(s) for s in STEPS}
CFG = OrchestratorSettings()


# ------------------------------------------------------------------ policy


@pytest.mark.parametrize(
    ("change", "verdict", "code"),
    [
        ({}, Verdict.APPROVED, None),
        ({"antispoof": result("antispoof", Decision.FAIL, err("spoof_detected"))}, Verdict.REJECTED, "spoof_detected"),
        ({"face_match": result("face_match", Decision.FAIL, err("faces_do_not_match"))}, Verdict.REJECTED,
         "faces_do_not_match"),
        # a checksum failure may be a misread: a human looks, it is not rejected
        ({"document": result("ocr", Decision.FAIL, err("national_id_checksum"))}, Verdict.MANUAL_REVIEW,
         "national_id_checksum"),
        ({"face_match": result("face_match", Decision.REVIEW, warn("faces_match_uncertain"))}, Verdict.MANUAL_REVIEW,
         "faces_match_uncertain"),
        ({"antispoof": UNAVAILABLE}, Verdict.MANUAL_REVIEW, "module_unavailable"),
        ({"liveness": None}, Verdict.MANUAL_REVIEW, "step_missing"),
        ({"liveness": result("liveness", Decision.REVIEW)}, Verdict.MANUAL_REVIEW, "liveness_review"),
    ],
)
def test_policy_table(change, verdict, code):
    out = decide({**ALL_PASS, **change}, CFG, STEPS)
    assert out.verdict is verdict
    if code:
        assert code in {b.code for b in out.blocking}


def test_strong_scores_never_compensate_for_a_failure():
    """Rule-based, not a weighted sum: four perfect steps and one rejected."""
    results = {**{s: result(s) for s in STEPS}, "liveness": result("liveness", Decision.FAIL, err("face_changed"))}
    for r in results.values():
        r.score = 1.0 if r.decision is Decision.PASS else 0.0
    assert decide(results, CFG, STEPS).verdict is Verdict.REJECTED


def test_a_rejecting_warning_is_a_business_rule():
    results = {**ALL_PASS, "document": result("ocr", Decision.REVIEW, warn("under_age"))}
    assert decide(results, CFG, STEPS).verdict is Verdict.MANUAL_REVIEW
    strict = OrchestratorSettings(reject_warning_codes=["under_age"])
    assert decide(results, strict, STEPS).verdict is Verdict.REJECTED


def test_liveness_can_be_optional():
    results = {**ALL_PASS, "liveness": None}
    assert decide(results, CFG, tuple(s for s in STEPS if s != "liveness")).verdict is Verdict.APPROVED


# ------------------------------------------------------------------ case flow

PORTRAIT = base64.b64encode(cv2.imencode(".jpg", np.full((120, 100, 3), 128, np.uint8))[1].tobytes()).decode()


class Fakes:
    def __init__(self, *, spoof=None, match=None, live=None, selfie_ready=True):
        self.spoof = spoof or result("antispoof")
        self.match = match or result("face_match")
        self.live = live or result("liveness")
        self.selfie_ready = selfie_ready
        self.calls: list[str] = []

    # 01
    def run_bytes(self, data, profile=None, *, return_portrait=False):
        self.calls.append("ocr")
        return result("ocr", values={"national_id": "0012345679"}, derived={"national_id_valid": True},
                      portrait_base64=PORTRAIT if return_portrait else None)

    # 02
    def start(self, capture_id=None, check_modes=None):
        if capture_id is not None:  # liveness.start(capture_id)
            self.calls.append(f"liveness.start:{capture_id}")
            return SimpleNamespace(session_id="live-1", tasks=[SimpleNamespace(name="BLINK")])
        self.calls.append("capture.start")
        return SimpleNamespace(session_id="cap-session")

    def result(self, session_id):
        if session_id == "live-1":
            return self.live
        if not self.selfie_ready:
            from kyc.modules.face_capture.pipeline import SessionClosed
            raise SessionClosed("not finished")
        return result("face_quality", capture_id="cap-1", selfie_base64="xx")

    # 04 / 03
    def verify(self, *args):
        if len(args) == 1:
            self.calls.append(f"antispoof:{args[0]}")
            if self.spoof == UNAVAILABLE:
                raise ModelNotAvailable("model missing")
            return self.spoof
        self.calls.append(f"face_match:{args[1]}")
        return self.match


def service(fakes, **cfg):
    return CaseService(Settings(orchestrator=OrchestratorSettings(**cfg)), ocr=fakes, capture=fakes,
                       antispoof=fakes, face_match=fakes, liveness=fakes)


def full_case(svc):
    case = svc.create()
    svc.submit_document(case.case_id, b"jpeg")
    svc.start_capture(case.case_id)
    svc.start_liveness(case.case_id)
    return case.case_id


def test_a_complete_good_case_is_approved_and_links_the_same_selfie():
    fakes = Fakes()
    svc = service(fakes)
    verdict = svc.finalize(full_case(svc))
    assert verdict["verdict"] == "approved"
    assert verdict["values"]["national_id"] == "0012345679"
    # liveness, anti-spoof and face match all used the selfie this case captured
    assert {"liveness.start:cap-1", "antispoof:cap-1", "face_match:cap-1"} <= set(fakes.calls)
    assert "portrait_base64" not in str(verdict) and "selfie_base64" not in str(verdict)


def test_a_spoof_rejects_the_case():
    svc = service(Fakes(spoof=result("antispoof", Decision.FAIL, err("spoof_detected"))))
    verdict = svc.finalize(full_case(svc))
    assert verdict["verdict"] == "rejected"
    assert verdict["blocking_reasons"][0]["code"] == "spoof_detected"


def test_an_unavailable_module_never_approves():
    svc = service(Fakes(spoof=UNAVAILABLE))
    assert svc.finalize(full_case(svc))["verdict"] == "manual_review"


def test_liveness_needs_the_selfie_first():
    svc = service(Fakes(selfie_ready=False))
    case = svc.create()
    svc.start_capture(case.case_id)
    with pytest.raises(CaseConflict) as exc:
        svc.start_liveness(case.case_id)
    assert exc.value.code == "selfie_missing"


def test_finalize_refuses_an_incomplete_case():
    svc = service(Fakes())
    case = svc.create()
    svc.submit_document(case.case_id, b"jpeg")
    with pytest.raises(CaseConflict) as exc:
        svc.finalize(case.case_id)
    assert exc.value.details["missing"] == ["selfie", "liveness"]


def test_attempts_are_limited():
    svc = service(Fakes(), max_document_attempts=2)
    case = svc.create()
    svc.submit_document(case.case_id, b"a")
    svc.submit_document(case.case_id, b"b")
    with pytest.raises(CaseConflict) as exc:
        svc.submit_document(case.case_id, b"c")
    assert exc.value.code == "attempts_exhausted"


def test_a_finalized_case_is_frozen_and_finalize_is_idempotent():
    svc = service(Fakes())
    cid = full_case(svc)
    first = svc.finalize(cid)
    assert svc.finalize(cid) is first
    with pytest.raises(CaseConflict):
        svc.submit_document(cid, b"later")


def test_api_case_flow():
    fakes = Fakes()
    svc = service(fakes)
    app.dependency_overrides[get_service] = lambda: svc
    try:
        client = TestClient(app)
        cid = client.post("/v1/cases").json()["case_id"]
        assert client.post(f"/v1/cases/{cid}/finalize").status_code == 409  # nothing done yet
        doc = client.post(f"/v1/cases/{cid}/document", files={"file": ("c.jpg", b"jpeg", "image/jpeg")}).json()
        assert doc["data"]["portrait_base64"]  # shown back to the caller who sent the card
        assert client.post(f"/v1/cases/{cid}/capture").json()["capture_session_id"] == "cap-session"
        assert client.post(f"/v1/cases/{cid}/liveness").json()["liveness_session_id"] == "live-1"
        assert client.post(f"/v1/cases/{cid}/face-match").json()["module"] == "face_match"
        verdict = client.post(f"/v1/cases/{cid}/finalize").json()
        assert verdict["verdict"] == "approved" and verdict["policy_version"]
        assert client.get(f"/v1/cases/{cid}").json()["verdict"] == "approved"
        assert client.get("/v1/cases/nope").status_code == 404
    finally:
        app.dependency_overrides.clear()



def test_face_match_now_is_reused_at_finalize_and_redone_for_a_new_card():
    fakes = Fakes()
    svc = service(fakes)
    cid = full_case(svc)
    svc.match_now(cid)
    svc.finalize(cid)
    assert fakes.calls.count("face_match:cap-1") == 1  # reused, not run twice

    fakes2 = Fakes()
    svc2 = service(fakes2)
    cid2 = full_case(svc2)
    svc2.match_now(cid2)
    svc2.submit_document(cid2, b"another card")  # a new card invalidates the match
    svc2.finalize(cid2)
    assert fakes2.calls.count("face_match:cap-1") == 2


def test_face_match_now_needs_card_and_selfie():
    svc = service(Fakes())
    case = svc.create()
    with pytest.raises(CaseConflict) as exc:
        svc.match_now(case.case_id)
    assert exc.value.code == "card_missing"
