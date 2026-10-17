"""Hosted-test-service features: per-customer access codes and the
encrypted, time-limited case archive."""

from __future__ import annotations

import datetime as dt

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from kyc.api.app import create_app
from kyc.core.config import OrchestratorSettings, Settings, get_settings
from kyc.modules.orchestrator.archive import LocalArchive, _expired_days
from kyc.modules.orchestrator.router import get_service
from kyc.modules.orchestrator.service import CaseNotFound, CaseService

from .test_orchestrator import Fakes, full_case

KEY = Fernet.generate_key().decode()


# ------------------------------------------------------------------ access codes


@pytest.fixture
def gated(monkeypatch):
    monkeypatch.setenv("KYC_REQUIRE_ACCESS", "true")
    monkeypatch.setenv("KYC_ACCESS_CODES", '{"acme": "code-a", "globex": "code-b"}')
    get_settings.cache_clear()
    app = create_app()
    svc = CaseService(get_settings(), ocr=Fakes(), capture=Fakes(), antispoof=Fakes(), face_match=Fakes(),
                      liveness=Fakes())
    app.dependency_overrides[get_service] = lambda: svc
    yield TestClient(app)
    get_settings.cache_clear()


def test_v1_needs_a_valid_code(gated):
    assert gated.get("/v1/access/whoami").status_code == 401
    assert gated.get("/v1/access/whoami", headers={"X-Access-Code": "wrong"}).status_code == 401
    assert gated.get("/v1/access/whoami", headers={"X-Access-Code": "code-a"}).json() == {"customer": "acme"}
    assert gated.get("/health").status_code == 200  # health stays open


def test_a_customer_cannot_see_another_customers_case(gated):
    a, b = {"X-Access-Code": "code-a"}, {"X-Access-Code": "code-b"}
    cid = gated.post("/v1/cases", headers=a).json()["case_id"]
    assert gated.get(f"/v1/cases/{cid}", headers=a).status_code == 200
    assert gated.get(f"/v1/cases/{cid}", headers=b).status_code == 404  # not 403: no hint it exists
    assert gated.post(f"/v1/cases/{cid}/finalize", headers=b).status_code == 404


def test_service_level_tenant_check():
    svc = CaseService(Settings(), ocr=Fakes(), capture=Fakes(), antispoof=Fakes(), face_match=Fakes(), liveness=Fakes())
    case = svc.create("acme")
    with pytest.raises(CaseNotFound):
        svc.view(case.case_id, customer="globex")


# ------------------------------------------------------------------ archive


class Capture:
    jpeg = b"selfie-bytes"


class ArchiveFakes(Fakes):
    def get_capture(self, capture_id):
        return Capture()


def test_a_finalized_case_is_archived_encrypted(tmp_path):
    archive = LocalArchive(tmp_path, KEY, retention_days=30)
    fakes = ArchiveFakes()
    svc = CaseService(Settings(orchestrator=OrchestratorSettings()), ocr=fakes, capture=fakes, antispoof=fakes,
                      face_match=fakes, liveness=fakes, archive=archive)
    cid = full_case(svc)
    verdict = svc.finalize(cid)
    assert verdict["archived"] is True
    folder = next((tmp_path / "cases").iterdir()) / cid
    names = sorted(p.name for p in folder.iterdir())
    assert names == ["card.jpg.enc", "record.json.enc", "selfie.jpg.enc"]
    raw = (folder / "selfie.jpg.enc").read_bytes()
    assert b"selfie-bytes" not in raw  # encrypted at rest
    assert Fernet(KEY.encode()).decrypt(raw) == b"selfie-bytes"
    assert b"0012345679" not in (folder / "record.json.enc").read_bytes()


def test_retention_deletes_only_folders_older_than_the_limit(tmp_path):
    archive = LocalArchive(tmp_path, KEY, retention_days=30)
    for day in ("2026-08-01", "2026-09-20", "2026-10-15"):
        archive.save("c1", day, {"x": 1}, {})
    removed = archive.purge(now=dt.date(2026, 10, 17))
    assert removed == 1
    assert sorted(p.name for p in (tmp_path / "cases").iterdir()) == ["2026-09-20", "2026-10-15"]


def test_expired_days_ignores_non_date_folders():
    assert _expired_days(["2020-01-01", "notes", "2026-10-16"], 30, dt.date(2026, 10, 17)) == ["2020-01-01"]


def test_archive_failure_does_not_lose_the_verdict(tmp_path):
    class Broken:
        def save(self, *a):
            raise OSError("disk full")

        def purge(self, now=None):
            return 0

    fakes = ArchiveFakes()
    svc = CaseService(Settings(), ocr=fakes, capture=fakes, antispoof=fakes, face_match=fakes, liveness=fakes,
                      archive=Broken())
    verdict = svc.finalize(full_case(svc))
    assert verdict["verdict"] == "approved" and verdict["archived"] is False
