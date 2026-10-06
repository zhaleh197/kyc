from __future__ import annotations

import base64

import cv2
import pytest
from fastapi.testclient import TestClient

from kyc.api.app import app
from kyc.modules.ocr.pipeline import DocumentOcrPipeline
from kyc.modules.ocr.router import get_pipeline

from .conftest import FakeDetector, SequenceRecognizer, make_card, photograph


@pytest.fixture
def client(relaxed_settings, good_values):
    # One shared recogniser: the factory is called per field, so returning a
    # fresh instance each time would restart the scripted sequence.
    recognizer = SequenceRecognizer(good_values)
    pipeline = DocumentOcrPipeline(
        relaxed_settings,
        detector_factory=lambda profile: FakeDetector(),
        recognizer_factory=lambda backend_id: recognizer,
    )
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def card_jpeg() -> bytes:
    ok, buf = cv2.imencode(".jpg", photograph(make_card()))
    assert ok
    return buf.tobytes()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_profiles_lists_the_iranian_card(client):
    profiles = client.get("/v1/ocr/profiles").json()
    iran = next(p for p in profiles if p["id"] == "ir_national_card_front")
    assert iran["country"] == "IR"
    assert {f["key"] for f in iran["fields"]} >= {"national_id", "first_name", "birth_date"}


def test_document_upload_returns_a_module_result(client, card_jpeg):
    response = client.post(
        "/v1/ocr/document",
        files={"file": ("card.jpg", card_jpeg, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["module"] == "ocr"
    assert body["decision"] == "pass"
    assert body["data"]["values"]["national_id"] == "0012345679"


def test_document_base64_endpoint(client, card_jpeg):
    response = client.post(
        "/v1/ocr/document/base64",
        json={"image_base64": base64.b64encode(card_jpeg).decode()},
    )
    assert response.status_code == 200
    assert response.json()["data"]["values"]["first_name"] == "زهرا"


def test_data_url_prefix_is_accepted(client, card_jpeg):
    data_url = "data:image/jpeg;base64," + base64.b64encode(card_jpeg).decode()
    response = client.post("/v1/ocr/document/base64", json={"image_base64": data_url})
    assert response.status_code == 200


def test_unknown_profile_returns_404(client, card_jpeg):
    response = client.post(
        "/v1/ocr/document/base64",
        json={"image_base64": base64.b64encode(card_jpeg).decode(), "profile": "iq_national_card"},
    )
    assert response.status_code == 404
    assert response.json()["error"] == "profile_not_found"


def test_undecodable_payload_returns_400(client):
    response = client.post("/v1/ocr/document/base64", json={"image_base64": "bm90LWFuLWltYWdl"})
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_input"


def test_too_small_image_returns_422(client):
    ok, buf = cv2.imencode(".jpg", make_card()[:80, :120])
    assert ok
    response = client.post(
        "/v1/ocr/document/base64",
        json={"image_base64": base64.b64encode(buf.tobytes()).decode()},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "image_quality"


def test_missing_detector_model_returns_503(relaxed_settings, card_jpeg, tmp_path):
    """The real pipeline must refuse loudly when a model artifact is absent -
    a KYC service that silently stops reading a field is worse than one that
    fails the request."""
    import base64 as b64

    from kyc.modules.ocr.pipeline import DocumentOcrPipeline as RealPipeline

    # Point at an empty directory rather than the project's real models/ -
    # this test asserts what happens when the artifact is missing, and it
    # must not start passing or failing based on whatever is actually
    # installed there.
    settings = relaxed_settings.model_copy(update={"models_dir": tmp_path})
    app.dependency_overrides[get_pipeline] = lambda: RealPipeline(settings)
    try:
        response = TestClient(app).post(
            "/v1/ocr/document/base64",
            json={"image_base64": b64.b64encode(card_jpeg).decode()},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["error"] == "model_not_available"
