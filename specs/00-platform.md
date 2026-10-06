# 00 — Platform: what every module shares

Status: describes code that exists in `kyc/core/` and `kyc/api/`. Every module
spec (01–07) assumes this document and only states deviations from it.

## 1. Architecture

Modular monolith: one FastAPI process (`kyc/api/app.py`), one router per
module under `kyc/modules/<name>/`, shared image decoding and model loading.
Splitting a router into its own service later is a deployment change, not a
rewrite — so no module may reach into another module's internals. Cross-module
data (e.g. the OCR portrait feeding face match) goes through public schemas or
through the orchestrator (07).

Layout per module, mirroring `kyc/modules/ocr/`:

```
kyc/modules/<name>/
  router.py      HTTP surface, /v1/<name>/...
  pipeline.py    orchestration; detector/model factories injectable
  schemas.py     request + `data` payload models
  ...            module-specific stages
```

## 2. The result contract — `ModuleResult` (`kyc/core/schemas.py`)

Every module endpoint that makes a judgement returns:

| Field | Type | Rule |
|---|---|---|
| `module` | str | stable module id (`ocr`, `face_quality`, `face_match`, `antispoof`, `liveness`, `forensics`) |
| `version` | str | module version; bump on any behaviour change |
| `decision` | `pass` \| `review` \| `fail` | see §3 |
| `score` | float 0–1 | "confidence this check passed"; a `fail` is capped ≤ 0.5 so callers can rank by score |
| `reasons` | list[`Reason`] | every finding, not just the first |
| `data` | dict | module-specific payload, defined in the module spec |
| `elapsed_ms` | float | wall time inside the module |

`Reason`: `code` (stable, snake_case, safe to branch on), `severity`
(`info`/`warn`/`error`), `message_en`, `message_fa`, optional `field`.
Persian messages are mandatory — the client does not keep a translation table.

**A module never returns a bare boolean.** The reasons are the audit trail an
operator reads during manual review.

## 3. Decision semantics (all modules)

- any `error` reason → `fail`
- else any `warn` reason → `review` (unless the module's config disables it)
- else → `pass`

A module that cannot judge (model missing, backend down) does **not** return
`fail` — it raises a typed error (§4). "We could not check" and "the check
failed" must never look the same to the orchestrator.

## 4. Errors (`kyc/core/errors.py`)

| Class | code | HTTP | Meaning / client action |
|---|---|---|---|
| `InvalidInput` | `invalid_input` | 400 | fix the request |
| `ImageQualityError` | `image_quality` | 422 | take another photo |
| `ModelNotAvailable` | `model_not_available` | 503 | ours; message names the missing file |
| `BackendNotInstalled` | `backend_not_installed` | 503 | ours; optional dependency missing |
| `ProfileNotFound` | `profile_not_found` | 404 | unknown document profile |
| (router) | `payload_too_large` | 413 | upload over `KYC_MAX_UPLOAD_BYTES` |

New modules add subclasses here rather than raising library exceptions. A
missing model must refuse (503), never silently degrade.

## 5. Configuration (`kyc/core/config.py`)

- Every tunable number lives in config, overridable by env var. Module code
  hard-codes no thresholds.
- Prefix `KYC_` for app-wide, `KYC_<MODULE>_` per module (`KYC_OCR_`,
  `KYC_FACE_MATCH_`, ...), one `BaseSettings` subclass per module nested in
  `Settings`.
- Every field carries a `description`; a threshold that was changed after
  measurement carries a comment saying what was measured (see
  `OcrSettings.min_sharpness`).
- `.env.example` must match the defaults in `config.py` (see 01 §9 — it
  currently does not).

## 6. Models and serving

- Trained on Kaggle GPUs, **served on CPU via onnxruntime**. Nothing at
  serving time imports torch or ultralytics or calls a hosted inference API.
  Exceptions must be argued in the module spec (liveness/MediaPipe is the
  likely one — see 05).
- Weights live in `models/` (not in git). Every artifact gets a
  `models/manifest.yaml` entry: dataset snapshot, command, commit, sha256,
  metrics with how they were measured, known limitations, **license**.
- `scripts/check_models.py` is extended per module to verify presence,
  sha256 and output shape.
- Models load lazily once per process and are cached.

## 7. Testing

- Pipelines take injectable model factories; the test suite substitutes fakes
  and needs no weights and no real identity documents.
- Real-data evaluation (accuracy numbers) is a separate, manual step whose
  result goes into the manifest — not into CI.
- No real person's document or face is committed to the repo. (The `phase*/`
  prototypes currently contain test photos; see §8.)

## 8. Security and privacy (applies to every module)

| Requirement | State |
|---|---|
| CORS same-origin by default; `KYC_CORS_ORIGINS` explicit list | done |
| Upload size limit, decompression-bomb guard | done (`MAX_DECODE_PIXELS`) |
| Authentication on every `/v1` endpoint | **not done — blocker for any deployment** |
| Rate limiting per client / per session | not done |
| Never log images, base64 payloads, or extracted values (names, national id) | to verify in every module |
| Images held in memory only; nothing written to disk by the API | to verify per module |
| Revoke the Roboflow key leaked in `phase6-ocr/*.ipynb` history | open (README) |
| Remove real-person photos from `phase*/` test folders, or confirm consent | open |
| Hard-coded DB password in `phase1_facedetect/database.py` | open; rotate if ever used outside a laptop |

## 9. Versioning

Endpoints are `/v1/...` from day one. A field rename or semantic change in
`data` is a `/v2`, not an edit. `ModuleResult.version` changes on any
behaviour change (threshold, model swap) so stored results can be traced to
the logic that produced them.
