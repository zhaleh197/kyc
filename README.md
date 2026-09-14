# KYC

Identity verification services: document OCR, face quality, face matching,
liveness, anti-spoofing and AI forensics.

The system is a **modular monolith**: one FastAPI process, one router per
module, shared image decoding and shared model loading. Each module returns
the same envelope — a decision, a score and the *reasons* behind it — so an
orchestrator can aggregate them into one verdict and an operator can read the
audit trail during manual review.

Models are **trained on Kaggle GPUs and served on CPU** through ONNX Runtime.
Nothing at serving time imports torch or ultralytics, and nothing calls a
hosted inference API.

---

## Status

| Module | State | Location |
|---|---|---|
| **Document OCR** | built, tested | `kyc/modules/ocr/` |
| Face quality / detection | prototype, not migrated | `phase1_facedetect/` |
| Face matching | prototype, not migrated | `phase2_facematch/` |
| Anti-spoofing (passive) | prototype, not migrated | `phase3-antispoof/` |
| Liveness (active) | prototype, not migrated | `phase4-liveness/` |
| AI forensics | not started | — |

The `phase*/` directories are the original standalone prototypes. They still
run on their own; they are migrated into `kyc/modules/` one at a time.

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements/dev.txt
pip install -r requirements/ocr.txt                 # easyocr backend, pulls CPU torch
uvicorn kyc.api.app:app --reload
```

Interactive docs at <http://127.0.0.1:8000/docs>.

```bash
pytest
```

82 tests, no model weights or real documents required — the pipeline's
detector and recogniser are injectable and the suite substitutes fakes.

---

## Document OCR

```
photo ─► downscale ─► rectify card ─► quality gate ─► detect fields
      ─► crop ─► recognise ─► normalise ─► validate ─► ModuleResult
```

**Endpoints**

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/ocr/profiles` | Supported document types and their fields |
| `POST` | `/v1/ocr/document` | Multipart upload |
| `POST` | `/v1/ocr/document/base64` | JSON with a base64 image or data URL |
| `GET` | `/health` | Liveness probe |

```bash
curl -X POST http://127.0.0.1:8000/v1/ocr/document \
  -F file=@card.jpg -F return_portrait=true
```

```json
{
  "module": "ocr",
  "decision": "review",
  "score": 0.81,
  "reasons": [
    {"code": "field_low_confidence", "severity": "warn",
     "message_fa": "اطمینان پایین در خواندن «نام پدر»", "field": "father_name"}
  ],
  "data": {
    "values": {"national_id": "0012345679", "first_name": "زهرا", "birth_date": "1370/05/12"},
    "derived": {"birth_date_gregorian": "1991-08-03", "age": 35, "national_id_valid": true},
    "quality": {"sharpness": 412.7, "brightness": 168.3, "glare_ratio": 0.011}
  },
  "elapsed_ms": 734.2
}
```

`decision` is `pass` / `review` / `fail`. Errors fail outright; warnings route
to manual review. Every field keeps both its `raw` recognition and its
normalised `value`, because when a checksum fails an operator needs to see
what the model actually emitted to tell a misread from a forgery.

### What is Iran-specific, and where

Nothing in the pipeline mentions Iran. Country logic lives in one place:

- `profiles/iran.py` — field list, detector class order, validation rules
- `validators.py` — national-code check digit, Jalali calendar, expiry
- `normalize.py` — Persian/Arabic digit and letter folding

Adding the Iraqi national card means writing `profiles/iraq.py`, registering
it, and training its detector. No pipeline change.

Checks that run today:

- **National code check digit** — a failing checksum blocks auto-approval;
  it is either a misread or a forgery, and both mean the same thing here.
- **Jalali dates** — real calendar arithmetic including leap years, so
  `1399/12/30` is accepted and `1400/12/30` is rejected. Converted to
  Gregorian for downstream systems.
- **Two-digit years are rejected, not expanded.** Guessing `70` → `1370`
  would fabricate a value in an identity document.
- **Script folding** — Arabic yeh/kaf and Eastern-Arabic digits are folded to
  their Persian forms so encoding differences do not read as OCR errors.
  Alef-madda is deliberately *not* folded: آرش and ارش are different names.
- **Image quality gate** — a blurry or blown-out card is rejected before OCR
  runs. Confident nonsense on an identity document is the worst failure mode
  available.

---

## Models

`models/` is not in version control. With a Kaggle API token the whole training
round-trip runs from here:

```bash
python -m scripts.roboflow_pull --workspace <ws> --project <proj> --version 8
python -m scripts.kaggle_run upload-data --path .data/ir_card_yolo --confirm
python -m scripts.kaggle_run push --data-dataset <owner>/ir-card-yolo
python -m scripts.kaggle_run watch
python -m scripts.kaggle_run pull
python -m scripts.check_models
```

`check_models` reports what is present, its sha256, and whether the detector's
class count agrees with the profile it serves. `roboflow_pull` also remaps
Roboflow's class order into the profile's order, so the artifact that reaches
Kaggle is correct before training starts.

Credentials are read from disk, never from arguments: `~/.kaggle/kaggle.json`
and `ROBOFLOW_API_KEY` in `.env`. `upload-data` is the only command that sends
card images anywhere, and it refuses to act without `--confirm`.

The OCR module needs `ir_national_card_front_fields.onnx` to run. Without it,
requests return `503 model_not_available` naming the missing file — a service
that silently stops reading a field is worse than one that refuses.

Text recognition ships with two backends, and the defaults are mixed based on
measurement, not architecture:

- `easyocr` — pretrained, no training needed, ~1–2 s per card on CPU. Still
  the default for names: on this project's held-out eval set it beats the
  fine-tuned CRNN (57.1% vs 38.8% exact-match).
- `crnn_onnx` / `crnn_onnx_digits` — fine-tuned from EasyOCR's own pretrained
  Arabic-script weights (see `training/kaggle/README.md`) rather than trained
  from scratch. `crnn_onnx_digits` is the default for national id and dates:
  a decisive win there (68.4% vs 7.0% exact-match), because easyocr's own
  `allowlist` parameter does not actually restrict its output — confirmed
  directly against the library — while this recogniser's allowlist masks
  logits before decoding, a hard constraint.

Both defaults are overridable via `KYC_OCR_TEXT_BACKEND` /
`KYC_OCR_DIGIT_BACKEND`. See `training/kaggle/README.md` for the full training
loop and `models/manifest.yaml` for the measurements behind these numbers.

---

## Configuration

Every tunable number is in `kyc/core/config.py` and overridable by environment
variable. Nothing is hard-coded in the module code. See `.env.example`.

CORS defaults to same-origin. Set `KYC_CORS_ORIGINS` to real origins in
production.

---

## Layout

```
kyc/
  core/            contract, config, errors, image primitives
  api/app.py       FastAPI app; one router per module
  modules/ocr/
    pipeline.py    orchestration
    rectify.py     card localisation, perspective, quality gate
    detector.py    ONNX YOLO runner (CPU)
    recognizers/   easyocr and CRNN-ONNX backends
    normalize.py   Persian/Arabic text folding
    validators.py  national code, Jalali calendar
    profiles/      per-country field definitions and rules
training/kaggle/   GPU training scripts that export ONNX
scripts/           operational helpers
tests/
phase*/            original prototypes, pending migration
```

---

## Security notes

- A **Roboflow API key is present in the git history** of
  `phase6-ocr/*.ipynb`. It must be revoked from the Roboflow dashboard —
  deleting the file does not remove it from history. The new OCR module needs
  no such key.
- The prototypes ship `allow_origins=["*"]` and no authentication. The new API
  defaults to same-origin; authentication is still to be added before any
  deployment.
- `phase1_facedetect/requirements.txt` is a full `pip freeze` of an unrelated
  environment and cannot be installed. Use `requirements/`.
