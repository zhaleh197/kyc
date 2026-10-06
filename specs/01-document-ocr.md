# 01 — Document OCR (retrospective)

Status: **built and tested** on branch `ocr-module`. This spec is written after
the fact: it records what the module does, why, what was measured, and what is
still open. Sources: `kyc/modules/ocr/`, `models/manifest.yaml`,
`training/kaggle/README.md`, commits `35e1351`, `6df8b78`, `d0d6ca6`.
Last updated 2026-10-06.

## 1. Purpose

Read the front of an Iranian national ID card (کارت ملی هوشمند) from a phone
photo and return every field with its raw recognition, a normalised value,
validation findings, and — on request — the cropped portrait for face match.

## 2. Scope

In scope: profile `ir_national_card_front`; fields national id, first name,
last name, father name, birth date, expiry date; portrait crop.

Not in scope (today): the back of the card, the old paper card, birth
certificate (شناسنامه), passport/MRZ, Iraqi card (the profile system is built
for it — `profiles/iraq.py` + a detector, no pipeline change). Authenticity of
the card is **not** judged here — that is 06-ai-forensics.

## 3. Pipeline

```
photo ─► ensure_min_size ─► downscale (max side) ─► [rectify, off] ─► quality gate
      ─► field detector (YOLOv8n ONNX) ─► best box per class ─► crop + pad + upscale
      ─► recogniser (per field kind) ─► normalise ─► profile validator ─► enrich
      ─► decision ─► ModuleResult
```

- **Quality gate** — sharpness (Laplacian variance), brightness, glare ratio.
  `card_blurry` / `card_too_dark` / `card_too_bright` are fatal: the module
  returns `fail` without reading, because confident nonsense on an identity
  document is the worst failure available.
- **Rectification is off** (`KYC_OCR_RECTIFY=false`). The classical contour
  rectifier found a card outline in only 12% of the training images (cards
  fill ~45% of frame, no clean border). The detector was therefore trained on
  unrectified frames and serving must match. Boxes are in frame coordinates.
- **Two recognisers, chosen per field kind** (by measurement, §7):
  text → `easyocr`; digits/dates → `crnn_onnx_digits`.
- **Allowlist is a hard constraint only in the CRNN**: `_mask_to_allowlist`
  sets banned logits to −inf before argmax. EasyOCR's `allowlist` does *not*
  restrict output (confirmed against the library).
- **Join order**: when EasyOCR splits a field into several boxes, free text is
  joined right-to-left, digits/dates left-to-right (numerals are LTR inside
  RTL text). Signal used: whether an allowlist is set.

## 4. Interfaces

| Method | Path | |
|---|---|---|
| GET | `/v1/ocr/profiles` | profiles and their fields |
| POST | `/v1/ocr/document` | multipart: `file`, `profile?`, `return_portrait?`, `return_card?` |
| POST | `/v1/ocr/document/base64` | JSON: `image_base64` (raw or data URL) + same options |

`data` = `DocumentOcrData`: `profile`, `country`, `fields{key: FieldResult}`,
`values{key: value}`, `derived` (`birth_date_gregorian`, `age`,
`expiry_date_gregorian`, `expired`, `national_id_valid`), `missing_fields`,
`quality{sharpness, brightness, glare_ratio}`, `portrait_base64?`,
`card_base64?`.

`FieldResult` keeps `raw` *and* `value`: when a checksum fails, the operator
needs what the model emitted to tell a misread from a forgery.

### Reason codes

| code | sev | source |
|---|---|---|
| `card_blurry`, `card_too_dark`, `card_too_bright` | error (fatal) | quality gate |
| glare finding | warn | quality gate |
| `field_not_located` | error | detector found no box for a required field |
| `field_low_confidence` | warn | recognition conf < `min_field_confidence` |
| `field_length_mismatch` | warn | e.g. national id ≠ 10 chars |
| `national_id_missing` / `_length` / `_checksum` | error | validator |
| `birth_date_missing` / `_invalid` / `_implausible` | error | validator |
| `under_age` | warn | age < 18 — policy, left to the orchestrator |
| `expiry_missing`, `expiry_invalid` | warn | validator |
| `card_expired`, `date_order` | error | validator |
| `{first,last,father}_name_missing` | error | validator |
| `{first,last,father}_name_suspicious` | warn | length < 2 |

Score = mean recognition confidence over **expected** fields (empty fields
count as 0); capped at 0.5 on `fail`.

## 5. Iran-specific rules (all in `profiles/iran.py`, `validators.py`, `normalize.py`)

- National code check digit (weights 10..2, mod 11); repdigits rejected;
  leading zeros significant.
- Jalali dates with real leap-year arithmetic (`1399/12/30` valid,
  `1400/12/30` invalid); years 1250–1500; converted to ISO Gregorian.
- Two-digit years are **rejected, not expanded** — guessing would fabricate
  identity data.
- Script folding: Arabic yeh/kaf/teh-marbuta etc. → Persian; Eastern-Arabic
  and Persian digits → ASCII; harakat, tatweel and bidi controls stripped;
  **ZWNJ kept; alef-madda not folded** (آرش ≠ ارش).
- Detector class order is fixed: `birthday, exp, fathername, idcard,
  idnumber, img, lastname, name`. Training refuses to export on mismatch; a
  reordered list would silently put the father's name in the national id.

## 6. Models

| Artifact | Role | Base | Data |
|---|---|---|---|
| `ir_national_card_front_fields.onnx` | field detector (required) | yolov8n | Roboflow `westco/idcardsegmentation-gz97d` v8: 423 train / 27 val / 19 test, unrectified |
| `crnn_fa_digits.onnx` + charset | digits/dates (default) | EasyOCR `arabic.pth`, fine-tuned | 542 hand-corrected crops |
| `crnn_fa_text.onnx` + charset | names (not default) | same | 900 hand-corrected crops |

Both CRNNs keep EasyOCR's 184-symbol charset (`--charset` is ignored with
`--finetune-from`); the digit-only guarantee comes from the serving-time mask.
Input height is read from each ONNX graph (64 px for EasyOCR-derived models).

## 7. Measured results (source: `models/manifest.yaml`)

Detector, recall@IoU0.5 on 19 held-out test images: 1.00 for all classes
except `name` and `lastname` 0.84 (crowded label column).

Recognition, held-out `test_` split, via the real inference path:

| Field kind | Backend | Exact match | Norm. edit distance |
|---|---|---|---|
| digits/dates (57 crops) | **crnn_onnx_digits** | **0.684** | 0.123 |
| | easyocr | 0.070 | 0.465 |
| names (49 crops) | **easyocr** | **0.571** | 0.350 |
| | crnn_onnx | 0.388 (normalised) | 0.344 |

These eval sets are small (49 / 57 crops, 19 images). Treat differences under
~10 points as noise.

## 8. Acceptance criteria

| # | Criterion | Status |
|---|---|---|
| A1 | No field value is ever returned without its `raw` | met (schema) |
| A2 | Invalid national id checksum never yields `pass` | met (test_validators) |
| A3 | Missing detector → 503 `model_not_available` naming the file | met |
| A4 | Blurry/dark/blown-out card → `fail` before any recognition | met |
| A5 | Suite runs with no weights and no real documents | met |
| A6 | National id exact-match ≥ 0.95 on held-out set | **proposed; currently 0.684 for digits overall — not met** |
| A7 | Names exact-match ≥ 0.85 (normalised) | **proposed; currently 0.571 — not met** |
| A8 | End-to-end field-level eval on whole cards (not crops) | **TBD — never measured** |
| A9 | p95 latency on target CPU | **TBD — README says ~1–2 s with easyocr, not measured systematically** |

Until A6/A7 are met, the module is fit for **assisted** review (operator sees
fields pre-filled), not for auto-approval on OCR alone. The checksum makes
national-id misreads self-detecting, which is why this is acceptable.

## 9. Known issues and follow-ups

1. ~~`.env.example` disagreed with `config.py`~~ — **fixed 2026-10-06**:
   `MIN_SHARPNESS=12`, `DIGIT_BACKEND=crnn_onnx_digits`, `RECTIFY=false` now
   explicit, each with the measurement behind it.
2. ~~`training/kaggle/README.md` §0.7 and §2 out of date~~ — **fixed
   2026-10-06**: §0.7 now documents the unrectified state and when to switch;
   §2 documents fine-tuning from `arabic.pth`, the bugs it hit, and the
   measured backend choice.
3. **Sharpness is measured on the whole frame**, mostly background. Measure on
   the detected `idcard` box instead, then recalibrate `min_sharpness`.
4. **`commit: TODO`** on all three manifest entries.
5. **Compound names lose their space** in the CRNN ("امیر حسین" →
   "امیرحسین"). Check label consistency before collecting more data.
6. `name`/`lastname` detector recall 0.84 — the main detector weakness.
7. Detector trained on one Roboflow dataset of 423 images; no evidence yet on
   other phones, lighting, or worn cards.
8. `/health` hard-codes `"modules": ["ocr"]`; should be derived from the
   registered routers once more modules exist.

## 10. Lessons carried into the other specs

- Write the train/serve geometry down first (rectify on/off) — and keep the
  docs in sync when it flips.
- Verify library claims directly (EasyOCR `allowlist`, tensor layout) at the
  *real* batch size before trusting them.
- Choose defaults by held-out measurement, per field kind, not by
  architecture preference.
- Calibrate thresholds on the geometry actually served.
- Select checkpoints by a metric that moves early (edit distance, not exact
  match).
- On Windows, `num_workers=0`.
- Data-provider APIs can lie (Roboflow listed an export whose objects 404'd);
  have a manual fallback path.
