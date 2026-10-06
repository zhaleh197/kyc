# 06 — AI forensics (document tampering, deepfakes, injection)

Status: **up-front spec, not started.** The only prior code is a heuristic
`AIForensicsChecker` in `phase1_facedetect/claude_api/main.py` (ELA + noise +
a GAN-frequency heuristic, thresholds 0.55 / 0.60, unmeasured). It is not a
baseline worth migrating.

## 1. Purpose

Detect evidence that an image was **manipulated or synthesised** rather than
photographed: an edited ID card (swapped portrait, altered digits), an
AI-generated or face-swapped selfie, or a stream injected through a virtual
camera. 04 asks "is this a photo of a screen?"; 06 asks "was this image made
or altered by software?".

## 2. Scope — split into two sub-checks with separate endpoints

**06a — Document forensics** (card image from OCR):

- portrait region inconsistent with the card (pasted photo): compression /
  noise inconsistency between `img` box and surrounding card
- digit tampering: per-field noise/JPEG-grid inconsistency on the `idnumber`
  and date boxes (the OCR boxes are free — reuse them)
- cross-field consistency that OCR already exposes: national-id checksum,
  birth < expiry (already in 01; forensics consumes, does not recompute)
- screen recapture of a card (moiré) — overlaps 04's idea, applied to the card
- EXIF/metadata: editing software tags, missing camera fields (weak signal;
  `info` only — metadata is trivially stripped or forged)

**06b — Selfie forensics**:

- face-swap / fully synthetic face classifier
- injection signals: frame timing (from 05), resolution/codec fingerprints of
  known virtual-camera software, client attestation if the app can provide it

Non-goals: judging whether the card *number* is real (needs a registry lookup
— a separate integration, out of scope); physical card security features
(holograms, UV) — impossible from one RGB photo.

## 3. Interfaces

`POST /v1/forensics/document` — card image + optional OCR field boxes.
`POST /v1/forensics/selfie` — selfie image or liveness session id.

`data`: `signals[{name, score, region?, evidence?}]` — one entry per detector,
so the reviewer sees *which* signal fired and where (a heatmap region for
ELA-like signals).

### Reason codes (proposed)

`portrait_region_inconsistent`, `field_region_inconsistent` (with `field`),
`recaptured_from_screen`, `editing_software_metadata` (info),
`synthetic_face_suspected`, `face_swap_suspected`,
`virtual_camera_suspected`.

## 4. Decision rules

Until a signal has measured precision on this project's data, it may only
emit **warn** (→ `review`), never error. Forensic heuristics are notorious for
false positives on ordinary phone JPEGs (recompressed by messaging apps,
beautified by camera software); a hard fail on an unmeasured heuristic rejects
real customers for their phone's image pipeline.

## 5. Models — none chosen yet

Candidates to evaluate, each needing a license check and a manifest entry:

- document: noise-residual / JPEG-ghost features + a small classifier trained
  on synthetically tampered cards (we can generate those: paste portraits,
  edit digits on the Roboflow card set — the labels already give the regions)
- selfie: a public deepfake detector (EfficientNet-B4 on FaceForensics++ is
  the common baseline; check dataset license — FF++ is research-only) or a
  commercially licensed alternative

## 6. Configuration — `KYC_FORENSICS_*`

Per signal: `enabled`, `warn_above`, later `error_above` (only once measured).

## 7. Acceptance criteria

| # | Criterion | Status |
|---|---|---|
| F1 | False-positive rate on genuine card photos (incl. ones forwarded via Telegram/WhatsApp) ≤ 2% per signal | proposed |
| F2 | Detects ≥ 80% of synthetically pasted portraits | proposed |
| F3 | Detects ≥ 80% of single-digit edits on `idnumber` | proposed |
| F4 | Every signal can be disabled independently via config | required |
| F5 | No signal emits `error` without a manifest entry showing its measured precision | required |

## 8. Assumptions to verify before building

1. **How are images delivered?** If the client app captures directly, card
   images are first-generation JPEGs and forensic signals are meaningful. If
   users can upload from gallery, most images are recompressed and ELA-style
   signals are near useless. This single fact decides half the module.
2. Synthetic tampering generated from the Roboflow set resembles real
   tampering well enough to train on. Get even 10 real forged samples from
   the business to sanity-check.
3. The OCR detector boxes (unrectified frame coordinates) are accurate enough
   to localise per-field forensic analysis (name/lastname recall is 0.84 —
   see 01 §7).
4. Can the client provide device attestation (Play Integrity / App Attest)?
   If yes, injection detection mostly moves there.

## 9. Migration

Nothing to migrate. Do not port `claude_api`'s `AIForensicsChecker`: its
weights and thresholds are invented and its `deepfake_probability` is not a
probability.

## 10. Test plan

Unit: each signal on synthetic pairs (clean vs tampered region). Eval:
FPR on a genuine set spanning capture paths; TPR on a synthetic tamper set
per tamper type; both in the manifest.

## 11. Open questions

- Is registry verification (national id → name/birth date via a government or
  bank API) available? It beats every pixel-level signal for document fraud.
- Build 06 at all before the other modules ship? Recommendation: 06a first
  (cheap, uses OCR boxes, synthetic data is easy); 06b after 05, since
  liveness + anti-spoof already cover the most common selfie attacks.
