# 02 — Face capture & quality (takes the selfie)

Status: **up-front spec**. Prototype: `phase1_facedetect/final.py` (Flask) +
`templates/final.html`. A single-image variant exists in
`phase1_facedetect/claude_api/main.py` (`/kyc/face-quality`).

**Implementation (branch `face-capture`)**: `kyc/modules/face_capture/` +
shared runners in `kyc/core/face/`. 2026-10-06: sessions, capture_id, core
checks. 2026-10-07: still-photo step, mouth, gaze, occlusion, background. See
§12 for what was measured while building it.

## 1. Purpose

**This module takes the selfie.** It runs a guided capture loop on the user's
camera: every frame is checked, the user is told what to fix ("move closer",
"open your eyes", "remove your mask"), and the first frame that passes every
check becomes *the* selfie used by face match (03) and anti-spoof (04).

What the prototype already does, and what this spec keeps: the browser opens
the webcam (`getUserMedia`), sends one frame per second to
`/process_imgSetting`, shows the returned warnings, and when the server answers
`"Face OK!"` stops the camera and keeps that frame (`lastValidFrame`), then
saves it (`/save_to_db2`).

It answers "is this a usable selfie, and here it is" — **not** "is this person
real?" (04/05) or "is this the right person?" (03).

## 2. The one thing to change from the prototype: the server keeps the selfie

In the prototype the *client* decides which frame is the selfie and uploads it
again. The server never checks that the uploaded selfie is the frame it
approved — and the page also offers "upload from file". So anyone can submit
any photo as the selfie.

Target: the server holds the capture session. When a frame passes, the
**server** stores that exact frame and returns a `capture_id`. Downstream
modules (03, 04, 07) take the `capture_id`, never a client-uploaded selfie.
**Decision (2026-10-06): there is no upload path for the selfie at all** —
no upload button in the UI and no endpoint that accepts a selfie file. The only
way a selfie enters the system is a frame sent into a capture session from the
camera.

### Video first, then a still photo — both from the same camera (decided 2026-10-07)

The user does not choose between "photo" and "video": offering a choice lets
an attacker pick the weaker path. One session does both, in order:

1. **Video** — frames stream in, each gets hints; `frames_required_ok`
   consecutive passing frames qualify the stream (`READY_FOR_STILL`).
2. **Still** — the client then takes a full-resolution photo from the same
   camera (`ImageCapture.takePhoto()` where available, otherwise the largest
   frame the stream offers) and sends it to the same session. It must pass
   the same checks **and** show the same person as the stream (ArcFace
   cosine ≥ `min_still_similarity`). That still is the selfie.
3. **Liveness** (05) continues on the same stream.

Why both: video gives live guidance and is what liveness/anti-spoof need; a
still gives face match a sharper, higher-resolution selfie than a compressed
720p video frame. A rejected still sends the user back to step 1.
`require_still=false` keeps the best video frame instead, for clients that
can't take stills.

Removing the button is not the protection by itself: the API can be called
directly (curl, a script) without the page. What enforces it is the server
side — frames are accepted only inside an open capture session, and 03/04/07
accept only a `capture_id`. The remaining route, feeding a photo or video
through a *virtual camera*, is handled by liveness (05) and injection
detection (06b).

## 3. Scope — the checks, and what each tells the user

All measured on the **face**, not the whole frame (OCR's whole-frame
sharpness is the mistake not to repeat, 01 §9.3). The team's requirement list
(2026-10-07) mapped to checks:

| Requirement | Check | How | Blocks capture? |
|---|---|---|---|
| Blurry / too dark / too bright | `face_blurry`, `face_too_dark`, `face_too_bright` | Laplacian variance and mean luma on the face crop resized to 256² | yes |
| Too far / too close | `face_too_small`, `face_too_large`, `face_off_center`, `face_cut_off` | face box vs frame | yes |
| Eyes open | `eyes_closed` | eye aspect ratio from 106 landmarks | yes |
| Mouth closed | `mouth_open` | inner-lip gap / mouth width from 106 landmarks | yes |
| Nothing covering the face | `face_occluded` | the prototype's MobileNetV3, exported to ONNX — **confident cases only**, see §12 | yes |
| Looking straight at the camera | `gaze_off_center` | iris located as the darkest region inside each eye outline (horizontal only) | yes |
| Head straight, not up/down/turned/tilted | `head_turned` (yaw, pitch), `head_tilted` (roll) | closed-form pose from 5 keypoints | yes |
| Plain background | `background_not_uniform` | edge density beside/above the head | **configurable** (`background_check`: off / info / warn / error; default info = hint shown, not blocking) |
| Only one person | `multiple_faces` | second face ≥ 35% of the main face's height | yes |
| Light even across the face | `uneven_lighting` | left/right luma asymmetry | no (info) |

Why background is not blocking by default: plain walls measured edge density
0–0.004, ordinary rooms 0.02–0.17 — almost every selfie at home would be
refused. Set `KYC_FACE_QUALITY_BACKGROUND_CHECK=error` where a passport-style
photo is actually required.

Non-goals: spoof detection (04), identity (03), ISO/IEC 19794-5 conformance
(criteria borrowed, compliance not claimed).

## 4. Interfaces

| Method | Path | |
|---|---|---|
| POST | `/v1/face-capture/sessions` | start; returns `session_id`, `expires_at` |
| POST | `/v1/face-capture/sessions/{id}/frames` | one video frame → feedback (also `/frames/base64`) |
| POST | `/v1/face-capture/sessions/{id}/still` | the still photo, only in `READY_FOR_STILL` (also `/still/base64`) |
| GET | `/v1/face-capture/sessions/{id}/result` | terminal `ModuleResult` incl. `capture_id` |
| POST | `/v1/face-quality/check` | stateless single-image check for back-office tools only (operator auth); never produces a `capture_id`, so its output can't reach face match |

Frame response (feedback, not a `ModuleResult`):

```json
{
  "state": "SEARCHING | READY_FOR_STILL | CAPTURED | EXPIRED",
  "primary_hint": {"code": "face_too_small", "message_fa": "نزدیک‌تر بیایید"},
  "hints": [{"code": "face_too_small", "message_fa": "نزدیک‌تر بیایید"}],
  "capture_id": null
}
```

`state` becomes `READY_FOR_STILL` after `frames_required_ok` consecutive
passing frames (2 — one lucky frame shouldn't qualify), then `CAPTURED` when
an accepted still arrives.

Result `data`:

```json
{
  "capture_id": "…",
  "face_box": [x1, y1, x2, y2],
  "landmarks_5": [[x, y], …],
  "metrics": {
    "face_height_ratio": 0.0, "sharpness": 0.0, "brightness": 0.0,
    "illumination_asymmetry": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
    "eye_openness_left": 0.0, "eye_openness_right": 0.0,
    "mouth_openness": 0.0, "gaze_offset": 0.0, "clear_logit": 0.0,
    "background_edge_density": 0.0, "still_similarity": 0.0
  },
  "source": "still | stream",
  "frames_seen": 0,
  "stills_rejected": 0,
  "selfie_base64": null
}
```

Box and landmarks are returned so 03/04 reuse this detection instead of
detecting again.

### Reason / hint codes

| code | sev | hint to the user (fa) |
|---|---|---|
| `no_face` | error | چهره پیدا نشد؛ رو به دوربین قرار بگیرید |
| `multiple_faces` | error | فقط خودتان در کادر باشید |
| `face_too_small` / `face_too_large` | error | نزدیک‌تر بیایید / کمی عقب‌تر بروید |
| `face_off_center` / `face_cut_off` | error | صورت را وسط کادر بیاورید / تمام صورت در کادر باشد |
| `face_blurry` | error | تصویر تار است؛ دوربین را ثابت نگه دارید |
| `face_too_dark` / `face_too_bright` | error | نور کافی نیست / نور زیاد است |
| `head_turned` / `head_tilted` | error | مستقیم به دوربین نگاه کنید / سرتان را صاف نگه دارید |
| `eyes_closed` | error | چشم‌هایتان را باز نگه دارید |
| `mouth_open` | error | دهانتان را ببندید |
| `gaze_off_center` | error | مستقیم به دوربین نگاه کنید |
| `face_occluded` | error | ماسک، عینک آفتابی یا دست را از جلوی صورت بردارید |
| `background_not_uniform` | configurable, default info | پس‌زمینه یکدست نیست؛ جلوی دیوار ساده بایستید |
| `uneven_lighting` | info | نور یکنواخت نیست؛ رو به نور بایستید |
| `still_face_mismatch` | error (still only) | عکس با تصویر ویدیو مطابقت ندارد |
| `capture_timeout` | error (terminal) | در زمان مقرر عکس قابل‌قبولی گرفته نشد |

## 5. Decision rules

During capture, errors are hints and the loop continues. The terminal
`ModuleResult` is `pass` when a frame was captured, `review` when the captured
frame carried only warnings, `fail` on `capture_timeout`.

## 6. Models

| Task | Candidate | Serving | Notes |
|---|---|---|---|
| detection + 5 landmarks | SCRFD or YuNet — **same choice as 03 §5** | ONNX | one detector for 02/03/04 |
| pose, eyes, mouth | 106-pt landmarks (`buffalo_l` 2d106) or MediaPipe FaceMesh | ONNX / TFLite | pose via solvePnP |
| occlusion | one small classifier (mask / sunglasses / other / none) | ONNX | trained on Kaggle |

The prototype runs **four** overlapping occlusion detectors (YOLO
`finalbestzh.pt`, MobileNet `obstruction_detector50.pt`, HF ViT
`dima806/face_obstruction_image_detection` re-created on every call, and
pixel-intensity heuristics), none with a recorded dataset or metric. Replace
with one model that has a manifest entry.

## 7. Configuration — `KYC_FACE_QUALITY_*`

`min_face_height_ratio`, `max_face_height_ratio`, `min_edge_margin_ratio`,
`min_sharpness`, `min_brightness`, `max_brightness`,
`max_illumination_asymmetry`, `max_yaw_deg`, `max_pitch_deg`, `max_roll_deg`,
`min_eye_openness`, `max_occlusion_prob`, `detector_conf`,
`frames_required_ok`, `session_ttl_s`,
per-check `enabled` flags (replacing the prototype's runtime `/set_settings`).

Prototype values, for reference only (none measured): Laplacian < 500 →
blurry; FFT mean < 0.5 → blurry; occlusion prob ≥ 0.25; YOLO conf 0.4;
lighting std 20; client sends 1 frame/s.

## 8. Acceptance criteria

| # | Criterion | Status |
|---|---|---|
| Q1 | Every hint carries a Persian instruction | required |
| Q2 | A selfie can only enter 03/04 via a server-issued `capture_id` | required |
| Q3 | Median time-to-capture for a cooperative user ≤ 5 s | proposed |
| Q4 | False rejects on good frames ≤ 5% | proposed; needs labelled real frames from target devices |
| Q5 | Catches ≥ 95% of: eyes closed, mask, sunglasses, head turned beyond limit | proposed |
| Q6 | Thresholds calibrated on face crops from the real browser capture path, recorded in config comments | TBD |
| Q7 | Per-frame p95 ≤ 150 ms on CPU | proposed |
| Q8 | No torch / ultralytics / transformers at serving | required |
| Q9 | Headscarves and beards are never reported as occlusion (false-occlusion rate on them ≤ 1%) | required — the prototype's model fails this at its own threshold, §12 |

## 9. Assumptions to verify before building

1. Browser webcam frames (JPEG from `canvas.toBlob`, default quality, often
   720p) have a very different sharpness scale from phone photos. Calibrate
   on that path, not on `testPic/`.
2. SCRFD/landmark ONNX files from `buffalo_l` load in plain onnxruntime
   without the `insightface` package (the prototype ships prebuilt `.whl`s
   because the package needs a compiler).
3. Pose from 5 landmarks is too coarse for pitch — check whether 106-pt is
   needed.
4. Channel order: the prototype converts BGR→RGB before MediaPipe, and its own
   commit message says "change grb to rgb (i know its wrong)". Write each
   model's expected order in the manifest and pin it in a test.
5. 1 frame/s (prototype interval) is fine for capture; liveness (05) needs
   more. If 02 and 05 share one camera session (§11), the client rate is set
   by 05.

## 10. Migration from the prototype

Keep: the guided capture loop and its UX, EAR eye openness, iris gaze, head
pose from landmarks, per-check toggles (as config).

Drop: Flask; global `settings` mutated by `POST /set_settings` (one client
changes the checks for everyone); client-side choice of the selfie;
the upload-from-file button and path (removed entirely, §2); the many `/process_video*` / `/process_img*`
variants; FAISS index + PostgreSQL (`database.py` has a hard-coded password)
— storing faces is not this module's job; background-uniformity check; three
of the four occlusion models; model loading at import time.

## 11. Open questions

- **One camera session for capture + liveness?** Recommended: capture the
  selfie (02), then continue straight into the liveness challenge (05) on the
  same stream, with 05 checking every sampled frame is the same person as the
  captured selfie. Then the selfie that goes to face match is provably the
  live person.
- Multiple faces: hard fail, or use the largest and warn? (Someone in the
  background is common; a second face held up on a phone is an attack.)
- Clear glasses allowed? Head covering allowed while the face oval is
  visible — a policy decision, not a model decision.

## 12. Build log — what was verified, what changed (2026-10-06)

Assumptions (§9):

- **2 — confirmed.** `det_10g`, `2d106det`, `w600k_r50` load in plain
  onnxruntime; no `insightface` package. `det_10g` has a fixed 640×640
  input. `2d106det` normalises inside its graph and must get raw 0–255
  pixels (the detector wants (x−127.5)/128) — a silent-wrong-output trap.
- **3 — resolved: 5 points are enough for pose, 106 are needed for eyes.**
  5-point **solvePnP was unstable**: mirroring a photo moved yaw from −6° to
  +20° (should be +6°) and gave 52° pitch on a frontal photo. Replaced with a
  closed-form, mirror-symmetric pose from the 5 keypoints
  (`kyc/core/face/geometry.py`); mirror test now agrees within detector noise,
  and a photo with the head tipped far back reads +56° pitch, correct sign.
- **4 — confirmed (channel order).** Landmarks rendered on a photo land on
  the face; eye indices were read off that render.
- **1 — still open.** No webcam frames yet; every threshold is a first guess
  from 15 still photos (marked in `config.py`).

Changes to this spec from what was found:

- **`uneven_lighting` is `info`, not `warn`.** 6 of 15 ordinary indoor test
  photos had left/right light asymmetry above 0.5 (on the whole face box and
  on its inner region alike). As a warning it would send many genuine users
  to manual review for normal room light.
- **Context for anti-spoof doesn't fit a normal selfie.** A face at 35% of
  frame height leaves room for only ~2× context, while MiniFASNet wants 2.7×
  and 4.0×. The check exists (`min_context_scale`) but is off until 04
  measures what it needs — see 04 §8.2.
- **Detector latency ~220–290 ms/frame** on the dev machine (no AVX2), over
  Q7's proposed 150 ms. Re-measure on the target server; `det_500m`
  (buffalo_s) is the fallback.

### 2026-10-07 — still step and the remaining checks

- **Gaze: the 2d106 "pupil" points don't track the iris.** Moving the iris in
  a photo by 0.2 eye-widths left points 38/88 exactly where they were — they
  are interpolated eye centres. Replaced by locating the iris directly (darkest
  25% of pixels inside the eye outline); on the same test it moved ±0.13, and
  test photos looking at the camera stay within ±0.10. Horizontal only.
- **Occlusion: the prototype's MobileNetV3 is biased against headscarves and
  beards.** At its own threshold (logit 0.25) it flagged 5 of 22 uncovered
  faces — four women in headscarves and a bearded man. Serving uses −3.0:
  none of the 22 flagged, the one real covered face (−4.28) caught, most
  synthetic masks missed. It is a stop-gap; Q5 needs a retrained model with
  headscarves and beards as *uncovered* examples. The prototype's YOLO
  (`finalbestzh.pt`) and HF ViT were not migrated.
- **Mouth**: closed mouths 0.004–0.048, a broad smile with teeth 0.124;
  threshold 0.15. No open-mouth samples yet.
- **Still ↔ video identity**: ArcFace cosine, same image rescaled 0.99,
  different people ≈ 0.0–0.3; threshold 0.5. End to end with the real
  models: a different person's still was rejected (`still_face_mismatch`),
  the same person's accepted (0.9985).
- **Latency**: 350–580 ms per frame with all checks on the dev machine (no
  AVX2). Fine for guidance at the prototype's 1 frame/s; re-measure on the
  server.

- **Darkness was reported as blur.** Darkening a sharp photo to 25% also
  drops its Laplacian variance below the floor, and since blur was checked
  first, the user was told "blurry" when the fix was more light. Blur is now
  judged only at acceptable exposure. (Found by the calibration script.)

Calibration tooling (2026-10-07): `/dev/face-capture` (only with
`KYC_DEBUG=true`) drives the real API from a webcam, shows per-frame
metrics, and saves labelled frames to the developer's own disk;
`python -m scripts.calibrate_face_quality --dir <frames>` reports, per label,
how often the expected check fired, how often normal/headscarf frames were
wrongly blocked, and the metric ranges to pick thresholds from.

Not built yet: retrained occlusion model (Q5), multi-worker session store
(Redis), threshold calibration on real webcam frames (tooling ready).

### 2026-10-11 — first real webcam test: three checks failed

A selfie was captured with **a hand over the mouth, eyes looking down, in
front of a marble-veined wall**. All three should have blocked:

| Check | Real frame | Why it passed | Change |
|---|---|---|---|
| Occlusion | logit −2.285 | threshold was −3.0; an uncovered headscarf test face scores −2.34, so **no threshold on this model separates the two** | interim −2.0 (catches it; nearest uncovered test face −1.92). The model must be retrained — Q5/Q9 |
| Mouth | openness 0.029 | landmarks "see" a closed mouth under the hand | none — cover is caught by occlusion, not the mouth check. A lip-redness test was tried and dropped: many uncovered faces score near 0 too |
| Gaze | horizontal 0.031 | only left/right was measured | vertical iris position added; looking down −0.159 vs −0.19 to −0.42 looking at the camera; window [−0.55, −0.17], **interim, one real sample** |
| Background | edge density 0.019 | Canny 40/120 doesn't see soft marble veins; and it was info-only | Canny 15/45 (wall 0.055, plain ≤ 0.017), threshold 0.03, **blocking by default** (team requirement) |

Also fixed: the iris locator used a hard "darkest 25%" cut, which selected
every pixel when the sclera was uniform and returned the outline's centre —
now a darkness-weighted centroid. And the hint order: the client shows the
first error, which was "look at the camera" for a covered mouth; blocking
hints are now ordered with cover first.

Lesson, again: thresholds from still photos did not survive the first live
frame. Calibration on labelled webcam frames is the next step, not optional.

### 2026-10-11 — second real test

A selfie passed with the face filling 75% of the frame in front of a room
background (dark doorway, bright wall). Changes:

- **Background: edges alone miss blurred backgrounds.** Webcams leave the
  background out of focus; the frame read edge density 0.027, and an
  out-of-focus office test photo only 0.017. Added luma spread (std): plain
  walls 3.5–20.4, the office 44.7, real room frames 42–50. Not plain =
  edge density > 0.03 **or** std > 30 (the prototype used std 25–30).
- **Too close: limit lowered from 0.80 to 0.65** of frame height. At 0.75
  there was almost no background to judge and anti-spoof had no context.
- **Hints no longer flicker.** The server still checks every frame (the
  capture streak needs it) but returns a debounced `primary_hint`: a problem
  becomes the shown hint only once it appears in 2 of the last 3 frames, and
  stays until another one does or the frames come back clean. The page also
  holds each message at least 1.5 s.

**Open (2026-10-11):** the room background was correctly rejected on a real
frame, but the *acceptance* side — a real plain wall at home, with its
shadows and light falloff, staying under std 30 — has not been tested on a
webcam yet. Assumed to work for now (team decision); verify before relying
on `background_check=error` in production, and raise
`max_background_luma_std` if plain walls are refused.

