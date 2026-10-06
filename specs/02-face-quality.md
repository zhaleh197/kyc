# 02 — Face capture & quality (takes the selfie)

Status: **up-front spec**. Prototype: `phase1_facedetect/final.py` (Flask) +
`templates/final.html`. A single-image variant exists in
`phase1_facedetect/claude_api/main.py` (`/kyc/face-quality`).

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

Removing the button is not the protection by itself: the API can be called
directly (curl, a script) without the page. What enforces it is the server
side — frames are accepted only inside an open capture session, and 03/04/07
accept only a `capture_id`. The remaining route, feeding a photo or video
through a *virtual camera*, is handled by liveness (05) and injection
detection (06b).

## 3. Scope

Checks on each frame (all measured on the **face crop**, not the whole frame —
OCR's whole-frame sharpness is the mistake not to repeat, 01 §9.3):

- exactly one face (§11 for the "largest + warn" alternative)
- face size and position — including a margin large enough for 04's crop
  (MiniFASNet's 4.0× box must fit inside the frame; see 04 §4.2)
- sharpness, brightness, lighting asymmetry across the face
- head pose (yaw / pitch / roll) within limits
- eyes open, gaze roughly forward
- occlusion: mask, sunglasses, hand, hair over eyes

Non-goals: spoof detection (04), identity (03), ISO/IEC 19794-5 conformance
(criteria borrowed, compliance not claimed), background uniformity (a
passport-photo rule; the prototype checks it — drop unless a client needs it).

## 4. Interfaces

| Method | Path | |
|---|---|---|
| POST | `/v1/face-capture/sessions` | start; returns `session_id`, `expires_at` |
| POST | `/v1/face-capture/sessions/{id}/frames` | one frame → feedback |
| GET | `/v1/face-capture/sessions/{id}/result` | terminal `ModuleResult` incl. `capture_id` |
| POST | `/v1/face-quality/check` | stateless single-image check for back-office tools only (operator auth); never produces a `capture_id`, so its output can't reach face match |

Frame response (feedback, not a `ModuleResult`):

```json
{
  "state": "SEARCHING | CAPTURED | EXPIRED",
  "hints": [{"code": "face_too_small", "message_fa": "نزدیک‌تر بیایید"}],
  "capture_id": null
}
```

`state` becomes `CAPTURED` after `frames_required_ok` consecutive passing
frames (proposed 2 — one lucky frame shouldn't pass); the best of those is
stored.

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
    "occlusion": {"mask": 0.0, "sunglasses": 0.0, "other": 0.0}
  },
  "frames_seen": 0,
  "selfie_base64": null
}
```

Box and landmarks are returned so 03/04 reuse this detection instead of
detecting again.

### Reason / hint codes (proposed)

| code | sev | hint to the user |
|---|---|---|
| `no_face` | error | face the camera |
| `multiple_faces` | error (§11) | only you in the frame |
| `face_too_small` / `face_too_large` | error | move closer / further |
| `face_off_center` / `face_cut_off` | error | center your face |
| `face_blurry` | error | hold still |
| `face_too_dark` / `face_too_bright` | error | lighting |
| `uneven_lighting` | warn | face the light |
| `head_turned` / `head_tilted` | error | look straight |
| `eyes_closed` | error | open your eyes |
| `gaze_off_center` | warn | look at the camera |
| `face_occluded_mask` / `_sunglasses` / `_other` | error | remove it |
| `capture_timeout` | error (terminal) | session ended without a usable frame |

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
