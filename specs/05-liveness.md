# 05 — Active liveness (challenge–response)

Status: **in progress** (branch `liveness`, 2026-10-14): `kyc/modules/liveness/`, see §12. Originally an up-front spec. Prototype: `phase4-liveness/` — FastAPI + MediaPipe
Face Mesh, session-based; 4 random tasks out of {BLINK, SMILE, LOOK_STRAIGHT,
TURN_LEFT, TURN_RIGHT, GO_FORWARD, GO_BACK}, 7 s per task, in-memory session
store with 2-min TTL. This is the most complete prototype; its README already
lists most production concerns.

## 1. Purpose

Prove a live, responsive person is in front of the camera *now*, by asking for
randomised actions and checking them frame by frame. Random order defeats a
pre-recorded video; per-task timeouts bound the attacker's reaction window.

## 2. Scope

In scope: session lifecycle, randomised challenge, per-frame evaluation,
terminal verdict as a `ModuleResult`.

Non-goals: real-time deepfake puppeteering of a virtual camera (06 + client
attestation); 3D masks (a masked attacker can blink and turn).

## 3. Interfaces

| Method | Path | |
|---|---|---|
| POST | `/v1/liveness/sessions` | create from a 02 `capture_id`; returns `session_id`, `tasks`, instructions (fa + en), `expires_at` |
| POST | `/v1/liveness/sessions/{id}/frames` | one frame (multipart or base64) → progress |
| GET | `/v1/liveness/sessions/{id}` | current state |
| GET | `/v1/liveness/sessions/{id}/result` | terminal `ModuleResult` (409 until terminal) |
| DELETE | `/v1/liveness/sessions/{id}` | abandon |

No `/reset`: a failed session is terminal; the client creates a new one (and
the orchestrator counts attempts). Reset on the same id lets an attacker retry
indefinitely against the same session.

Frame response (progress, not a `ModuleResult`): `state`
(`RUNNING`/`VERIFIED`/`FAILED`/`EXPIRED`), `current_task`,
`instruction_fa`, `task_index`, `total_tasks`, `time_remaining`, `hint_code`
(e.g. `no_face`, `multiple_faces`, `too_far`).

Result `data`: `tasks[{task, passed, elapsed_ms, frames}]`, `frames_received`,
`antispoof{min_live_score, mean_live_score, frames_checked}`,
`capture_id` (the 02 selfie this session was bound to), `min_identity_similarity`.

### Reason codes (proposed)

`task_timeout`, `face_lost`, `multiple_faces`, `face_changed` (identity
drift between frames), `spoof_during_session` (04 below threshold on sampled
frames), `frame_rate_anomaly` (timestamps impossible for a live camera),
`session_expired` — all error; `low_frame_rate` — warn.

## 4. Decision rules and binding

- All tasks passed within time and no error reason → `pass`.
- **Identity continuity**: the session is created from 02's `capture_id`
  and continues on the same camera stream. Sampled frames are embedded (03's
  model) and each must match the captured selfie. Otherwise an attacker
  passes 02 with the victim's photo and performs the challenge themselves —
  or the reverse.
- **Spoof on the same stream**: run 04 on sampled frames; the challenge proves
  responsiveness, 04 checks the medium.
- Together: 02 captured the selfie, 05 proved that same face is live, 03
  matches that selfie to the card portrait — so the live person is the one
  matched to the document.

## 5. Models

MediaPipe Face Mesh (468 landmarks + z). This is **TFLite via the
`mediapipe` package, not onnxruntime** — a deviation from 00 §6. Options:

1. accept `mediapipe` as a serving dependency (CPU, no torch — the policy's
   intent is met, its letter is not);
2. switch to 106-pt landmarks from `buffalo_l` (ONNX, shared with 02/03) and
   re-derive EAR/smile/yaw/depth on those points.

Recommendation: option 2 if the landmark quality on eyes/mouth suffices —
one landmark model for 02/03/05. Verify first (§8.1).

## 6. Configuration — `KYC_LIVENESS_*`

`tasks_per_session` (4), `task_time_limit_s` (7), `session_ttl_s` (120),
`max_frame_bytes`, `eye_ar_threshold` (0.21), `blink_min_frames` (2),
`blink_max_frames` (10), `yaw_threshold_deg` (20), `smile_ratio_threshold`
(0.75), `smile_frames_required` (3), `z_depth_threshold` (15),
`antispoof_every_n_frames`, `identity_check_every_n_frames`, `task_pool`.

All values in parentheses are the prototype's and **unmeasured**. Blink
frame-count thresholds silently depend on frame rate: 2–10 frames is
~70–330 ms at 30 fps but ~200 ms–1 s at 10 fps. Express them in
milliseconds using frame timestamps instead.

## 7. Acceptance criteria

| # | Criterion | Status |
|---|---|---|
| L1 | Genuine users complete a session on first try ≥ 90% (web, 3+ phone models, indoor lighting) | proposed |
| L2 | Replay of a recorded genuine session (any order) passes ≤ 1% | proposed — random order + per-session nonce makes this near 0 by design; test it |
| L3 | Printed photo / static screen passes 0% | required |
| L4 | Face swap mid-session → `face_changed` | required |
| L5 | Session state survives multiple uvicorn workers | required before deployment |
| L6 | Per-frame p95 ≤ 80 ms on CPU | proposed |

## 8. Assumptions to verify before building

1. `buffalo_l` 106-pt landmarks give usable EAR and mouth-corner signals
   (compare with MediaPipe on the same 50 frames).
2. **Frame rate reaching the server.** At 100–200 ms per upload (README
   suggestion) the server sees 5–10 fps; a blink lasts 100–400 ms. Measure
   whether blinks are reliably caught at that rate — likely not at 5 fps.
3. Yaw sign convention (TURN_LEFT = negative yaw) vs mirrored front-camera
   previews — the user's "left" and the image's left differ. Test on a real
   phone, not a laptop.
4. `GO_FORWARD` / `GO_BACK` use the landmark z mean relative to the first
   frame; confirm z is metric enough (MediaPipe z is relative) or use face-box
   scale change instead.
5. Session storage: the prototype holds a native `FaceMesh` object per
   session, which is why it can't go to Redis. With a stateless landmark model
   (§5 option 2) the per-session state is small counters → serialisable.

## 9. Migration from the prototype

Keep: task set, random order, per-task timeout, landmark-based detectors,
session TTL reaper, frame-size cap, the README's production notes.
Rewrite: session store behind an interface (in-memory for tests, Redis for
deployment); thresholds into config; frame-count thresholds into time;
per-session `FaceMesh` → shared stateless model.
Drop: `/reset`; `allow_origins=["*"]`; `/healthz` (use app-level `/health`).
Add: identity continuity, 04 on frames, `ModuleResult` on completion,
Persian instructions, attempt counting (in 07).

## 10. Test plan

Unit: each task detector against scripted landmark sequences (blink too
short, too long; turn in the wrong direction). State machine: timeouts,
face lost, terminal states immutable. Pipeline: fake landmark model → full
session to VERIFIED and to each failure. Manual: scripted real sessions on
phones; replay and photo attacks.

## 11. Open questions

- Frames over HTTP polling vs WebSocket? WebSocket halves overhead and gives
  trustworthy inter-frame timing; polling is simpler to deploy behind a proxy.
- Should the client send a short video clip instead of streaming frames?
  (Simpler server; loses real-time instructions.)
- Accessibility: users who cannot smile/blink on cue (facial paralysis) need a
  fallback path — manual review, not a hard fail.

## 12. Build log (2026-10-14)

Built: `POST /v1/liveness/sessions {capture_id}` → random tasks;
`POST /sessions/{id}/frames`; `GET /sessions/{id}/result` → `ModuleResult`.
Dev page: "start liveness" after a selfie is captured.

**Changes from the phase4 prototype**

| Prototype | Problem | Now |
|---|---|---|
| LOOK_STRAIGHT, CLOSE_EYES tasks | a still photo passes them | removed; pool = BLINK, TURN_LEFT, TURN_RIGHT, SMILE, MOVE_CLOSER — all movements |
| fixed EAR/yaw thresholds, one frame enough | a tilted photo "turns" | every task is relative to the person's **baseline** (median of the first frontal frames); turns must **start frontal** |
| blink measured in frames (2–10) | frame-rate dependent; phone sends ~1–2 frames/s | closed (< 0.6× baseline) then open (≥ 0.85×), any timing; instruction says "close for a second" because a 0.1–0.4 s blink falls between frames |
| face lost one frame → FAILED | brittle | up to `max_face_lost_s` (1.5 s) tolerated |
| no identity, no spoof check | anyone / any screen can do the moves | every 2nd frontal frame: ArcFace vs the captured selfie (< 0.45 → `face_changed`); anti-spoof (4.0× MiniFASNet, spec 04) on sharp frontal frames, median decides |
| MediaPipe FaceMesh per session | extra dependency, per-session native object | buffalo_l SCRFD + 2d106 landmarks, shared with 02/03/04 |
| `/reset` on a failed session | unlimited retries against one challenge | none: a failed session is final; start a new one |

Also added: `kyc/core/face/registry.py` — one loaded instance per model file
for the whole process (the four modules had each loaded their own copies; the
dev machine kept running out of memory).

Real-photo sanity check (tester's phone selfies): frontal yaw within ±10°
and sign flips when mirrored; baseline eye openness 0.24–0.33; neutral
mouth/eye-distance 0.70–0.76 (smile threshold ×1.12 ≈ 0.81).

**To verify on a phone:** the TURN_LEFT/RIGHT direction
(`turn_left_is_positive_yaw`, reasoned for an un-mirrored front-camera frame,
not yet observed), whether blinks are caught at the real frame rate, and the
anti-spoof scores of *video* frames (calibrated on stills so far).
