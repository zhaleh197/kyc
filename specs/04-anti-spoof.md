# 04 — Passive anti-spoofing (presentation attack detection)

Status: **up-front spec**. Prototype: `phase3-antispoof/` (commit `bc1ef49`).

## 1. Purpose

From the captured selfie and sampled liveness frames, estimate whether the
face is a live person in front of the camera or a presentation attack:
printed photo, photo/video on a screen (replay), cut-out. Passive = no user
action. It complements active liveness (05); neither is enough alone.

## 2. What the prototype does (reconstructed from the code)

- **Models**: two MiniFASNet classifiers from Minivision's
  Silent-Face-Anti-Spoofing (Apache-2.0):
  `2.7_80x80_MiniFASNetV2.pth` and `4_0_0_80x80_MiniFASNetV1SE.pth`. Each
  sees an 80×80 crop of the face box enlarged by the factor in its filename
  (2.7× and 4.0×) — so they look at the face *plus surroundings*, which is
  where screen bezels, paper edges and moiré show up.
- **Detector**: RetinaFace in Caffe (`deploy.prototxt` + `.caffemodel`).
- **Scoring**: softmax of each model over 3 classes, summed, divided by the
  model count. `real = class 1`.
- **Decision**: `real_score ≥ 0.9` → `real`; else if class 2 > class 0 →
  `uncertain` ("unrecognized_pattern"); else `fake` ("classic_spoof_pattern").
- **Changes from upstream**: the 3:4 aspect-ratio check was disabled
  (`check_image` always returns True); the "uncertain" tier was added.
- **Serving**: torch on `DEVICE_ID = 0`, FastAPI with `allow_origins=["*"]`,
  16 MB upload limit, a visualisation endpoint, debug `print`s.

### What's wrong with it

1. **Class 2 is not "unknown".** Upstream `test.py` treats only `label == 1`
   as real; 0 and 2 are both spoof classes (different attack types). So
   "uncertain" fires on a *specific kind of spoof*, not on uncertainty.
2. **The 3:4 check existed for a reason.** The 4.0× crop must fit in the
   frame. On a landscape webcam frame, or with the face close to the camera,
   `_get_new_box` silently shrinks the scale, so the model sees a different
   crop from the one it was trained on — and nothing reports it.
3. **The 0.9 threshold was never measured.** No eval set exists.
4. Models are found by listing the model directory, so any stray file breaks
   it, and the order depends on the filesystem.
5. Torch at serving, plus a second face detector beside the one in 02/03.

## 3. Proposed approach

**Keep the models, fix how they're used, then measure.** MiniFASNet is small
(80×80, ~ms on CPU), Apache-2.0 and designed for exactly this; there's no
reason to switch before we have a measurement that says it's not good enough.

1. **Export both models to ONNX** (Kaggle or locally — tiny); check that ONNX
   output matches torch within 1e-4. No torch at serving.
2. **Correct scoring**: `live_score = mean over models of P(class 1)`;
   `spoof_score = 1 − live_score`. Report per-model scores. Remove the class-2
   "uncertain" label — uncertainty is the band between two thresholds (§6).
3. **Correct geometry instead of forcing 3:4**: use the face box from 02's
   detector; for each model compute the scaled crop; if it doesn't fit inside
   the frame, **report `crop_clamped`** rather than silently shrinking. 02's
   capture loop already guides the user to a face size and position where the
   4.0× crop fits (`min_edge_margin_ratio`), so clamping becomes rare.
   Verify that boxes from SCRFD/YuNet give the same scores as the RetinaFace
   boxes the models were trained with (§8.2) — if not, add a box-mapping step
   or keep RetinaFace (convert to ONNX).
4. **Score several frames, not one**: the captured selfie plus every Nth
   liveness frame (05). Aggregate with the **minimum** live score over frames
   (an attacker has to fool every frame) and report the median too.
5. **Calibrate on our own data** (§7): collect bona fide frames from the real
   browser capture path, plus prints and screen replays (phone, laptop,
   monitor) of consenting people. Measure the pretrained models first.
6. **Fine-tune only if needed**: if APCER/BPCER miss the targets, fine-tune on
   Kaggle with the upstream training code (`src/train_main.py`, already in
   the repo) on the collected set, export ONNX, record in the manifest — the
   same loop as OCR's CRNN.
7. **Final "live" needs 04 and 05 together** (07 policy). Camera-injection /
   virtual-camera attacks are 06, not this module.

## 4. Scope

In scope: print and replay attacks on RGB frames from phone / webcam.
Non-goals: 3D silicone masks (needs depth/IR — say so in the API docs);
injection / virtual cameras (06); card forgery (06).

## 5. Interfaces

`POST /v1/antispoof/check` — `capture_id` (from 02), or a liveness
`session_id` (05), or (dev only) an image. Optional `face_box` to skip
detection.

`data`:

```json
{
  "live_score": 0.0,
  "live_score_median": 0.0,
  "frames_checked": 1,
  "per_model": [{"name": "minifasnet_v2_s2.7", "live_score": 0.0},
                {"name": "minifasnet_v1se_s4.0", "live_score": 0.0}],
  "crop_clamped": false,
  "face_box": [x1, y1, x2, y2]
}
```

### Reason codes (proposed)

| code | sev |
|---|---|
| `no_face` | error |
| `spoof_detected` | error — `live_score < reject_below` |
| `spoof_uncertain` | warn — between the thresholds |
| `crop_clamped` | warn — model saw a smaller context than it was trained on |
| `models_disagree` | warn — per-model scores differ by > `max_disagreement` |
| `face_too_small_for_pad` | warn — face crop below ~80 px before resize |

## 6. Decision rules and configuration — `KYC_ANTISPOOF_*`

`live_score ≥ accept_above` → pass; `< reject_below` → fail; between →
review. Both thresholds come from the measured curve (§7).

Config: `accept_above`, `reject_below`, `max_disagreement`, `min_face_px`,
`models` (explicit list with each one's scale and input size),
`frame_aggregation` (`min`), `liveness_frame_stride`.

## 7. Acceptance criteria (ISO/IEC 30107-3 terms)

| # | Criterion | Status |
|---|---|---|
| S1 | Eval set: ≥ 200 bona fide + ≥ 200 per attack type (print, screen replay), ≥ 3 phone models, ≥ 3 screen types, in the manifest | proposed |
| S2 | APCER ≤ 5% per attack type at the operating point (reported per type, not pooled) | proposed |
| S3 | BPCER ≤ 5% on the real browser capture path | proposed |
| S4 | ONNX output = torch output within 1e-4 | required |
| S5 | p95 ≤ 100 ms per frame on CPU (both models) | proposed |
| S6 | Pretrained baseline measured and recorded *before* any fine-tune | required |

## 8. Assumptions to verify before building

1. ~~Class semantics~~ — **resolved**: upstream `test.py` takes `label == 1`
   as real, anything else as fake.
2. A different detector's boxes (SCRFD/YuNet vs RetinaFace) shift the scaled
   crops enough to change scores. Compare scores on 50 frames with both.
   **Finding from 02 (2026-10-06):** a normal selfie (face ~35% of frame
   height) leaves room for only ~2× context, so the 2.7× and 4.0× crops are
   clamped on typical frames, not rare ones. Measure scores at the clamped
   context before deciding whether capture must push the user back
   (`KYC_FACE_QUALITY_MIN_CONTEXT_SCALE`).
3. Browser JPEG compression (`canvas.toBlob` default quality) removes the
   texture cues these models use. Measure BPCER on frames from that path; if
   it's bad, raise the client's JPEG quality before touching the model.
4. Modern OLED / high-refresh screens are much harder than the
   `spoofimagetest/` samples. Collect them on purpose.

## 9. Migration

Keep: MiniFASNet ensemble, per-model scaled crops, averaging, the upstream
training code for a later fine-tune.
Drop: torch at serving; directory listing for models; the class-2
"uncertain" tier; the disabled-but-present 3:4 check; `print`s; startup
globals; `allow_origins=["*"]`; the visualisation endpoint; the heuristic
moiré/saturation checker in `phase1_facedetect/claude_api` (made-up weights
0.4/0.35/0.25).

## 10. Test plan

Unit: crop geometry including a box near the edge (must report
`crop_clamped`) against the upstream implementation. Parity: ONNX vs torch on
fixed inputs. Pipeline: fake classifier → each reason code; frame aggregation
uses the minimum. Eval (manual): APCER/BPCER per attack type, in the manifest.

## 11. Open questions

- Collecting attack data means photographing prints and screens of real
  people — needs their consent and a storage rule.
- If the client app (not a browser) is planned, raise JPEG quality /
  resolution for capture there; it directly affects this module.
