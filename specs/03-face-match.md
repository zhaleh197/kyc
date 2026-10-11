# 03 — Face matching (document portrait vs selfie)

Status: **in progress** (branch `face-match`, 2026-10-11): `kyc/modules/face_match/` — card portrait vs `capture_id`, thresholds proposed from repo data, see §12. Originally an up-front spec. Prototype: `phase2_facematch/facematching.py`
(InsightFace `buffalo_l`, `POST /compare`). A second variant exists in
`phase1_facedetect/claude_api/main.py` (`/kyc/face-match`).

## 1. Purpose

Answer: *is the person in the selfie the person on the ID card?* Input is the
portrait crop from OCR (`portrait_base64`, 01) and the selfie captured by 02,
referenced by its server-issued `capture_id` (never a client-uploaded image in
production — see 02 §2). Output is a similarity, a calibrated decision, and
reasons.

## 2. Scope

In scope: 1:1 verification between two images; card-portrait vs live-selfie
as the primary pair.

Non-goals: 1:N search / duplicate-identity detection (the prototype's FAISS
index — separate future module, with its own privacy review); age estimation;
spoof detection (04).

## 3. Interfaces

`POST /v1/face-match/verify` — multipart `reference` + `probe`, or JSON with
`reference_base64` / `probe_base64`. Optional `reference_kind`
(`id_card_portrait` | `selfie`) selects the threshold set.

`data`:

```json
{
  "similarity": 0.53,
  "threshold": 0.0,
  "threshold_set": "id_card_portrait",
  "reference_face": {"box": [..], "det_score": 0.0, "faces_found": 1},
  "probe_face": {"box": [..], "det_score": 0.0, "faces_found": 1},
  "model": "arcface_r50_w600k"
}
```

Report **raw cosine similarity** in [-1, 1]. Do not report the prototype's
`(sim + 1) / 2 * 100` "percent" — it looks like a probability, is not one, and
compresses the useful range (cosine 0.45 becomes "72.5%").

### Reason codes (proposed)

| code | sev |
|---|---|
| `reference_no_face` / `probe_no_face` | error |
| `reference_multiple_faces` / `probe_multiple_faces` | warn (largest face used) |
| `reference_low_quality` (small, blurry card photo) | warn |
| `faces_do_not_match` | error — similarity < `reject_below` |
| `faces_match_uncertain` | warn — between `reject_below` and `accept_above` |

## 4. Decision rules

Two thresholds, not one: `similarity ≥ accept_above` → pass;
`< reject_below` → fail; between → review. Both chosen from a measured
FMR/FNMR curve (§7), per `reference_kind` — card portraits are small, printed,
sometimes years old, and score systematically lower than selfie-vs-selfie.

## 5. Models — and what to do about `buffalo_l`

| Component | Candidate | Serving |
|---|---|---|
| detection + 5-pt landmarks | SCRFD `det_10g.onnx` (in `buffalo_l`) **or** YuNet (OpenCV zoo, MIT) | ONNX |
| alignment | similarity transform to the ArcFace 112×112 template | numpy/cv2 |
| embedding | ArcFace R50 `w600k_r50.onnx` (in `buffalo_l`) **or** SFace (OpenCV zoo, Apache-2.0) | ONNX, CPU |

### The licence problem

InsightFace's model zoo says, verbatim: "ALL models are available for
non-commercial research purposes only." That covers `buffalo_l` (and
`antelopev2`). The *code* is MIT; the *weights* are not. A KYC service used by
a business is commercial use.

### Decision path (project is commercial — confirmed 2026-10-06)

**Current decision (2026-10-06): keep `buffalo_l` as is for now; the licence
decision is deferred.** Development and migration go ahead with `buffalo_l`.
The licence question stays open and must be settled before commercial launch —
tracked in §11. The steps below are the path for when it is revisited.

The team's own testing found `buffalo_l` the best of the models tried.
Shipping it commercially without a licence is not an option, so the choice is
between **licensing the best model** and **making a free model good enough**.
(This is an engineering summary, not legal advice — have a lawyer confirm the
final choice.)

1. **Now, regardless of the outcome:**
   - Put the models behind `FaceDetector` / `FaceEmbedder` interfaces,
     injected like OCR's recognisers. Thresholds are stored per model
     (`model` in `data`), because a threshold for one model means nothing for
     another.
   - Build the paired eval set (card portrait vs camera selfie, same and
     different people) — it is needed for thresholds anyway (M1).
   - **Ask InsightFace for a commercial licence quote** for `buffalo_l`
     (detector + recogniser). Check that payment and contract are actually
     possible from where the business operates, not just the price.
2. **Measure on that set**, at the operating point, not on accuracy overall:
   `buffalo_l` vs YuNet (MIT) + SFace (Apache-2.0). The question is how much
   FNMR at FMR ≤ 0.1% (M2) the free pair loses on *card portrait vs selfie* —
   earlier tests may not have measured that specific pair.
3. **Decide:**

| Result | Choice |
|---|---|
| Licence obtainable at an acceptable price | **Buy it, ship `buffalo_l`.** Least engineering, best accuracy. |
| No licence, and SFace's gap is small | Ship SFace; widen the `review` band to cover the gap. |
| No licence, and SFace's gap is large | Ship SFace with a wide `review` band (more manual review), and **fine-tune SFace on our own consented KYC pairs** on Kaggle — the same path that took OCR's digits from 7% to 68%. Card-portrait-vs-selfie is a narrow domain; a small domain fine-tune can close much of the gap. |

Ruled out: training from scratch (needs millions of commercially licensed
faces); other open models such as AdaFace, FaceNet or ArcFace variants — their
weights are trained on MS1M / WebFace / Glint / VGGFace2, which are
research-only datasets, so they have the same problem as `buffalo_l`;
fine-tuning *from* `buffalo_l` — the result is still a derivative of
non-commercial weights.

Fine-tuning on own users' data requires their consent for that use and a
retention rule (07 §5) — put it in the consent text from day one so the data
can be used later.

Have legal confirm SFace too: OpenCV zoo licenses the files Apache-2.0, but
the zoo page doesn't name the training dataset.

## 6. Configuration — `KYC_FACE_MATCH_*`

`accept_above`, `reject_below` (per `reference_kind`), `det_conf`,
`min_face_px`, `embedding_model`, `detector_model`.

Prototype values for reference only: `phase2` match if percent > 72.5
(= cosine > 0.45); `claude_api` cosine distance ≤ 0.40 (= similarity ≥ 0.60).
The two prototypes disagree by 0.15 in cosine space and neither was measured.

## 7. Acceptance criteria

| # | Criterion | Status |
|---|---|---|
| M1 | Threshold chosen from a measured curve on **card-portrait vs selfie** pairs, recorded in the manifest with the pair count | TBD — requires a paired dataset |
| M2 | At the chosen `accept_above`: FMR ≤ 0.1% | proposed |
| M3 | At the chosen `reject_below`: FNMR ≤ 3% | proposed |
| M4 | Same image in both slots → similarity > 0.95 (sanity) | required test |
| M5 | Embedding of a flipped / slightly rotated face stays within ε of original | required test |
| M6 | p95 ≤ 300 ms on CPU for the pair (2 detections + 2 embeddings) | proposed |
| M7 | Card portrait and selfie decoded, detected, aligned identically to how the eval was run | required |

## 8. Assumptions to verify before building

1. **Is there a paired dataset?** The whole module's threshold depends on
   (card portrait, selfie) pairs of the same and different people. The
   Roboflow card set has portraits but no matching selfies. Without this, M1
   cannot be met and the module must stay `review`-only. This is the largest
   risk in this spec — resolve it first.
2. The OCR portrait crop (2% padding, frame coordinates, unrectified,
   possibly tilted) is detectable by SCRFD at all. Run SCRFD over the 19
   held-out card images' `img` crops and count detections.
3. Card portraits may be greyscale or low-contrast; check whether converting
   the selfie to greyscale before embedding raises genuine-pair similarity.
4. `buffalo_l` ONNX files load in plain onnxruntime without the
   `insightface` package (see 02 §8.2).
5. The prototype takes `faces[0]`, which is detector order, not the largest
   face. Confirm the fix (largest by area) doesn't change results on
   `testimages/`.
6. Channel order: `phase2` feeds BGR (correct for insightface's `get`),
   `read_image` helper converts to RGB but is unused. Pin it in a test.

## 9. Migration from the prototype

Keep: SCRFD + ArcFace pipeline, L2-normalised embeddings, cosine similarity.
Drop: the percent mapping; double `FaceAnalysis` initialisation at import;
`ctx_id=0` (GPU) default; `print` of similarity; `allow_origins=["*"]`;
committed `.whl` files for dlib/insightface; test photos of real people.

## 10. Test plan

Unit: alignment against a known landmark set; cosine on fixed vectors.
Pipeline: fake detector/embedder → every reason code and each side of both
thresholds. Eval (manual): FMR/FNMR/ROC on the paired set, stored with the
chosen thresholds in the manifest.

## 11. Open questions

- **Before commercial launch:** settle the `buffalo_l` licence (deferred 2026-10-06; see §5).
- Retain embeddings? They are biometric data; default is **no** — compute,
  compare, discard.

## 12. Build log (2026-10-11)

Built: `POST /v1/face-match/verify` (multipart `reference` + `capture_id`, and
`/verify/base64`). The probe is only a `capture_id` — the selfie the
face-capture module approved and kept; there is no selfie upload field (a
test enforces it). Reuses `kyc/core/face/` (SCRFD + ArcFace, buffalo_l).

Assumptions checked (§8):

- **2 — the OCR portrait crop is detectable: yes, with a margin.** On the
  46 real val/test cards, SCRFD found the face in **46/46** once a 30% black
  margin was added (faces touching the crop edge are missed without it).
  On Roboflow's augmented training images (rotated, noise) 128/461 failed;
  a rotation fallback (±90/180, then ±30…150°) recovered 85 of them. Cards are
  not rectified before OCR, so a real card photographed sideways needs this.
  Card-portrait faces: 36–400 px tall, median 160.
- **3 — grayscale: no gain** (0.44 → 0.43 on a genuine pair). Kept color.
- **1 — paired dataset: partial.** The repo's card set contains the cards of
  people who also appear in the test selfies. Assuming the matches below are
  the same people (**to be confirmed by the team**):

| Pair | Similarity |
|---|---|
| Card vs phone selfie, person B | 0.64–0.73 |
| Card vs phone selfie, person A | 0.42–0.68 |
| Card vs webcam selfie, the tester (real val/test cards) | 0.38–0.44 |
| same, with a hand over the mouth | 0.25–0.30 |
| Two webcam selfies of the tester | 0.69 |
| Selfies vs other people's real cards | ≤ 0.22 (median ≈ 0.04) |
| Card vs card, different people (augmented set, duplicates grouped) | p99 0.19, p99.9 0.31 |

**Card-vs-selfie scores far lower than selfie-vs-selfie** — the 0.45 used
by the prototype would have sent the tester to review. Proposed:
`accept_above` 0.40, `reject_below` 0.25, review between. Not yet an M1-grade
measurement: tens of pairs, labels assumed, one webcam.

Also found: the dataset holds many exact copies of the same card
(`-Copy`, similarity 1.0) — any future eval must group by identity first.
