# Training on Kaggle, serving on CPU

Every model in this project is trained on a Kaggle GPU and exported to ONNX.
Inference runs through `onnxruntime` on CPU. Nothing at serving time imports
`torch`, `ultralytics`, or calls a hosted inference API.

```
Kaggle (GPU)                          this repo (CPU)
──────────────                        ────────────────
label data ──► train ──► export ONNX ──► models/ ──► onnxruntime ──► /v1/ocr
```

## 0. The automated path

Two one-time steps in a browser, then everything runs from here.

**One-time, in the browser:**

1. Kaggle > Settings > API > *Create New Token*, save `kaggle.json` to
   `~/.kaggle/kaggle.json`
2. On the pushed kernel: Add-ons > Secrets > add `ROBOFLOW_API_KEY`, and
   Settings > Internet **On**, Accelerator **GPU**

Secrets cannot be created through the Kaggle API, so step 2 is unavoidable —
but it is also the right place for the key. A key pasted into a notebook is
exactly how the previous one leaked.

**Then, from here:**

```bash
pip install -r requirements/dev.txt
python -m scripts.kaggle_run doctor
python -m scripts.kaggle_run push      # uploads the code, runs the notebook
python -m scripts.kaggle_run watch
python -m scripts.kaggle_run pull      # exported .onnx lands in models/
python -m scripts.check_models
```

The notebook pulls the cards from Roboflow **inside the kernel** and rectifies
them there. That is not just convenience: some networks cannot reach
Roboflow's CDN at all, while Kaggle's can.

If your own network *can* reach Roboflow, you can prepare the dataset locally
instead and upload it:

```bash
python -m scripts.roboflow_pull --workspace <ws> --project <proj> --version 8
python -m scripts.rectify_dataset --src .data/ir_card_yolo --out .data/ir_card_rectified
python -m scripts.kaggle_run upload-data --path .data/ir_card_rectified --confirm
python -m scripts.kaggle_run push --data-dataset <owner>/ir-card-yolo
```

then set `DATA_SOURCE = "kaggle-dataset"` and `RECTIFY = False` in the
notebook's first cell.

### If uploads or downloads fail with 403

Kaggle's blob endpoints and Roboflow's CDN are Google-hosted and some networks
block them intermittently. Every command here retries. Two things that helped
concretely:

- the source bundle is uploaded as **one** archive, because four separate blob
  uploads in quick succession reliably tripped a 403 partway through
- a 403 on `datasets/create/version` means the dataset does not exist yet, not
  that the upload was blocked — the tooling tells those apart by URL

`push` bundles `kyc/`, `training/` and `scripts/` into a private Kaggle dataset
and attaches it to the notebook, so the kernel trains against the same profile
definitions the serving pipeline uses — the class order cannot drift.

`upload-data` is the only command that sends card images anywhere. It prints a
summary and refuses to act without `--confirm`.

Secrets are read from disk, never from arguments: `~/.kaggle/kaggle.json` and
`ROBOFLOW_API_KEY` in `.env`. A key passed on the command line ends up in shell
history.

The rest of this document describes what those commands do, and the manual
equivalents if you prefer to run them in the Kaggle UI.

## 0.5 Smoke-test the chain before spending GPU time

The whole chain — train, export to ONNX, load through
`kyc/modules/ocr/detector.py`, run the pipeline — can be exercised on a laptop
CPU with a synthetic dataset in a few minutes:

```bash
python -m scripts.make_synthetic_dataset --count 80
python training/kaggle/train_field_detector.py \
  --data .data/synthetic/data.yaml --epochs 12 --imgsz 320 \
  --batch 8 --device cpu --out .data/smoke_models
```

The synthetic cards are not training data — they exist so a plumbing bug costs
minutes here instead of hours after a Kaggle run.

**If the run exits 0 with no epochs and no artifacts**, check the log for a
`polars` CPU-feature warning. Ultralytics 8.4 depends on polars, whose default
wheel needs AVX2 and hard-crashes without it — which looks exactly like a
successful run. Fix:

```bash
pip install "polars[rtcompat]"
```

Serving is unaffected: onnxruntime runs fine without AVX2.

## 0.7 Train on rectified cards, not on raw photos

This one is easy to get wrong and expensive to discover late.

The serving pipeline flattens the card **before** it runs the detector, so at
inference time the detector only ever sees a 1024x646 rectified card. If it was
trained on raw phone photos, the training and serving distributions disagree:
the model burns capacity on perspective it will never encounter, and boxes land
slightly off on exactly the fields that matter.

Roboflow datasets are almost always labelled on raw photos. Convert them:

```bash
python -m scripts.rectify_dataset --src .data/ir_card_yolo --out .data/ir_card_rectified
```

That runs the same rectifier the runtime uses and moves every annotation
through the identical 3x3 transform, so labels stay on their fields. One
implementation, one geometry — the dataset cannot drift from the runtime.

Then train on the rectified folder:

```bash
python -m scripts.kaggle_run upload-data --path .data/ir_card_rectified --confirm
```

Watch the report. If the card outline could not be found in more than about a
quarter of the images, inspect a few before training — the classical rectifier
struggles on busy backgrounds and borderless scans, and a mis-rectified card
teaches the detector bad boxes. Those images are resized rather than warped and
stay in the set unless you pass `--require-border`.

## 1. Field detector

Detects the card outline and the box of every field on the rectified card.

```bash
pip install ultralytics onnx onnxsim
python train_field_detector.py \
  --data /kaggle/input/<your-dataset>/data.yaml \
  --profile ir_national_card_front \
  --epochs 120 --imgsz 640
```

`data.yaml` must list classes in exactly this order, matching
`IR_CLASS_NAMES` in `kyc/modules/ocr/profiles/iran.py`:

```yaml
names:
  0: birthday
  1: exp
  2: fathername
  3: idcard
  4: idnumber
  5: img
  6: lastname
  7: name
```

The script refuses to export on a mismatch. That check is not paranoia — a
reordered class list silently writes the father's name into the national id
field, and every downstream check would still pass.

Output: `ir_national_card_front_fields.onnx` → copy into `models/`.

`yolov8n` is the right starting point. On a rectified card the task is nearly
a fixed-layout problem, so a nano model reaches high mAP and stays fast on CPU.
Only move to `yolov8s` if `img`/`idnumber` AP50 stays below ~0.95.

## 2. Text recognisers

Train two, from the same script:

```bash
# free text: names
python train_crnn.py --labels /kaggle/input/ir-lines/labels.tsv \
                     --name crnn_fa_text --epochs 60

# digits: national id and dates
python train_crnn.py --labels /kaggle/input/ir-digits/labels.tsv \
                     --name crnn_fa_digits --charset "0123456789/" --epochs 40
```

Labels are a TSV: `relative/image/path<TAB>text`, one line per crop.

Outputs, all four into `models/`:

```
crnn_fa_text.onnx     crnn_fa_text.charset.txt
crnn_fa_digits.onnx   crnn_fa_digits.charset.txt
```

Then switch the backends:

```bash
KYC_OCR_TEXT_BACKEND=crnn_onnx
KYC_OCR_DIGIT_BACKEND=crnn_onnx_digits
```

## Bootstrapping the label set

You do not need labelled crops on day one. The default `easyocr` backend runs
without any training, so:

1. Train the field detector first — that only needs box labels on whole cards.
2. Run it over your card images and save every field crop plus a guess from
   `easyocr`:

   ```bash
   python -m scripts.bootstrap_crnn_dataset --src .data/ir_card_norm --out .data/crnn_bootstrap
   ```

   Crops are cut with the exact padding and upscale settings the serving
   pipeline uses, so the CRNN trains on the same distribution it will be
   served in production. Names and dates land in separate TSVs
   (`labels_text.tsv` / `labels_digits.tsv`) next to `crops/text/` and
   `crops/digits/`.

3. Open the TSVs and correct the guessed text. Fixing OCR output is several
   times faster than transcribing from scratch — on this project's first
   real batch, clean detections like `محمد` and `روح الله` came back
   correct or nearly so, while the digit-only fields did not (see the note
   on allowlists below) and every date needed retyping. Delete the row for
   any crop that is not legible; a wrong label teaches the model a wrong
   answer more effectively than a missing one costs it.
4. Train the CRNNs on the corrected set and switch the backends:

   ```bash
   python training/kaggle/train_crnn.py --labels .data/crnn_bootstrap/labels_text.tsv \
       --name crnn_fa_text
   python training/kaggle/train_crnn.py --labels .data/crnn_bootstrap/labels_digits.tsv \
       --name crnn_fa_digits --charset "0123456789/"
   ```

### easyocr's `allowlist` is not reliable

Confirmed directly against the reader: even with `allowlist="0123456789/"`,
Arabic letters still came out of digit-only crops. This is a limitation of
easyocr itself, not of the pipeline's own masking — `crnn_onnx.py`'s decoder
masks logits before argmax, which is why the trained CRNN does not have this
problem once it exists. Until then, expect the digit-field guesses in
`labels_digits.tsv` to need full retyping rather than light correction.

## Why a separate digit model

The national id is the one field that must be exactly right, and it is
verifiable — it carries a check digit. A recogniser whose alphabet is only
`0123456789` cannot emit a letter into it, and `crnn_onnx.py` additionally
masks the logits to the allowlist so a banned symbol can never win a timestep.
The runner-up character is emitted instead of the position being dropped.

## Reproducibility

Record for each artifact, in `models/manifest.yaml`: the dataset snapshot, the
commit, the command line, and the validation metric. A KYC model that nobody
can retrace is a model nobody can defend after an incident.
