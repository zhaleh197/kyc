# Model artifacts

This directory is **not** in version control (see `.gitignore`). Model weights
are build outputs: they are large, they change on a different cadence than the
code, and a repository that carries them becomes impossible to clone.

Put the exported files here and record each one in `manifest.yaml`.

## What the OCR module looks for

| File | Required | Produced by |
|---|---|---|
| `ir_national_card_front_fields.onnx` | yes | `training/kaggle/train_field_detector.py` |
| `crnn_fa_text.onnx` + `crnn_fa_text.charset.txt` | only with `KYC_OCR_TEXT_BACKEND=crnn_onnx` | `training/kaggle/train_crnn.py` |
| `crnn_fa_digits.onnx` + `crnn_fa_digits.charset.txt` | only with `KYC_OCR_DIGIT_BACKEND=crnn_onnx_digits` | `training/kaggle/train_crnn.py` |

Without the field detector, `/v1/ocr/document` returns HTTP 503 with
`model_not_available` and a message naming the missing file. That is deliberate:
an identity pipeline that silently degrades is worse than one that refuses.

## Verifying a drop

```bash
python -m scripts.check_models
```

Reports which artifacts are present, their sha256, and whether the detector's
output width matches the profile's class count.
