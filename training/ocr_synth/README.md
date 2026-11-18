# Synthetic name lines for the PaddleOCR recogniser

`make_lines.py` draws field-crop-like images of Persian (Iranian card) and
Kurdish Sorani (Iraqi Kurdistan card) name lines, in PaddleOCR recognition
format, for fine-tuning `models/ppocr_arabic_v5_rec.onnx`.

```bash
pip install -r requirements/dev.txt
python -m training.ocr_synth.make_lines --out .data/ocr_synth --count 200000
```

Why: PP-OCRv5 Arabic was never trained on Persian-only or Kurdish letters,
although its character list has them. Measured on 900 synthetic lines with
the current model (2026-11-19):

| line | read exactly |
|---|---|
| Persian, no «گ ژ پ چ» | 88% (real cards: ~86% - the synthetic lines are about as hard) |
| Persian with «گ ژ پ چ» | 39% |
| Kurdish Sorani | 0% - lost most: ە ێ ڕ ڵ ۆ |

The character list already holds every Persian and Sorani letter, so the
fine-tune keeps the model's dictionary and output layer as they are.

- Text: `texts/surnames_fa.txt`, `texts/given_names_ckb.txt`,
  `kyc/modules/ocr/lexicon/given_names_fa.txt`, surnames built from roots +
  suffixes, and random-letter lines so rare letters are seen often.
- Shaping: HarfBuzz (`render.py`); fonts: the Windows system fonts that
  cover both alphabets (Tahoma, Times, Simplified Arabic, Arial, Segoe UI,
  Arabic Typesetting, Sakkal Majalla, ...). The fonts stay on this machine;
  only the rendered images are uploaded for training.
- Look: tuned against real crops - light cyan card tint, faint guilloche,
  dark ink, the pipeline's crop margins, 24-80 px tall, blur, tilt, glare,
  one or two JPEG passes (phone, then messenger).
- Labels are in reading order. PaddleOCR reverses Arabic-script labels itself
  when the dictionary path contains "arabic"; check this in the notebook
  before training.
