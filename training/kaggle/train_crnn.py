"""Train a CRNN+CTC line recogniser on a Kaggle GPU and export it to ONNX.

Two models are normally trained from the same script:

    # names and other free text
    python train_crnn.py --labels /kaggle/input/ir-lines/labels.tsv \
                         --name crnn_fa_text --epochs 60

    # national id, dates - a 11-symbol alphabet
    python train_crnn.py --labels /kaggle/input/ir-digits/labels.tsv \
                         --name crnn_fa_digits --charset "0123456789/" --epochs 40

Dataset format is a TSV with two columns, `relative/image/path<TAB>text`,
relative to `--root` (defaults to the labels file's directory).

Why a separate digit model: the national id is the field that must be exactly
right, and a recogniser whose alphabet contains only digits cannot hallucinate
a letter into it. Constraining the alphabet is the cheapest accuracy win in
the whole pipeline.

Exports `<name>.onnx` plus `<name>.charset.txt`; both go into `models/`.
Index 0 is reserved for the CTC blank and is not written to the charset file -
kyc/modules/ocr/recognizers/crnn_onnx.py prepends it on load.
"""

from __future__ import annotations

import argparse
import random
import unicodedata
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - training-only dependency
    raise SystemExit("torch is required. On Kaggle it is preinstalled; locally: pip install torch") from exc

import cv2

IMG_HEIGHT = 32
IMG_WIDTH = 256
BLANK = 0


# --------------------------------------------------------------------- data


def read_labels(labels_path: Path, root: Path | None) -> list[tuple[Path, str]]:
    base = root or labels_path.parent
    items: list[tuple[Path, str]] = []
    for line_number, line in enumerate(labels_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            print(f"[warn] {labels_path.name}:{line_number} is not two tab-separated columns; skipped")
            continue
        path, text = parts
        items.append((base / path.strip(), unicodedata.normalize("NFC", text.strip())))
    return items


def build_charset(items: list[tuple[Path, str]], explicit: str | None) -> list[str]:
    if explicit:
        symbols = list(dict.fromkeys(explicit))
    else:
        symbols = sorted({ch for _, text in items for ch in text})
    return ["<blank>"] + symbols


class LineDataset(Dataset):
    """Grayscale text lines, resized to a fixed 32xW canvas.

    Rotation augmentation is deliberately wide, not mild. The field detector
    this project pairs with runs on unrectified frames (see rectify=false in
    kyc/core/config.py - the classical corner-finder fails on ~88% of this
    dataset's photos even restricted to a padded crop around the card, so
    there is no cheap fix upstream). A field crop can therefore be tilted by
    tens of degrees, and the model has to read it as-is rather than assuming
    upright print.
    """

    def __init__(
        self,
        items: list[tuple[Path, str]],
        charset: list[str],
        train: bool,
        img_height: int = IMG_HEIGHT,
        img_width: int = IMG_WIDTH,
    ):
        self.items = items
        self.index_of = {symbol: i for i, symbol in enumerate(charset)}
        self.train = train
        # Overridable, not just the module constants: a model fine-tuned from
        # EasyOCR's pretrained weights (--finetune-from) trains at their
        # height (64), not this project's from-scratch convention (32).
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, text = self.items[index]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read {path}")
        if self.train:
            image = self._augment(image)
        image = self._fit(image)
        tensor = torch.from_numpy((image.astype(np.float32) / 127.5) - 1.0).unsqueeze(0)
        targets = [self.index_of[ch] for ch in text if ch in self.index_of]
        return tensor, torch.tensor(targets, dtype=torch.long), len(targets)

    @staticmethod
    def _augment(image: np.ndarray) -> np.ndarray:
        if random.random() < 0.3:
            k = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (k, k), 0)
        if random.random() < 0.3:
            alpha = random.uniform(0.8, 1.25)
            beta = random.uniform(-25, 25)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if random.random() < 0.2:
            noise = np.random.normal(0, 6, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if random.random() < 0.5:
            # +-20 degrees, not +-1.5: real crops here come from an
            # unrectified card and are routinely tilted this much or more
            # (see the class docstring). A model trained only on near-upright
            # text would be learning a distribution it will not be served.
            angle = random.uniform(-20.0, 20.0)
            h, w = image.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h), borderValue=255)
        return image

    def _fit(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        target_h, target_w = self.img_height, self.img_width
        scale = target_h / max(h, 1)
        new_w = min(target_w, max(8, int(round(w * scale))))
        resized = cv2.resize(image, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
        if new_w < target_w:
            # White padding: the model is trained on white-background crops, so
            # black padding would read as ink.
            resized = np.hstack([resized, np.full((target_h, target_w - new_w), 255, np.uint8)])
        return resized


def collate(batch):
    images = torch.stack([b[0] for b in batch])
    targets = torch.cat([b[1] for b in batch])
    lengths = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return images, targets, lengths


# -------------------------------------------------------------------- model


class CRNN(nn.Module):
    """Classic CRNN: 7 conv layers down to a 1-pixel-high feature strip,
    then a 2-layer BiLSTM, then a per-timestep classifier.

    Width is reduced by 4, so a 256px crop yields 64 timesteps - comfortably
    more than the longest field on an ID card.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),          # 16 x W/2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),        # 8 x W/4
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),  # 4 x W/4
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),  # 2 x W/4
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),      # 1 x W/4-1
        )
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=False)
        self.head = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)                 # N, C, 1, T
        features = features.squeeze(2)         # N, C, T
        features = features.permute(2, 0, 1)   # T, N, C
        recurrent, _ = self.rnn(features)
        return self.head(recurrent)            # T, N, num_classes


# ------------------------------------------------------------ transfer learning
#
# Training a recogniser from random weights on a few hundred hand-corrected
# crops asks a lot of a model that has never seen a Persian letterform. The
# `easyocr` package (already a project dependency for the bootstrap backend)
# ships a pretrained Arabic-script recogniser - same family of architecture as
# CRNN above (ResNet feature extractor + 2-layer BiLSTM + CTC), same blank=0
# convention, and its 184-symbol charset was checked directly against this
# project's corrected labels: every character used (Persian letters, ZWNJ-free
# text, digits, '/') is already in it. So instead of training the small CRNN
# from scratch, --finetune-from starts from those weights.
#
# The one incompatible thing is input height: EasyOCR trained at 64px, this
# script's own models at 32px. kyc/modules/ocr/recognizers/crnn_onnx.py reads
# the height back out of each .onnx file's own input shape at load time, so
# both conventions serve correctly without a config flag - the ONNX graph is
# the single source of truth for that.

EASYOCR_IMG_HEIGHT = 64
EASYOCR_NETWORK_PARAMS = {"input_channel": 1, "output_channel": 512, "hidden_size": 512}


def load_easyocr_charset() -> list[str]:
    """The exact 184-symbol Arabic-script charset EasyOCR's 'arabic_g1' model
    was trained on (covers Arabic, Persian, Urdu, Uyghur). Index 0 is blank,
    matching this script's own convention - see CTCLabelConverter in the
    installed easyocr package, which builds its charset the same way."""
    try:
        from easyocr.config import recognition_models
    except ImportError as exc:
        raise SystemExit(
            "--finetune-from needs the easyocr package: pip install -r requirements/ocr.txt"
        ) from exc
    characters = recognition_models["gen1"]["arabic_g1"]["characters"]
    return ["<blank>"] + list(characters)


def check_charset_coverage(items: list[tuple[Path, str]], charset: list[str]) -> None:
    """Warn about any character the charset cannot represent.

    LineDataset silently drops characters missing from `index_of` - correct
    for tolerating the occasional OCR-guess typo the user already fixed, but
    dangerous to stay silent about here: a whole class of characters missing
    from a *pretrained* charset would quietly corrupt every label containing
    one, and nothing downstream would say so.
    """
    known = set(charset)
    missing: dict[str, int] = {}
    for _, text in items:
        for ch in text:
            if ch not in known:
                missing[ch] = missing.get(ch, 0) + 1
    if missing:
        detail = ", ".join(f"{ch!r} (U+{ord(ch):04X}) x{count}" for ch, count in sorted(missing.items()))
        print(f"[warn] {len(missing)} character(s) not in the charset and will be dropped from their labels: {detail}")


class EasyOcrModelAdapter(nn.Module):
    """Adapts EasyOCR's Model(input, text) to this script's Model(input).

    The `text` argument is accepted by EasyOCR's forward() but never read in
    its body (a leftover from a shared base supporting attention decoders
    too) - passing None through it is safe. Wrapping it here means the
    training loop, evaluate() and export_onnx() do not need to know which
    architecture produced a given checkpoint.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        # EasyOCR's own forward returns (N, T, C) - its BidirectionalLSTM
        # uses batch_first=True. This script's CRNN and its nn.CTCLoss call
        # (constructed without batch_first=True, so time-major) both expect
        # (T, N, C). Verified directly: a batch=1 smoke test cannot tell the
        # two layouts apart since N=1 makes them numerically identical, and
        # it did NOT catch this - only checking with the real training batch
        # size would have. Permuting here is the one place that needs to
        # know about the difference; everything downstream (the training
        # loop, evaluate(), export_onnx()) stays written against a single
        # (T, N, C) convention regardless of which architecture produced it.
        logits = self.inner(x, None)
        return logits.permute(1, 0, 2)


def build_easyocr_model(num_class: int) -> EasyOcrModelAdapter:
    try:
        from easyocr.model.model import Model as EasyOcrModel
    except ImportError as exc:
        raise SystemExit(
            "--finetune-from needs the easyocr package: pip install -r requirements/ocr.txt"
        ) from exc
    return EasyOcrModelAdapter(EasyOcrModel(num_class=num_class, **EASYOCR_NETWORK_PARAMS))


def load_pretrained_easyocr_weights(model: EasyOcrModelAdapter, checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        raise SystemExit(
            f"--finetune-from checkpoint not found at {checkpoint_path}.\n"
            "EasyOCR downloads it on first use of a reader for an Arabic-script language "
            "(ar/fa/ug/ur); run the bootstrap script once, or point --finetune-from at "
            "wherever it landed (usually ~/.EasyOCR/model/arabic.pth)."
        )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Trained under nn.DataParallel, so every key is prefixed "module." - our
    # (single-process, non-parallel) instance has no such wrapper.
    stripped = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.inner.load_state_dict(stripped, strict=True)


# ----------------------------------------------------------------- training


def greedy_decode(logits: torch.Tensor, charset: list[str]) -> list[str]:
    best = logits.argmax(dim=2).permute(1, 0).cpu().numpy()  # N, T
    texts = []
    for row in best:
        chars, previous = [], -1
        for index in row:
            index = int(index)
            if index != previous and index != BLANK:
                chars.append(charset[index])
            previous = index
        texts.append("".join(chars))
    return texts


def evaluate(model, loader, charset, device) -> tuple[float, float]:
    """Returns (exact-match accuracy, mean normalised edit distance)."""
    model.eval()
    exact = total = 0
    distance_sum = 0.0
    inverse = dict(enumerate(charset))
    with torch.no_grad():
        for images, targets, lengths in loader:
            logits = model(images.to(device))
            predictions = greedy_decode(logits, charset)
            offset = 0
            for prediction, length in zip(predictions, lengths.tolist(), strict=True):
                truth = "".join(inverse[int(t)] for t in targets[offset:offset + length])
                offset += length
                total += 1
                exact += int(prediction == truth)
                distance_sum += _levenshtein(prediction, truth) / max(1, len(truth))
    model.train()
    return (exact / max(1, total)), (distance_sum / max(1, total))


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def export_onnx(model, charset: list[str], out_dir: Path, name: str, opset: int, img_height: int = IMG_HEIGHT) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    dummy = torch.zeros(1, 1, img_height, IMG_WIDTH)
    onnx_path = out_dir / f"{name}.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["logits"],
        opset_version=opset,
        # Static shapes only: the serving side letterboxes to 32xIMG_WIDTH and
        # a fixed graph is meaningfully faster on CPU.
        dynamic_axes=None,
        # Recent torch defaults to the dynamo-based exporter, which needs the
        # separate `onnxscript` package. The classic TorchScript exporter
        # handles this small conv+LSTM model fine and has no extra dependency.
        dynamo=False,
    )
    charset_path = out_dir / f"{name}.charset.txt"
    # Index 0 is the CTC blank and is re-added at load time.
    charset_path.write_text("\n".join(charset[1:]), encoding="utf-8")
    print(f"[done] {onnx_path}")
    print(f"[done] {charset_path}  ({len(charset) - 1} symbols)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", required=True, help="TSV of 'relative/path<TAB>text'")
    parser.add_argument("--root", default=None, help="Image root; defaults to the labels file's directory")
    parser.add_argument("--name", default="crnn_fa_text", help="Artifact base name; must match the backend id")
    parser.add_argument("--charset", default=None, help="Explicit alphabet, e.g. '0123456789/'")
    parser.add_argument("--epochs", type=int, default=60)
    # Small on purpose: this script's whole point is training on the hand-
    # corrected bootstrap sets from scripts/bootstrap_crnn_dataset.py, which
    # run a few hundred rows, not tens of thousands. batch=64 against ~400
    # examples gave only ~6 optimiser steps per epoch - too few for a
    # from-scratch CTC model to escape predicting nothing but blank.
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--lr", type=float, default=None, help="Default 3e-4 from scratch, 1e-4 with --finetune-from"
    )
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--out", default="/kaggle/working/models")
    parser.add_argument("--device", default=None, help="'cuda', 'cpu', or unset to auto-detect")
    parser.add_argument(
        "--finetune-from",
        default=None,
        help="Path to EasyOCR's pretrained arabic.pth to start from instead of random "
        "weights (usually ~/.EasyOCR/model/arabic.pth). Overrides --charset with EasyOCR's "
        "own 184-symbol charset and trains at their 64px input height.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    items = read_labels(Path(args.labels), Path(args.root) if args.root else None)
    if not items:
        raise SystemExit("No labelled samples found")
    random.shuffle(items)
    split = max(1, int(len(items) * args.val_split))
    val_items, train_items = items[:split], items[split:]

    finetuning = args.finetune_from is not None
    img_height = EASYOCR_IMG_HEIGHT if finetuning else IMG_HEIGHT
    lr = args.lr if args.lr is not None else (1e-4 if finetuning else 3e-4)

    if finetuning:
        if args.charset:
            print("[warn] --charset is ignored with --finetune-from; using EasyOCR's own charset")
        charset = load_easyocr_charset()
        check_charset_coverage(items, charset)
    else:
        charset = build_charset(items, args.charset)
    print(f"[data] {len(train_items)} train / {len(val_items)} val, {len(charset) - 1} symbols")

    # Auto-detection alone is not safe on Kaggle: it has handed out Tesla
    # P100s (sm_60) while its preinstalled torch only supports sm_70+, so
    # cuda.is_available() reports True right up until the first kernel
    # launch fails. --device lets the caller (the notebook, which already
    # checks torch.cuda.get_device_capability against torch.cuda.get_arch_list)
    # force cpu after making that check; left unset, this still auto-detects
    # for plain local runs.
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[data] training on {device}" + (f", fine-tuning from {args.finetune_from}" if finetuning else ""))

    # num_workers=0: a multi-worker DataLoader hung indefinitely (12h+, zero
    # output past the first warning, worker processes alive but never
    # delivering a batch) when this ran locally on Windows from a background
    # process. The dataset here is at most a few thousand small grayscale
    # PNGs - loading is not the bottleneck on either Kaggle or a local CPU
    # box, so the marginal parallel-loading speedup is not worth the
    # reliability risk that cost 12 hours to notice.
    train_loader = DataLoader(
        LineDataset(train_items, charset, train=True, img_height=img_height),
        # drop_last=True was silently discarding up to a full batch of the
        # (already small) training set every epoch - a proportionally large
        # loss of data when the whole set is a few hundred rows.
        batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(
        LineDataset(val_items, charset, train=False, img_height=img_height),
        batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=0,
    )

    if finetuning:
        model = build_easyocr_model(len(charset))
        load_pretrained_easyocr_weights(model, Path(args.finetune_from))
        model = model.to(device)
    else:
        model = CRNN(len(charset)).to(device)
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Selected by lowest edit distance, not highest exact-match accuracy.
    # Exact match is a harsh, discrete signal - on a small validation set it
    # can sit at literal 0.0 for many epochs while the model is genuinely
    # improving, and a checkpoint saved only on *strict* accuracy gains then
    # freezes on the first (near-random) epoch and never updates again. Edit
    # distance moves continuously and actually reflects that improvement.
    best_edit = float("inf")
    best_accuracy_at_best_edit = 0.0
    out_dir = Path(args.out)
    checkpoint = out_dir / f"{args.name}.best.pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for images, targets, lengths in train_loader:
            images = images.to(device)
            logits = model(images)
            log_probs = logits.log_softmax(2)
            input_lengths = torch.full((images.size(0),), logits.size(0), dtype=torch.long)
            loss = criterion(log_probs, targets, input_lengths, lengths)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss)
        scheduler.step()

        accuracy, edit = evaluate(model, val_loader, charset, device)
        print(f"[{epoch:3d}/{args.epochs}] loss={running / max(1, len(train_loader)):.4f} "
              f"exact={accuracy:.4f} edit={edit:.4f}")
        if edit < best_edit:
            best_edit = edit
            best_accuracy_at_best_edit = accuracy
            torch.save(model.state_dict(), checkpoint)

    print(f"[best] edit distance {best_edit:.4f}  (exact-match {best_accuracy_at_best_edit:.4f} at that checkpoint)")
    if not checkpoint.exists():
        raise SystemExit("No checkpoint was ever saved - training produced no usable model.")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    export_onnx(model, charset, out_dir, args.name, args.opset, img_height=img_height)


if __name__ == "__main__":
    main()
