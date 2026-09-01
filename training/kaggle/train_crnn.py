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

    Augmentation is deliberately mild: these crops come from a rectified card,
    so the model does not need to learn perspective, only print variation,
    sensor noise and mild blur.
    """

    def __init__(self, items: list[tuple[Path, str]], charset: list[str], train: bool):
        self.items = items
        self.index_of = {symbol: i for i, symbol in enumerate(charset)}
        self.train = train

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
        if random.random() < 0.2:
            angle = random.uniform(-1.5, 1.5)
            h, w = image.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h), borderValue=255)
        return image

    @staticmethod
    def _fit(image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        scale = IMG_HEIGHT / max(h, 1)
        new_w = min(IMG_WIDTH, max(8, int(round(w * scale))))
        resized = cv2.resize(image, (new_w, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
        if new_w < IMG_WIDTH:
            # White padding: the model is trained on white-background crops, so
            # black padding would read as ink.
            resized = np.hstack([resized, np.full((IMG_HEIGHT, IMG_WIDTH - new_w), 255, np.uint8)])
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


def export_onnx(model, charset: list[str], out_dir: Path, name: str, opset: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    dummy = torch.zeros(1, 1, IMG_HEIGHT, IMG_WIDTH)
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
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--out", default="/kaggle/working/models")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    items = read_labels(Path(args.labels), Path(args.root) if args.root else None)
    if not items:
        raise SystemExit("No labelled samples found")
    random.shuffle(items)
    split = max(1, int(len(items) * args.val_split))
    val_items, train_items = items[:split], items[split:]

    charset = build_charset(items, args.charset)
    print(f"[data] {len(train_items)} train / {len(val_items)} val, {len(charset) - 1} symbols")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[data] training on {device}")

    train_loader = DataLoader(
        LineDataset(train_items, charset, train=True),
        batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=2, drop_last=True,
    )
    val_loader = DataLoader(
        LineDataset(val_items, charset, train=False),
        batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=2,
    )

    model = CRNN(len(charset)).to(device)
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_accuracy = -1.0
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
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), checkpoint)

    print(f"[best] exact-match accuracy {best_accuracy:.4f}")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    export_onnx(model, charset, out_dir, args.name, args.opset)


if __name__ == "__main__":
    main()
