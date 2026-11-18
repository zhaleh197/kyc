"""Synthetic name-line crops for fine-tuning the PaddleOCR recogniser.

    python -m training.ocr_synth.make_lines --out .data/ocr_synth --count 200000

Writes PaddleOCR recognition data: images/ plus train.txt / val.txt with one
`images/<name>.jpg<TAB><label>` per line. Labels are in reading (logical)
order; PaddleOCR reverses Arabic-script labels itself when its character
dict path contains "arabic" (see training/kaggle/README.md).

Each image imitates a field crop from the OCR pipeline on a phone photo of a
card: light tinted card background with faint pattern, dark ink, the
pipeline's crop margins (6% side, 15% top/bottom), the size real crops come
at (27-78 px tall), blur, perspective and JPEG. Text is mixed:

    fa   Iranian card: a given name, surname or father's name
    ckb  Iraqi Kurdistan card (Sorani): 1-3 names, own/father/grandfather
    mix  random letters, so rare letters (ڕ ڵ ۆ ێ ژ گ ...) are seen often

Nothing here reads a real card: texts come from the lists in texts/ and from
kyc/modules/ocr/lexicon/given_names_fa.txt, fonts from the system font folder.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.ocr_synth.render import covers, render_line  # noqa: E402

HERE = Path(__file__).parent
FONT_DIR = Path(r"C:\Windows\Fonts")
# Arabic-script fonts with full Persian + Sorani coverage on Windows; bold
# weights listed twice, since cards print names in bold.
FONTS = [
    "tahomabd.ttf", "tahomabd.ttf", "tahoma.ttf", "timesbd.ttf", "timesbd.ttf", "times.ttf",
    "simpbdo.ttf", "simpbdo.ttf", "simpo.ttf", "arialbd.ttf", "arial.ttf", "segoeuib.ttf",
    "segoeui.ttf", "arabtype.ttf", "majallab.ttf", "majalla.ttf", "micross.ttf", "calibrib.ttf",
]

SURNAME_ROOTS = (  # noqa: SIM905 - a word list reads better as text
    "احمد محمد حسین رضا کریم موسی جعفر کاظم رحیم صادق قاسم اکبر نور یوسف حیدر مراد محمود "
    "رستم شریف طاهر ابراهیم امیر باقر خلیل فرهاد عزیز بهرام کیان منصور خسرو شیر گل نیک "
    "پاک ژیان گودرز چراغ پیر فیروز روشن"
).split()
SURNAME_SUFFIXES = [
    "ی", "ی", "ی", "‌زاده", "زاده", "‌پور", "پور", "‌نیا", "نیا", "‌فر",
    "‌نژاد", "‌وند", "یان", "ی‌فر", "لو", "‌خانی", "‌بیگی",
]
PERSIAN_LETTERS = "آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"
SORANI_LETTERS = "ئابپتجچحخدرڕزژسشعغفڤقکگلڵمنوۆهھەیێ"


def read_list(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [s.strip() for s in lines if s.strip() and not s.lstrip().startswith("#")]


class Texts:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.given_fa = read_list(REPO_ROOT / "kyc/modules/ocr/lexicon/given_names_fa.txt")
        self.surnames_fa = read_list(HERE / "texts/surnames_fa.txt")
        self.given_ckb = read_list(HERE / "texts/given_names_ckb.txt")

    def persian(self) -> str:
        r = self.rng.random()
        if r < 0.45:
            return self.rng.choice(self.given_fa)
        if r < 0.75:
            return self.rng.choice(self.surnames_fa)
        if r < 0.92:
            return self.rng.choice(SURNAME_ROOTS) + self.rng.choice(SURNAME_SUFFIXES)
        return f"{self.rng.choice(self.surnames_fa)} {self.rng.choice(self.surnames_fa)}"  # «غریبی کانی پان»

    def sorani(self) -> str:
        words = [self.rng.choice(self.given_ckb) for _ in range(self.rng.choice((1, 1, 2, 3)))]
        text = " ".join(words)
        if self.rng.random() < 0.3:
            text = text.replace("ھ", "ه")  # both spellings of /h/ are in use
        return text

    def letters(self) -> str:
        alphabet = self.rng.choice((PERSIAN_LETTERS, SORANI_LETTERS))
        words = ["".join(self.rng.choice(alphabet) for _ in range(self.rng.randint(2, 7)))
                 for _ in range(self.rng.randint(1, 3))]
        return " ".join(words)

    def sample(self) -> tuple[str, str]:
        r = self.rng.random()
        if r < 0.5:
            return "fa", self.persian()
        if r < 0.85:
            return "ckb", self.sorani()
        return "mix", self.letters()


def label_of(text: str) -> str:
    """What the model must output: no ZWNJ (it is drawn as a non-join, not a
    character, and the model has no class for it)."""
    return " ".join(text.replace("\u200c", "").split())


# ---------------------------------------------------------------- images


def background(rng: random.Random, h: int, w: int) -> np.ndarray:
    """Pale card tint with a faint pattern, like the guilloche behind the text."""
    # Iranian card: light cyan (BGR ~ 195,185,155 in phone photos); Iraqi
    # cards and lighting vary, so the tint spreads around it.
    base = np.array(rng.choice([(195, 185, 155), (205, 195, 170), (215, 205, 185), (200, 195, 180),
                                (220, 215, 205), (205, 200, 210)]), float)
    base += np.array([rng.uniform(-10, 10) for _ in range(3)])
    img = np.ones((h, w, 3)) * base
    # gradient (uneven light)
    gx = np.linspace(rng.uniform(-18, 0), rng.uniform(0, 18), w)[None, :, None]
    gy = np.linspace(rng.uniform(-12, 0), rng.uniform(0, 12), h)[:, None, None]
    img += gx + gy
    # faint curves
    for _ in range(rng.randint(0, 5)):
        color = base + np.array([rng.uniform(-14, 10) for _ in range(3)])
        amp, freq, phase = rng.uniform(2, h / 2), rng.uniform(0.01, 0.08), rng.uniform(0, 6.3)
        y0 = rng.uniform(0, h)
        pts = np.array([[x, y0 + amp * math.sin(freq * x + phase)] for x in range(0, w, 3)], np.int32)
        cv2.polylines(img, [pts], False, color.tolist(), rng.randint(1, 3), cv2.LINE_AA)
    img += np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(1, 6), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def draw(rng: random.Random, text: str, font: Path) -> np.ndarray:
    size = rng.randint(36, 64)
    mask = render_line(text, font, size).astype(np.float32) / 255
    if rng.random() < 0.3:  # print weight: thicker / thinner ink
        k = np.ones((2, 2), np.uint8)
        mask = cv2.dilate(mask, k) if rng.random() < 0.6 else cv2.erode(mask, k)
    th, tw = mask.shape
    # The pipeline crops the detector box with 6% side and 15% top/bottom
    # margin; the detector's own box sits a little loose around the ink.
    loose_x, loose_y = rng.uniform(0.02, 0.12), rng.uniform(0.05, 0.25)
    pad_x = int(tw * (loose_x + 0.06)) + 2
    pad_y = int(th * (loose_y + 0.15)) + 2
    h, w = th + 2 * pad_y, tw + 2 * pad_x
    img = background(rng, h, w).astype(np.float32)
    ink = np.array([rng.uniform(15, 60)] * 3) + np.array([rng.uniform(-6, 6) for _ in range(3)])
    ox = pad_x + rng.randint(-pad_x // 2, pad_x // 2)
    oy = pad_y + rng.randint(-pad_y // 3, pad_y // 3)
    region = img[oy : oy + th, ox : ox + tw]
    alpha = mask[: region.shape[0], : region.shape[1], None] * rng.uniform(0.8, 1.0)
    region[:] = region * (1 - alpha) + ink * alpha
    return np.clip(img, 0, 255).astype(np.uint8)


def degrade(rng: random.Random, img: np.ndarray) -> np.ndarray:
    """A phone photo of a card: tilt, perspective, small size, blur, JPEG."""
    h, w = img.shape[:2]
    # rotation / perspective (crops come from a card photographed at an angle)
    angle = rng.gauss(0, 1.2)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.4:
        d = 0.04
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.float32([[rng.uniform(-d, d) * w, rng.uniform(-d, d) * h] for _ in range(4)])
        img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)
    # real crops are 27-78 px tall: shrink to that, as the camera saw it
    target_h = rng.randint(24, 80)
    scale = target_h / h
    img = cv2.resize(img, (max(8, int(w * scale)), target_h), interpolation=cv2.INTER_AREA)
    # blur: focus or hand shake
    r = rng.random()
    if r < 0.45:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.4))
    elif r < 0.6:
        k = rng.randint(2, 4)
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1 / k
        turn = cv2.getRotationMatrix2D((k / 2, k / 2), rng.uniform(0, 180), 1)
        img = cv2.filter2D(img, -1, cv2.warpAffine(kernel, turn, (k, k)))
    # light: brightness / contrast / a glare spot
    alpha, beta = rng.uniform(0.85, 1.15), rng.uniform(-18, 15)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    if rng.random() < 0.15:
        gh, gw = img.shape[:2]
        yy, xx = np.mgrid[0:gh, 0:gw]
        cx, cy, rad = rng.uniform(0, gw), rng.uniform(0, gh), rng.uniform(gh * 0.5, gw * 0.6)
        spot = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad**2))[..., None] * rng.uniform(30, 90)
        img = np.clip(img + spot, 0, 255).astype(np.uint8)
    # compression: phone camera, then often a messenger (Telegram) re-save
    for _ in range(rng.choice((1, 1, 2))):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(35, 92)])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def generate(out: Path, count: int, seed: int, val_fraction: float) -> None:
    rng = random.Random(seed)
    texts = Texts(rng)
    fonts = [FONT_DIR / f for f in FONTS if (FONT_DIR / f).exists()]
    if not fonts:
        raise SystemExit(f"No usable fonts found in {FONT_DIR}")
    (out / "images").mkdir(parents=True, exist_ok=True)
    train, val, kinds = [], [], {"fa": 0, "ckb": 0, "mix": 0}
    made = 0
    while made < count:
        kind, text = texts.sample()
        font = rng.choice(fonts)
        if not covers(font, text):
            continue
        label = label_of(text)
        img = degrade(rng, draw(rng, text, font))
        name = f"{made:07d}.jpg"
        cv2.imwrite(str(out / "images" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (val if rng.random() < val_fraction else train).append(f"images/{name}\t{label}")
        kinds[kind] += 1
        made += 1
        if made % 5000 == 0:
            print(f"  {made}/{count}", flush=True)
    (out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    print(f"{made} images -> {out}  (train {len(train)}, val {len(val)}; {kinds})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=REPO_ROOT / ".data/ocr_synth")
    p.add_argument("--count", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=1405)
    p.add_argument("--val-fraction", type=float, default=0.02)
    args = p.parse_args()
    generate(args.out, args.count, args.seed, args.val_fraction)


if __name__ == "__main__":
    main()
