"""Pull a labelled card dataset out of Roboflow and normalise it for training.

    python -m scripts.roboflow_pull --project idcardsegmentation-gz97d --version 8

Roboflow exports YOLO classes in its own order, which is almost never the
order the document profile declares. This script downloads the export, reads
the class names out of Roboflow's own `data.yaml`, and remaps every annotation
into the profile's order - so the artifact that reaches Kaggle is already
correct and `train_field_detector.py` will not refuse it.

The API key is read from the environment (or `.env`), never from an argument:
a key on the command line ends up in your shell history.

    ROBOFLOW_API_KEY=...        in E:/1405/kyc/.env

Generate a NEW key first. The key that appears in this repository's notebook
history is compromised and must be revoked from the Roboflow dashboard.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kyc.modules.ocr.profiles import get_profile  # noqa: E402
from scripts.prepare_dataset import build_remap, find_pairs, parse_label, write_data_yaml  # noqa: E402

# Roboflow names the validation split "valid"; ultralytics and our own
# data.yaml call it "val". Both are accepted as input so that re-running
# this over an already-normalised dataset does not silently drop the
# validation set - which is exactly what happened once, and ultralytics
# then refused the data.yaml for a missing "val:" key.
SPLITS = ("train", "valid", "val", "test")
SPLIT_ALIASES = {"valid": "val"}


class MissingExport(RuntimeError):
    """The version's export file is gone from Roboflow's storage.

    Permanent, so it must not be retried - only a regeneration in the
    dashboard, or a different version, can fix it.
    """


# Roboflow builds exports on demand; large versions take a while.
EXPORT_TIMEOUT = 900
EXPORT_POLL_SECONDS = 10

# Both spellings are accepted: ROBOFLOW_API_KEY is what Roboflow's own docs
# use, ROBOFLOW_KEY is what people tend to type.
KEY_NAMES = ("ROBOFLOW_API_KEY", "ROBOFLOW_KEY")


def load_api_key() -> str:
    key = next((os.environ[n] for n in KEY_NAMES if os.environ.get(n)), None)
    if not key:
        # Minimal .env reader: pydantic-settings is for the app, not for a CLI.
        env_file = REPO_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                name, _, value = line.partition("=")
                if name.strip() in KEY_NAMES:
                    key = value.strip().strip('"').strip("'")
                    break
    if not key:
        raise SystemExit(
            "\n".join(
                [
                    "ROBOFLOW_API_KEY is not set.",
                    f"Put it in {REPO_ROOT / '.env'} yourself - do not pass it as an argument:",
                    "    ROBOFLOW_API_KEY=your-new-key",
                    "",
                    "Generate a NEW key. The one in this repo's notebook history is",
                    "compromised and should be revoked in the Roboflow dashboard.",
                ]
            )
        )
    return key


def fetch_export(workspace: str, project: str, version: int, fmt: str, dest: Path, key: str) -> Path:
    """Download and unpack one Roboflow export, without the SDK in the way.

    The SDK wraps the two-step flow (ask the API for a signed link, then fetch
    it from Google storage) and reports any failure as a bare `BadZipFile`,
    having already written the error page to disk as `roboflow.zip`. That hid
    the real cause through several rounds of debugging - the response was an
    HTML error, and nothing said so.

    Doing it directly means an unexpected response is quoted in the exception
    instead of being mistaken for a corrupt archive.
    """
    import zipfile

    import requests

    dest.mkdir(parents=True, exist_ok=True)

    # Roboflow builds an export lazily. Requesting the format endpoint queues
    # the build and answers `ready: false` with a progress fraction; the signed
    # link it returns meanwhile points at an object that does not exist yet.
    # Downloading without waiting is what produced "NoSuchKey" from storage,
    # which the SDK then surfaced as a corrupt-zip error.
    endpoint = f"https://api.roboflow.com/{workspace}/{project}/{version}/{fmt}"
    deadline = time.time() + EXPORT_TIMEOUT
    link = None

    while time.time() < deadline:
        meta = requests.get(endpoint, params={"api_key": key}, timeout=90)

        # 202 means the export does not exist yet and this request queued it.
        # Treating it as an error - which is what a naive `!= 200` check does -
        # turns "wait a minute" into "this dataset cannot be downloaded".
        if meta.status_code == 202:
            print(f"[rf] Roboflow is generating the {fmt} export for v{version}; waiting")
            time.sleep(EXPORT_POLL_SECONDS)
            continue
        if meta.status_code != 200:
            raise RuntimeError(f"export endpoint returned HTTP {meta.status_code}: {summarise(meta.text)}")

        payload = meta.json()
        export = payload.get("export") or {}
        if export.get("ready") is False or payload.get("ready") is False:
            progress = export.get("progress", payload.get("progress", 0.0)) or 0.0
            print(f"[rf] Roboflow is still building the {fmt} export: {progress * 100:.0f}%")
            time.sleep(EXPORT_POLL_SECONDS)
            continue

        link = export.get("link")
        break

    if not link:
        available = (meta.json().get("version") or {}).get("exports")
        raise RuntimeError(
            f"no usable download link for format {fmt!r} within {EXPORT_TIMEOUT}s; "
            f"formats listed for this version: {available}"
        )

    response = requests.get(link, timeout=600, stream=True, allow_redirects=True)
    if response.status_code == 404 and "NoSuchKey" in response.text:
        # The API happily hands out a signed link for an export whose object is
        # gone from storage. Retrying cannot fix that, and six attempts against
        # a permanent 404 only buries the real message.
        raise MissingExport(
            f"Roboflow lists a {fmt} export for v{version} but the file is missing from its storage. "
            f"Regenerate that version in the Roboflow dashboard (Versions > v{version} > Download), "
            f"or use a version whose export still exists."
        )
    if response.status_code != 200:
        raise RuntimeError(f"storage returned HTTP {response.status_code}: {summarise(response.text)}")

    archive = dest / "roboflow.zip"
    with archive.open("wb") as handle:
        for chunk in response.iter_content(1 << 20):
            handle.write(chunk)

    if not zipfile.is_zipfile(archive):
        head = archive.read_bytes()[:400]
        content_type = response.headers.get("content-type", "?")
        raise RuntimeError(
            f"downloaded {archive.stat().st_size} bytes of {content_type}, not a zip. "
            f"Response began: {summarise(head.decode('utf-8', 'replace'))}"
        )

    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(dest)
    archive.unlink()
    return dest


def summarise(text: str, limit: int = 220) -> str:
    """Collapse an HTML or JSON error body to one readable line."""
    import re

    stripped = re.sub(r"<[^>]+>", " ", text or "")
    return " ".join(stripped.split())[:limit] or "(empty response)"


def download(
    workspace: str | None,
    project: str,
    version: int,
    fmt: str,
    dest: Path,
    attempts: int = 6,
    backoff: float = 5.0,
) -> Path:
    """Fetch the export, retrying transient network failures.

    Roboflow's version and storage endpoints are reached through a CDN that,
    from some networks, intermittently answers 403 or drops the TLS handshake
    on requests that succeed moments later. Retrying is the difference between
    "this does not work here" and "this takes two minutes".
    """
    key = load_api_key()
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            print(f"[rf] downloading {workspace}/{project}:v{version} as {fmt} (attempt {attempt})")
            location = fetch_export(workspace, project, version, fmt, dest, key)
            print(f"[rf] downloaded to {location}")
            return location
        except MissingExport as exc:
            raise SystemExit(f"[rf] {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - requests raises a wide family here
            last_error = exc
            summary = " ".join(str(exc).split())[:120]
            print(f"[rf] attempt {attempt}/{attempts} failed: {type(exc).__name__}: {summary}")
            if attempt < attempts:
                time.sleep(backoff * attempt)

    raise SystemExit(
        "\n".join(
            [
                f"Could not download {workspace}/{project}:v{version} after {attempts} attempts.",
                f"Last error: {type(last_error).__name__}: {str(last_error)[:200]}",
                "",
                "If the failures are 403 pages from Google, the CDN is blocking this",
                "network intermittently. Two ways round it:",
                "  - re-run; the block is not consistent and often clears",
                "  - or let the Kaggle kernel pull the dataset instead, where the",
                "    network is unrestricted (see training/kaggle/README.md)",
            ]
        )
    )


def read_roboflow_classes(location: Path) -> list[str]:
    import yaml

    data_yaml = location / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"No data.yaml in the export at {location}")
    spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = spec.get("names")
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    if isinstance(names, list):
        return list(names)
    raise SystemExit(f"Could not read class names from {data_yaml}")


def normalise(location: Path, out: Path, profile_id: str) -> int:
    """Copy the Roboflow export into `out`, remapped to the profile's order."""
    profile = get_profile(profile_id)
    class_names = profile.class_names
    source_names = read_roboflow_classes(location)

    print(f"[map] roboflow : {source_names}")
    print(f"[map] profile  : {list(class_names)}")

    extra = [n for n in source_names if n not in class_names]
    if extra:
        raise SystemExit(
            f"Roboflow classes not present in the profile: {extra}\n"
            f"Either rename them in Roboflow, or add them to class_names in the profile module."
        )
    remap = build_remap(source_names, class_names)
    if remap == {i: i for i in range(len(source_names))}:
        print("[map] orders already agree; copying through")
    else:
        print(f"[map] remapping {dict(sorted(remap.items()))}")

    if out.exists():
        shutil.rmtree(out)

    written: dict[str, int] = {}
    counts: dict[str, int] = dict.fromkeys(class_names, 0)

    for split in SPLITS:
        split_dir = location / split
        if not split_dir.is_dir():
            continue
        target_split = SPLIT_ALIASES.get(split, split)
        if target_split in written:
            continue  # both spellings present; the first one wins
        (out / target_split / "images").mkdir(parents=True)
        (out / target_split / "labels").mkdir(parents=True)

        pairs = find_pairs(split_dir)
        for image, label in pairs:
            shutil.copy2(image, out / target_split / "images" / image.name)
            lines = []
            if label is not None:
                for class_id, rest in parse_label(label):
                    mapped = remap.get(class_id)
                    if mapped is None:
                        continue
                    counts[class_names[mapped]] += 1
                    lines.append(f"{mapped} {rest}")
            body = "\n".join(lines)
            (out / target_split / "labels" / f"{image.stem}.txt").write_text(
                body + "\n" if body else "", encoding="utf-8"
            )
        written[target_split] = len(pairs)

    if not written:
        raise SystemExit(f"No train/valid/val/test folders found in {location}")

    write_data_yaml(out, class_names, list(written))

    print(f"\n[done] {out}")
    for split, count in written.items():
        print(f"  {split:<6} {count:5d} images")
    print("\n  annotations per class:")
    problems = 0
    for name in class_names:
        flag = "   <-- NONE" if counts[name] == 0 else ""
        print(f"    {name:<14} {counts[name]:6d}{flag}")
    missing = [n for n, c in counts.items() if c == 0]
    if missing:
        print(f"\n[error] no annotations at all for: {missing}")
        print("        The detector cannot learn a class it never sees.")
        problems = 1

    print("\nNext:")
    print("  python -m scripts.kaggle_run upload-data --path " + str(out))
    print("  python -m scripts.kaggle_run push --data-dataset <owner>/<slug>")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=None, help="Roboflow workspace id; defaults to the key owner")
    # Not required with --skip-download: normalising an export you already
    # have on disk needs no project coordinates.
    parser.add_argument("--project", default=None, help="Roboflow project id")
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--format", default="yolov8", help="Roboflow export format")
    parser.add_argument("--profile", default="ir_national_card_front")
    parser.add_argument("--raw", default=str(REPO_ROOT / ".data" / "roboflow_raw"), help="Where the export lands")
    parser.add_argument("--out", default=str(REPO_ROOT / ".data" / "ir_card_yolo"), help="Normalised dataset")
    parser.add_argument("--skip-download", action="store_true", help="Reuse an export already in --raw")
    args = parser.parse_args()
    if not args.skip_download and (not args.project or args.version is None):
        raise SystemExit("--project and --version are required unless --skip-download is given")

    raw = Path(args.raw)
    if args.skip_download:
        if not raw.exists():
            raise SystemExit(f"--skip-download given but {raw} does not exist")
        location = raw
        print(f"[rf] reusing {location}")
    else:
        raw.parent.mkdir(parents=True, exist_ok=True)
        location = download(args.workspace, args.project, args.version, args.format, raw)

    return normalise(location, Path(args.out), args.profile)


if __name__ == "__main__":
    raise SystemExit(main())
