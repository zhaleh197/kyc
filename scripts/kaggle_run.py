"""Drive the Kaggle training run from here: push, watch, pull.

    python -m scripts.kaggle_run doctor
    python -m scripts.kaggle_run push --data-dataset <owner>/<your-card-dataset>
    python -m scripts.kaggle_run watch
    python -m scripts.kaggle_run pull

`push` uploads **source code only** - `kyc/`, `training/`, `scripts/` - as a
private Kaggle dataset, then pushes a notebook that trains against a card
dataset *you* have already uploaded. This script never uploads identity
documents; putting real ID cards on a third-party service is your decision to
make deliberately, not a side effect of running a helper.

Requires a Kaggle API token at `~/.kaggle/kaggle.json` (Kaggle > Settings >
API > Create New Token). Run `doctor` first - it checks everything and tells
you exactly what is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SRC_DATASET_SLUG = "kyc-src"          # must match SRC_DATASET in the notebook
KERNEL_SLUG = "kyc-field-detector"
NOTEBOOK = REPO_ROOT / "training" / "kaggle" / "train_field_detector.ipynb"
STATE_FILE = REPO_ROOT / ".kaggle_run.json"

# Only these paths are bundled. Everything else - datasets, models, the venv -
# stays local.
SOURCE_PATHS = ["kyc", "training", "scripts", "pyproject.toml"]

# Kaggle reports these upper-cased ("ERROR"), the SDK sometimes camel-cased.
# Compared case-insensitively so the watch loop actually terminates.
TERMINAL_STATES = {"complete", "error", "cancelacknowledged", "cancelrequested"}


# ------------------------------------------------------------------ plumbing


def load_api():
    """Authenticate the Kaggle client, with a readable failure.

    Credentials are checked *before* importing kaggle: the package calls
    authenticate() at import time and raises a bare OSError, which would bury
    the actual instruction under a stack trace.
    """
    token = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    if not token.exists() and not has_env:
        raise SystemExit(
            "\n".join(
                [
                    f"No Kaggle credentials found at {token}",
                    "Put them there yourself - this script never asks you to paste a key:",
                    "  1. kaggle.com -> your avatar -> Settings -> API -> Create New Token",
                    "  2. save the downloaded kaggle.json to the path above",
                    "  3. re-run this command",
                ]
            )
        )

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise SystemExit("The kaggle package is not installed:  pip install kaggle") from None
    except OSError as exc:
        raise SystemExit(f"Kaggle rejected the credentials at {token}: {exc}") from None

    api = KaggleApi()
    api.authenticate()
    return api


def username(api) -> str:
    name = os.environ.get("KAGGLE_USERNAME")
    if name:
        return name
    config = getattr(api, "config_values", {}) or {}
    name = config.get("username")
    if not name:
        raise SystemExit("Could not determine the Kaggle username from your credentials.")
    return name


def read_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def write_state(**values) -> None:
    state = read_state()
    state.update(values)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def with_retries(action, label: str, attempts: int = 5, backoff: float = 6.0):
    """Retry a Kaggle API call that the network may intermittently block.

    Kaggle's blob uploads terminate on Google-hosted endpoints, which from some
    networks answer 403 on a request that succeeds moments later. The SDK's own
    retry does not cover this, so a single 403 aborts a push that would work on
    the next try.
    """
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except Exception as exc:  # noqa: BLE001 - the SDK raises bare HTTPError
            last_error = exc
            summary = " ".join(str(exc).split())[:120]
            print(f"[retry] {label} attempt {attempt}/{attempts} failed: {type(exc).__name__}: {summary}")
            if attempt < attempts:
                time.sleep(backoff * attempt)
    raise SystemExit(
        "\n".join(
            [
                f"{label} failed after {attempts} attempts.",
                f"Last error: {type(last_error).__name__}: {str(last_error)[:200]}",
                "",
                "If these are 403s, this network is intermittently blocking Kaggle's",
                "upload endpoint. Re-running usually gets through; the block is not",
                "consistent.",
            ]
        )
    )


def normalise_state(value) -> str:
    """Reduce whatever kernels_status returns to a bare lowercase state.

    The SDK has returned a plain string, a camel-cased string and an enum
    (`KernelWorkerStatus.ERROR`) across versions. Comparing the raw value meant
    the watch loop never recognised a terminal state and polled until the
    network gave out.
    """
    return str(value).rsplit(".", 1)[-1].strip().strip('"').lower()


def wait_for_dataset(api, dataset_id: str, timeout: int) -> None:
    """Block until Kaggle reports the dataset version as ready.

    Kaggle unpacks an uploaded archive asynchronously. Attaching a kernel to a
    version that is still processing mounts an empty directory, and the
    notebook then quietly falls back to whatever other source it can find -
    which is how a run once trained against stale code.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            status = str(api.dataset_status(dataset_id)).strip().strip('"').lower()
        except Exception as exc:  # noqa: BLE001 - status is best-effort
            print(f"[push] dataset status unavailable ({type(exc).__name__}); continuing")
            return
        if status == "ready":
            print("[push] dataset version is ready")
            return
        if status == "error":
            raise SystemExit(f"Kaggle reported an error processing {dataset_id}")
        print(f"[push] dataset status: {status}; waiting")
        time.sleep(5)
    print(f"[warn] dataset {dataset_id} still not ready after {timeout}s; pushing anyway")


def is_missing_dataset_error(exc: Exception) -> bool:
    """True when Kaggle refused a *version* call because the dataset is new.

    Kaggle returns 403 for both "this dataset does not exist" and "your upload
    was blocked", so the status code alone cannot decide. The endpoint in the
    URL can: datasets/create/version means the dataset is missing.
    """
    message = str(exc)
    return "datasets/create/version" in message and ("403" in message or "404" in message)


def bundle_sources(target: Path, staging: Path) -> None:
    """Stage the code the kernel needs, then upload it as ONE archive.

    Kaggle's per-folder zipping would make this four separate blob uploads, and
    the upload endpoint starts answering 403 after a few in quick succession.
    A single archive is one blob, so the whole push either works or fails once.
    Kaggle expands the zip when the dataset is attached, so the kernel still
    sees a normal directory tree.
    """
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "runs", "*.onnx", "*.pt")
    tree = staging / "kyc-src"
    tree.mkdir(parents=True)

    for name in SOURCE_PATHS:
        source = REPO_ROOT / name
        if not source.exists():
            print(f"[warn] {name} does not exist; skipped")
            continue
        destination = tree / name
        if source.is_dir():
            shutil.copytree(source, destination, ignore=ignore)
        else:
            shutil.copy2(source, destination)

    archive = shutil.make_archive(str(staging / "kyc-src"), "zip", root_dir=tree)
    shutil.move(archive, target / "kyc-src.zip")
    size = (target / "kyc-src.zip").stat().st_size
    print(f"[push] bundled {len(SOURCE_PATHS)} paths into one archive ({size / 1024:.0f} KB)")


# ------------------------------------------------------------------ commands


def cmd_doctor(args: argparse.Namespace) -> int:
    # importlib, not `import kaggle`: the kaggle package calls authenticate()
    # at import time and raises before we can print a useful message.
    from importlib.util import find_spec

    print("kaggle package     :", end=" ")
    if find_spec("kaggle") is None:
        print("MISSING  ->  pip install kaggle")
        return 1
    print("installed")

    api = load_api()
    print("credentials        : ok")
    print("username           :", username(api))
    print("notebook           :", "found" if NOTEBOOK.exists() else f"MISSING at {NOTEBOOK}")

    state = read_state()
    if state:
        print("last push          :", json.dumps(state, indent=2))
    else:
        print("last push          : none yet")

    print(
        "\nNext: upload your labelled card dataset to Kaggle yourself, then\n"
        "  python -m scripts.kaggle_run push --data-dataset <owner>/<slug>"
    )
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    if not NOTEBOOK.exists():
        raise SystemExit(f"Notebook not found at {NOTEBOOK}")

    api = load_api()
    user = username(api)
    src_id = f"{user}/{SRC_DATASET_SLUG}"
    kernel_id = f"{user}/{KERNEL_SLUG}"

    with tempfile.TemporaryDirectory() as tmp:
        # --- 1. source code as a private dataset -------------------------
        src_dir = Path(tmp) / "src"
        src_dir.mkdir()
        staging = Path(tmp) / "staging"
        staging.mkdir()
        bundle_sources(src_dir, staging)
        (src_dir / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "title": "KYC source",
                    "id": src_id,
                    "licenses": [{"name": "other"}],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"[push] source dataset {src_id}")

        def upload_source():
            try:
                api.dataset_create_version(str(src_dir), version_notes=args.notes, dir_mode="skip", quiet=False)
                print("       new version created")
            except Exception as exc:  # noqa: BLE001 - the SDK raises bare HTTPError
                # Kaggle answers 403, not 404, when you version a dataset that
                # does not exist yet. Telling that apart from a blocked upload
                # is what the URL is for: a failure on the *version* endpoint
                # means "create it first", one on /blobs/upload means the
                # network dropped it and retrying is the right move.
                if is_missing_dataset_error(exc):
                    print("       dataset does not exist yet; creating it")
                    api.dataset_create_new(str(src_dir), public=False, dir_mode="skip", quiet=False)
                    print("       created")
                else:
                    raise

        with_retries(upload_source, "source dataset upload", attempts=args.attempts)

        # Dataset processing is asynchronous. A kernel attached to a version
        # that is still being unpacked mounts nothing, and the notebook then
        # sees no source at all - so wait for "ready" rather than guessing with
        # a fixed sleep.
        wait_for_dataset(api, src_id, args.dataset_wait)

        # --- 2. the training kernel ---------------------------------------
        kernel_dir = Path(tmp) / "kernel"
        kernel_dir.mkdir()
        shutil.copy2(NOTEBOOK, kernel_dir / NOTEBOOK.name)

        sources = [src_id]
        if args.data_dataset:
            sources.append(args.data_dataset)
        else:
            print("[warn] no --data-dataset given; attach your card dataset in the Kaggle UI before running")

        (kernel_dir / "kernel-metadata.json").write_text(
            json.dumps(
                {
                    "id": kernel_id,
                    "title": "KYC field detector",
                    "code_file": NOTEBOOK.name,
                    "language": "python",
                    "kernel_type": "notebook",
                    "is_private": True,
                    "enable_gpu": True,
                    # pip install ultralytics needs network access. Kaggle only
                    # grants it to phone-verified accounts.
                    "enable_internet": True,
                    "dataset_sources": sources,
                    "competition_sources": [],
                    "kernel_sources": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"[push] kernel {kernel_id}")
        with_retries(lambda: api.kernels_push(str(kernel_dir)), "kernel push", attempts=args.attempts)

    write_state(kernel=kernel_id, src_dataset=src_id, data_dataset=args.data_dataset, pushed_at=time.time())
    print(f"\n[done] https://www.kaggle.com/code/{kernel_id}")
    print("Edit DATA_YAML in the first cell to point at your dataset, then:")
    print("  python -m scripts.kaggle_run watch")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    api = load_api()
    kernel = args.kernel or read_state().get("kernel")
    if not kernel:
        raise SystemExit("No kernel known. Pass --kernel <owner>/<slug> or run `push` first.")

    print(f"watching {kernel} (Ctrl-C to stop; the run continues on Kaggle)")

    # A kernel keeps its previous terminal status until the new run is
    # scheduled, so polling immediately after a push reads the *last* run and
    # reports a failure that has not happened. Give the queue a moment to move
    # off that stale state before believing it.
    if args.settle:
        first = normalise_state(with_retries(lambda: api.kernels_status(kernel), "status check", attempts=4))
        if first in TERMINAL_STATES:
            print(f"  initial status is {first}; waiting {args.settle}s in case this is the previous run")
            time.sleep(args.settle)

    while True:
        status = with_retries(lambda: api.kernels_status(kernel), "status check", attempts=4)
        state = status.get("status") if isinstance(status, dict) else getattr(status, "status", str(status))
        message = status.get("failureMessage") if isinstance(status, dict) else getattr(status, "failureMessage", "")
        stamp = time.strftime("%H:%M:%S")
        print(f"  [{stamp}] {state}" + (f" - {message}" if message else ""))
        if normalise_state(state) in TERMINAL_STATES:
            if normalise_state(state) == "complete":
                print("\nRun finished. Fetch the model:  python -m scripts.kaggle_run pull")
                return 0
            print(f"\nRun ended in state {state}. Open the kernel on Kaggle for the full log.")
            return 1
        time.sleep(args.interval)


def cmd_pull(args: argparse.Namespace) -> int:
    api = load_api()
    kernel = args.kernel or read_state().get("kernel")
    if not kernel:
        raise SystemExit("No kernel known. Pass --kernel <owner>/<slug> or run `push` first.")

    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        print(f"[pull] downloading output of {kernel}")
        with_retries(lambda: api.kernels_output(kernel, path=tmp), "output download", attempts=args.attempts)
        moved = 0
        for path in Path(tmp).rglob("*"):
            if path.is_file() and path.suffix in {".onnx", ".txt"} and "models" in path.parts:
                shutil.copy2(path, destination / path.name)
                print(f"       {path.name} -> {destination / path.name}")
                moved += 1
        if moved == 0:
            print(f"[warn] no model artifacts found in the kernel output (looked under {tmp})")
            return 1

    print("\nVerify the drop:  python -m scripts.check_models")
    return 0


def cmd_upload_data(args: argparse.Namespace) -> int:
    """Upload a prepared card dataset to Kaggle as a PRIVATE dataset.

    Deliberately gated behind --confirm and a dry-run summary. This is the one
    command in the project that sends identity documents to a third party, and
    that should never happen as an incidental side effect.
    """
    source = Path(args.path)
    if not (source / "data.yaml").exists():
        raise SystemExit(f"{source} does not look like a prepared dataset (no data.yaml)")

    files = [p for p in source.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)
    images = sum(1 for p in files if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})

    print(f"source : {source}")
    print(f"files  : {len(files)} ({images} images)")
    print(f"size   : {total_bytes / 1e6:.1f} MB")
    print("target : Kaggle, PRIVATE dataset")
    print("\nThis uploads identity-document images to a third-party service.")
    print("They will be private, but they leave this machine and Kaggle's retention")
    print("policy applies. Make sure that is what you intend.")

    if not args.confirm:
        print("\nDry run. Re-run with --confirm to actually upload.")
        return 0

    api = load_api()
    user = username(api)
    dataset_id = f"{user}/{args.slug}"

    metadata = source / "dataset-metadata.json"
    metadata.write_text(
        json.dumps({"title": args.title, "id": dataset_id, "licenses": [{"name": "other"}]}, indent=2),
        encoding="utf-8",
    )
    try:
        print(f"[upload] {dataset_id}")
        def upload():
            try:
                api.dataset_create_version(str(source), version_notes=args.notes, dir_mode="zip", quiet=False)
                print("         new version created")
            except Exception as exc:  # noqa: BLE001 - the SDK raises bare HTTPError
                if is_missing_dataset_error(exc):
                    print("         dataset does not exist yet; creating it")
                    api.dataset_create_new(str(source), public=False, dir_mode="zip", quiet=False)
                else:
                    raise

        with_retries(upload, "dataset upload", attempts=args.attempts)
    finally:
        metadata.unlink(missing_ok=True)

    write_state(data_dataset=dataset_id)
    print(f"\n[done] https://www.kaggle.com/datasets/{dataset_id}")
    print(f"Next:  python -m scripts.kaggle_run push --data-dataset {dataset_id}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="Check credentials and setup").set_defaults(func=cmd_doctor)

    upload = subparsers.add_parser("upload-data", help="Upload a prepared card dataset (private)")
    upload.add_argument("--path", required=True, help="Folder produced by prepare_dataset or roboflow_pull")
    upload.add_argument("--slug", default="ir-card-yolo")
    upload.add_argument("--title", default="Iranian national card fields")
    upload.add_argument("--notes", default="automated upload")
    upload.add_argument("--confirm", action="store_true", help="Actually upload; without it this is a dry run")
    upload.add_argument("--attempts", type=int, default=5, help="Retries when the network blocks an upload")
    upload.set_defaults(func=cmd_upload_data)

    push = subparsers.add_parser("push", help="Upload the source and push the training notebook")
    push.add_argument("--data-dataset", default=None, help="Your card dataset as <owner>/<slug>")
    push.add_argument("--notes", default="automated push", help="Dataset version notes")
    push.add_argument("--dataset-wait", type=int, default=300, help="Seconds to wait for the dataset to become ready")
    push.add_argument("--attempts", type=int, default=5, help="Retries per upload when the network blocks one")
    push.set_defaults(func=cmd_push)

    watch = subparsers.add_parser("watch", help="Poll the kernel until it finishes")
    watch.add_argument("--kernel", default=None)
    watch.add_argument("--interval", type=int, default=30)
    watch.add_argument(
        "--settle",
        type=int,
        default=45,
        help="Seconds to keep waiting when the very first poll already reports a terminal state, "
        "since that is usually the previous run's status. 0 disables.",
    )
    watch.set_defaults(func=cmd_watch)

    pull = subparsers.add_parser("pull", help="Download the exported model into models/")
    pull.add_argument("--kernel", default=None)
    pull.add_argument("--out", default=str(REPO_ROOT / "models"))
    pull.add_argument("--attempts", type=int, default=5, help="Retries when the network drops the download")
    pull.set_defaults(func=cmd_pull)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
