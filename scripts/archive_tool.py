"""Manage the encrypted case archive (kyc/modules/orchestrator/archive.py).

    python -m scripts.archive_tool key                      # new KYC_ARCHIVE_KEY (keep it secret)
    python -m scripts.archive_tool list   [--repo user/kyc-archive]
    python -m scripts.archive_tool show CASE_ID --out DIR   [--repo user/kyc-archive]
    python -m scripts.archive_tool purge  [--repo user/kyc-archive]

Without --repo it works on the local archive (KYC_ARCHIVE_DIR). `show`
decrypts one case - record.json, card.jpg, selfie.jpg - into DIR; it needs
KYC_ARCHIVE_KEY in the environment or .env. With --repo it reads the private
Hugging Face dataset using your logged-in token (`huggingface-cli login`).
Decrypted files are identity data: delete them when done.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kyc.core.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["key", "list", "show", "purge"])
    parser.add_argument("case_id", nargs="?")
    parser.add_argument("--repo", default=None, help="private HF dataset repo, e.g. user/kyc-archive")
    parser.add_argument("--out", type=Path, default=Path(".data/archive_view"))
    args = parser.parse_args()

    if args.command == "key":
        from cryptography.fernet import Fernet

        print(Fernet.generate_key().decode())
        return

    s = get_settings()
    if args.repo:
        from huggingface_hub import HfApi

        api = HfApi()
        files = [f for f in api.list_repo_files(args.repo, repo_type="dataset") if f.startswith("cases/")]
    else:
        root = s.archive_dir
        stored = (root / "cases").rglob("*.enc") if (root / "cases").exists() else []
        files = [p.relative_to(root).as_posix() for p in stored]

    cases = sorted({tuple(f.split("/")[1:3]) for f in files if f.count("/") >= 3})
    if args.command == "list":
        for day, case_id in cases:
            print(day, case_id)
        print(f"{len(cases)} case(s)", file=sys.stderr)
        return

    if args.command == "purge":
        from kyc.modules.orchestrator.archive import HFDatasetArchive, LocalArchive

        archive = (HFDatasetArchive(args.repo, s.archive_key, s.retention_days) if args.repo
                   else LocalArchive(s.archive_dir, s.archive_key, s.retention_days))
        print(f"deleted {archive.purge()} day folder(s) older than {s.retention_days} days")
        return

    # show
    if not args.case_id:
        raise SystemExit("show needs a CASE_ID")
    match = [(d, c) for d, c in cases if c == args.case_id]
    if not match:
        raise SystemExit(f"case {args.case_id} not found")
    day = match[0][0]
    from cryptography.fernet import Fernet

    fernet = Fernet(s.archive_key.encode())
    out = args.out / args.case_id
    out.mkdir(parents=True, exist_ok=True)
    for f in [f for f in files if f"/{args.case_id}/" in f]:
        name = f.rsplit("/", 1)[-1].removesuffix(".enc")
        if args.repo:
            from huggingface_hub import hf_hub_download

            blob = Path(hf_hub_download(args.repo, f, repo_type="dataset")).read_bytes()
        else:
            blob = (s.archive_dir / f).read_bytes()
        (out / name).write_bytes(fernet.decrypt(blob))
    print(f"decrypted case {args.case_id} ({day}) into {out}")


if __name__ == "__main__":
    main()
