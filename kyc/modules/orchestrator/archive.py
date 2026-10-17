"""Archive of finalized cases: card photo, selfie and verdict, encrypted,
kept `retention_days` and then deleted.

Layout (both backends): cases/<YYYY-MM-DD>/<case_id>/{record.json,card.jpg,selfie.jpg}.enc
Every file is encrypted with Fernet (AES-128-CBC + HMAC) under
KYC_ARCHIVE_KEY, so the storage provider - a private Hugging Face dataset on
the hosted test service - never holds readable identity data. Purging
deletes whole date folders older than the retention period.

scripts/archive_tool.py makes a key, lists the archive and decrypts a case.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Protocol

from kyc.core.config import Settings
from kyc.core.errors import KycError

logger = logging.getLogger("kyc.archive")


class ArchiveMisconfigured(KycError):
    code = "archive_misconfigured"
    http_status = 500


class CaseArchive(Protocol):
    def save(self, case_id: str, day: str, record: dict, files: dict[str, bytes]) -> None: ...
    def purge(self, now: dt.date | None = None) -> int: ...


def _fernet(key: str):
    from cryptography.fernet import Fernet

    if not key:
        raise ArchiveMisconfigured("KYC_ARCHIVE_KEY is required to archive cases (scripts/archive_tool.py key)")
    return Fernet(key.encode())


def _encrypted(fernet, record: dict, files: dict[str, bytes]) -> dict[str, bytes]:
    out = {"record.json.enc": fernet.encrypt(json.dumps(record, ensure_ascii=False).encode("utf-8"))}
    for name, data in files.items():
        out[f"{name}.enc"] = fernet.encrypt(data)
    return out


def _expired_days(days: list[str], retention_days: int, now: dt.date) -> list[str]:
    cutoff = now - dt.timedelta(days=retention_days)
    out = []
    for day in days:
        try:
            if dt.date.fromisoformat(day) < cutoff:
                out.append(day)
        except ValueError:
            continue  # not a date folder - leave it alone
    return out


class LocalArchive:
    def __init__(self, root: Path, key: str, retention_days: int):
        self.root, self.fernet, self.retention_days = Path(root) / "cases", _fernet(key), retention_days

    def save(self, case_id: str, day: str, record: dict, files: dict[str, bytes]) -> None:
        folder = self.root / day / case_id
        folder.mkdir(parents=True, exist_ok=True)
        for name, data in _encrypted(self.fernet, record, files).items():
            (folder / name).write_bytes(data)

    def purge(self, now: dt.date | None = None) -> int:
        if not self.root.exists():
            return 0
        days = _expired_days([p.name for p in self.root.iterdir() if p.is_dir()], self.retention_days,
                             now or dt.date.today())
        for day in days:
            shutil.rmtree(self.root / day, ignore_errors=True)
        return len(days)


class HFDatasetArchive:
    """A private Hugging Face dataset repo, written with the Space's HF_TOKEN."""

    def __init__(self, repo_id: str, key: str, retention_days: int, token: str | None = None):
        if not repo_id:
            raise ArchiveMisconfigured("KYC_ARCHIVE_HF_REPO is required for the hf archive")
        from huggingface_hub import HfApi

        self.repo_id, self.fernet, self.retention_days = repo_id, _fernet(key), retention_days
        self.api = HfApi(token=token)
        self.api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    def save(self, case_id: str, day: str, record: dict, files: dict[str, bytes]) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for name, data in _encrypted(self.fernet, record, files).items():
                (Path(tmp) / name).write_bytes(data)
            self.api.upload_folder(folder_path=tmp, path_in_repo=f"cases/{day}/{case_id}", repo_id=self.repo_id,
                                   repo_type="dataset", commit_message=f"case {case_id}")

    def purge(self, now: dt.date | None = None) -> int:
        try:
            entries = self.api.list_repo_tree(self.repo_id, path_in_repo="cases", repo_type="dataset")
            days = [e.path.split("/")[-1] for e in entries]
        except Exception:  # noqa: BLE001 - no "cases" folder yet
            return 0
        expired = _expired_days(days, self.retention_days, now or dt.date.today())
        for day in expired:
            self.api.delete_folder(f"cases/{day}", repo_id=self.repo_id, repo_type="dataset",
                                   commit_message=f"retention: delete {day}")
        return len(expired)


def build_archive(settings: Settings) -> CaseArchive | None:
    backend = settings.archive_backend
    if backend == "off":
        return None
    if backend == "local":
        return LocalArchive(settings.archive_dir, settings.archive_key, settings.retention_days)
    import os

    return HFDatasetArchive(settings.archive_hf_repo, settings.archive_key, settings.retention_days,
                            token=os.environ.get("HF_TOKEN"))


class RetentionClock:
    """Runs purge() at most once per `interval_s`, from whoever calls tick()."""

    def __init__(self, archive: CaseArchive, interval_s: float = 6 * 3600):
        self.archive, self.interval_s, self._last = archive, interval_s, 0.0

    def tick(self) -> None:
        if time.time() - self._last < self.interval_s:
            return
        self._last = time.time()
        try:
            removed = self.archive.purge()
            if removed:
                logger.info("retention: deleted %d day folder(s)", removed)
        except Exception:  # noqa: BLE001 - retention must never break a verification
            logger.exception("retention purge failed")
