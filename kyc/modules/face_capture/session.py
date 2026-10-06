"""Capture sessions and captured selfies.

The server - not the client - decides which frame is the selfie and keeps it.
Downstream modules receive a `capture_id` and look the selfie up here, so a
client cannot substitute a different photo after passing the checks
(specs/02-face-quality.md §2).

In-memory and per-process for now. `CaptureStore` is the seam for a shared
store (Redis) once the API runs with more than one worker.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class CaptureState(str, Enum):
    SEARCHING = "SEARCHING"
    CAPTURED = "CAPTURED"
    EXPIRED = "EXPIRED"


@dataclass
class Capture:
    """A selfie the server approved. `jpeg` is the exact frame that passed."""

    capture_id: str
    session_id: str
    jpeg: bytes
    face_box: tuple[float, float, float, float]
    kps: np.ndarray
    metrics: dict[str, float]
    score: float
    warnings: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class CaptureSession:
    session_id: str
    expires_at: float
    state: CaptureState = CaptureState.SEARCHING
    frames_seen: int = 0
    ok_streak: int = 0
    # Best passing frame of the current streak: (score, jpeg, assessment).
    candidate: tuple | None = None
    capture_id: str | None = None
    last_hints: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def expired(self, now: float | None = None) -> bool:
        return (now or time.time()) >= self.expires_at


class CaptureStore:
    def __init__(self, ttl_s: int):
        self.ttl_s = ttl_s
        self._sessions: dict[str, CaptureSession] = {}
        self._captures: dict[str, Capture] = {}
        self._lock = threading.Lock()

    def create(self) -> CaptureSession:
        session = CaptureSession(session_id=secrets.token_urlsafe(16), expires_at=time.time() + self.ttl_s)
        with self._lock:
            self._reap()
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> CaptureSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def save_capture(self, capture: Capture) -> None:
        with self._lock:
            self._captures[capture.capture_id] = capture

    def get_capture(self, capture_id: str) -> Capture | None:
        with self._lock:
            capture = self._captures.get(capture_id)
        if capture and time.time() - capture.created_at > self.ttl_s * 5:
            # Captures outlive their session long enough for the case to use
            # them, but are not kept indefinitely: they are biometric data.
            with self._lock:
                self._captures.pop(capture_id, None)
            return None
        return capture

    @staticmethod
    def new_capture_id() -> str:
        return secrets.token_urlsafe(16)

    def _reap(self) -> None:
        now = time.time()
        stale = [sid for sid, s in self._sessions.items() if now - s.expires_at > self.ttl_s]
        for sid in stale:
            del self._sessions[sid]
        old = [cid for cid, c in self._captures.items() if now - c.created_at > self.ttl_s * 5]
        for cid in old:
            del self._captures[cid]
