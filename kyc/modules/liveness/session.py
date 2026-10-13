"""Liveness sessions, in memory per process (the same seam as face capture's
CaptureStore for a shared store later)."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from kyc.core.schemas import Reason

from .tasks import Baseline, FrameMeasure, TaskState


class LivenessState(str, Enum):
    BASELINE = "BASELINE"  # collecting frontal frames for the person's baseline
    RUNNING = "RUNNING"  # doing the tasks
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


TERMINAL = (LivenessState.VERIFIED, LivenessState.FAILED, LivenessState.EXPIRED)


@dataclass
class LivenessSession:
    session_id: str
    capture_id: str
    selfie_embedding: np.ndarray
    tasks: list[TaskState]
    expires_at: float
    state: LivenessState = LivenessState.BASELINE
    started_at: float = field(default_factory=time.time)
    task_index: int = 0
    task_started_at: float = 0.0
    task_durations: list[float] = field(default_factory=list)
    baseline_samples: list[FrameMeasure] = field(default_factory=list)
    baseline: Baseline | None = None
    last_face_at: float = field(default_factory=time.time)
    frames_seen: int = 0
    identity: list[float] = field(default_factory=list)
    live_scores: list[float] = field(default_factory=list)
    failure: Reason | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def current_task(self) -> TaskState | None:
        return self.tasks[self.task_index] if self.task_index < len(self.tasks) else None


class LivenessStore:
    def __init__(self, ttl_s: int):
        self.ttl_s = ttl_s
        self._sessions: dict[str, LivenessSession] = {}
        self._lock = threading.Lock()

    def create(self, capture_id: str, selfie_embedding: np.ndarray, tasks: list[TaskState]) -> LivenessSession:
        session = LivenessSession(
            session_id=secrets.token_urlsafe(16), capture_id=capture_id, selfie_embedding=selfie_embedding,
            tasks=tasks, expires_at=time.time() + self.ttl_s,
        )
        with self._lock:
            now = time.time()
            for sid in [sid for sid, s in self._sessions.items() if now - s.expires_at > self.ttl_s]:
                del self._sessions[sid]
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> LivenessSession | None:
        with self._lock:
            return self._sessions.get(session_id)
