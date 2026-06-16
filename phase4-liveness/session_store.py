"""
In-memory session store for liveness checks.

This is intentionally simple (single-process, in-memory). For production
with multiple workers/instances, replace with a shared store (Redis, etc.)
and externalize the MediaPipe processing, or pin sessions to a single
worker via sticky routing — FaceMesh objects are not easily serializable.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field

from .detector import LivenessSession

SESSION_TTL_SECONDS = 120  # idle sessions are reaped after this


@dataclass
class _Entry:
    session: LivenessSession
    last_seen: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, ttl_seconds: int = SESSION_TTL_SECONDS):
        self._sessions: dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def create(self, task_time_limit: float | None = None) -> tuple[str, LivenessSession]:
        session_id = uuid.uuid4().hex
        session = LivenessSession(task_time_limit=task_time_limit)
        with self._lock:
            self._sessions[session_id] = _Entry(session=session)
        return session_id, session

    def get(self, session_id: str) -> LivenessSession | None:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return None
            entry.last_seen = time.time()
            return entry.session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            entry = self._sessions.pop(session_id, None)
        if entry is not None:
            entry.session.close()
            return True
        return False

    def reap_expired(self):
        """Close and remove sessions idle longer than the TTL."""
        now = time.time()
        with self._lock:
            expired = [
                sid for sid, entry in self._sessions.items()
                if now - entry.last_seen > self._ttl
            ]
            for sid in expired:
                self._sessions[sid].session.close()
                del self._sessions[sid]
        return expired


# Module-level singleton used by the API
store = SessionStore()
