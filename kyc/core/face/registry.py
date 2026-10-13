"""One loaded instance per model file, shared by every module in the process.

Face capture, face match, anti-spoof and liveness all use the same buffalo_l
and MiniFASNet files. Loading them once per module tripled the memory - and
the dev machine ran out of it ("bad allocation" from onnxruntime) more than
once. onnxruntime sessions are safe to call from several threads.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Hashable
from typing import TypeVar

T = TypeVar("T")

_instances: dict[Hashable, object] = {}
_lock = threading.Lock()


def shared(key: Hashable, factory: Callable[[], T]) -> T:
    """The instance for `key`, created by `factory` on first use."""
    with _lock:
        if key not in _instances:
            _instances[key] = factory()
        return _instances[key]  # type: ignore[return-value]


def load_shared(cls, path, *args, providers: list[str], intra_threads: int = 0):
    """`cls.load(path, *args, providers, intra_threads)`, shared per
    (class, file, args, providers)."""
    key = (cls.__qualname__, str(path), args, tuple(providers), intra_threads)
    return shared(key, lambda: cls.load(path, *args, providers, intra_threads))
