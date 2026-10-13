"""Liveness tasks: what each one asks, and how a stream of measurements
completes it. Pure logic over per-frame measurements - no model - so every
rule is unit-testable with hand-made numbers.

Every task is a MOVEMENT relative to the person's own baseline (taken from
their first frontal frames), never a static pose: a photo can "look
straight" or show closed eyes, but cannot blink, turn from frontal, smile
or come closer on command.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kyc.core.config import LivenessSettings

INSTRUCTIONS_FA = {
    "BASELINE": "مستقیم به دوربین نگاه کنید",
    # "for a second": frames arrive ~1/s, and a normal blink (0.1-0.4 s)
    # usually falls between two of them.
    "BLINK": "چشم‌هایتان را یک ثانیه ببندید، بعد باز کنید",
    "TURN_LEFT": "سرتان را به سمت چپ بچرخانید",
    "TURN_RIGHT": "سرتان را به سمت راست بچرخانید",
    "SMILE": "لبخند بزنید",
    "MOVE_CLOSER": "گوشی را به صورتتان نزدیک‌تر کنید",
}
INSTRUCTIONS_EN = {
    "BASELINE": "Look straight at the camera",
    "BLINK": "Close your eyes for a second, then open them",
    "TURN_LEFT": "Turn your head to the left",
    "TURN_RIGHT": "Turn your head to the right",
    "SMILE": "Smile",
    "MOVE_CLOSER": "Bring the phone closer to your face",
}


@dataclass(frozen=True)
class FrameMeasure:
    """What liveness needs from one frame with exactly one face."""

    yaw: float
    eye_openness: float  # mean of both eyes (EAR)
    mouth_width: float  # mouth corner distance / eye distance
    face_height: float  # px


@dataclass
class Baseline:
    eye_openness: float
    mouth_width: float
    face_height: float


@dataclass
class TaskState:
    name: str
    seen_closed: bool = False  # BLINK
    started_frontal: bool = False  # TURN_*
    done: bool = False
    notes: dict = field(default_factory=dict)


def is_frontal(m: FrameMeasure, cfg: LivenessSettings) -> bool:
    return abs(m.yaw) <= cfg.frontal_max_yaw_deg


def advance(task: TaskState, m: FrameMeasure, base: Baseline, cfg: LivenessSettings) -> bool:
    """Feed one frame to the current task; True once it is completed."""
    if task.name == "BLINK":
        if m.eye_openness < cfg.blink_closed_ratio * base.eye_openness:
            task.seen_closed = True
        elif task.seen_closed and m.eye_openness >= cfg.blink_open_ratio * base.eye_openness:
            task.done = True
    elif task.name in ("TURN_LEFT", "TURN_RIGHT"):
        # Must start from frontal: a head already turned (or a photo tilted
        # once) does not count.
        if is_frontal(m, cfg):
            task.started_frontal = True
        toward_left = m.yaw > 0 if cfg.turn_left_is_positive_yaw else m.yaw < 0
        wanted = toward_left if task.name == "TURN_LEFT" else not toward_left
        if task.started_frontal and wanted and abs(m.yaw) >= cfg.turn_yaw_deg:
            task.done = True
    elif task.name == "SMILE":
        # Mouth width is only comparable when the head faces the camera.
        if is_frontal(m, cfg) and m.mouth_width >= cfg.smile_width_ratio * base.mouth_width:
            task.done = True
    elif task.name == "MOVE_CLOSER":
        if m.face_height >= cfg.closer_size_ratio * base.face_height:
            task.done = True
    else:
        raise ValueError(f"unknown liveness task {task.name!r}")
    task.notes = {"yaw": round(m.yaw, 1), "eyes": round(m.eye_openness, 3), "mouth": round(m.mouth_width, 3)}
    return task.done
