"""
FastAPI service for MediaPipe-based active liveness detection.

Flow:
  POST /sessions              -> create a session, get session_id + 4-task challenge
  POST /sessions/{id}/frame   -> submit a frame (multipart or base64 JSON), get status
  POST /sessions/{id}/reset   -> regenerate a fresh challenge for the session
  DELETE /sessions/{id}       -> end session early, free resources
  GET  /healthz               -> liveness probe for the service itself
"""

import asyncio
import base64
import binascii
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .detector import TASK_INSTRUCTIONS
from .session_store import store

logger = logging.getLogger("liveness_api")

MAX_FRAME_BYTES = 5 * 1024 * 1024  # 5 MB guard against oversized uploads


@asynccontextmanager
async def lifespan(app: FastAPI):
    reaper = asyncio.create_task(_session_reaper())
    try:
        yield
    finally:
        reaper.cancel()


async def _session_reaper():
    """Background task: periodically clear out idle sessions."""
    while True:
        await asyncio.sleep(30)
        try:
            expired = store.reap_expired()
            if expired:
                logger.info("Reaped %d expired session(s)", len(expired))
        except Exception:
            logger.exception("Session reaper failed")


app = FastAPI(
    title="Liveness Detection API",
    description="Active liveness detection via MediaPipe Face Mesh (blink, smile, head turn, etc.)",
    version="1.0.0",
    lifespan=lifespan,
)

# Adjust for your deployment — wide open here for ease of integration during dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class CreateSessionRequest(BaseModel):
    task_time_limit: float | None = Field(
        default=None,
        ge=1.0,
        le=60.0,
        description="Seconds allowed per task. Defaults to server setting (7s) if omitted.",
    )


class CreateSessionResponse(BaseModel):
    session_id: str
    task_list: list[str]
    task_instructions: dict[str, str]


class FrameRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded JPEG/PNG frame")


class FrameResponse(BaseModel):
    state: str
    message: str
    current_task: str | None
    current_task_instruction: str | None
    task_index: int
    total_tasks: int
    completed_tasks: int
    time_remaining: float | None
    task_list: list[str]
    is_live: bool


class SimpleMessage(BaseModel):
    detail: str


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _decode_frame(raw: bytes) -> np.ndarray:
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image payload")
    if len(raw) > MAX_FRAME_BYTES:
        raise HTTPException(status_code=413, detail="Frame exceeds maximum size of 5MB")

    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image data")
    return frame


def _get_session_or_404(session_id: str):
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return session


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/healthz", response_model=SimpleMessage)
def healthz():
    return {"detail": "ok"}


@app.post("/sessions", response_model=CreateSessionResponse, status_code=201)
def create_session(payload: CreateSessionRequest | None = None):
    task_time_limit = payload.task_time_limit if payload else None
    session_id, session = store.create(task_time_limit=task_time_limit)
    return CreateSessionResponse(
        session_id=session_id,
        task_list=session.task_list,
        task_instructions={t: TASK_INSTRUCTIONS[t] for t in session.task_list},
    )


@app.post("/sessions/{session_id}/frame", response_model=FrameResponse)
def submit_frame_json(session_id: str, payload: FrameRequest):
    session = _get_session_or_404(session_id)

    try:
        raw = base64.b64decode(payload.image_base64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    frame = _decode_frame(raw)
    result = session.process_frame(frame)
    return result


@app.post("/sessions/{session_id}/frame/upload", response_model=FrameResponse)
async def submit_frame_upload(session_id: str, file: UploadFile = File(...)):
    session = _get_session_or_404(session_id)
    raw = await file.read()
    frame = _decode_frame(raw)
    result = session.process_frame(frame)
    return result


@app.post("/sessions/{session_id}/reset", response_model=CreateSessionResponse)
def reset_session(session_id: str):
    session = _get_session_or_404(session_id)
    session.reset()
    return CreateSessionResponse(
        session_id=session_id,
        task_list=session.task_list,
        task_instructions={t: TASK_INSTRUCTIONS[t] for t in session.task_list},
    )


@app.delete("/sessions/{session_id}", response_model=SimpleMessage)
def delete_session(session_id: str):
    if not store.delete(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"detail": "session deleted"}
