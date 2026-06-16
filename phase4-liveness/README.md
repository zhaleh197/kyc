# Liveness Detection API

FastAPI wrapper around a MediaPipe Face Mesh active liveness check
(blink, smile, look straight, turn left/right, lean forward/back).
Each session gets 4 random tasks; the client must complete each within
a time limit (default 7s) by streaming camera frames to the API.

## Run

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Docs at `http://localhost:8000/docs`.

## Flow

1. **Create a session**

   ```bash
   curl -X POST http://localhost:8000/sessions
   ```

   ```json
   {
     "session_id": "a1b2c3...",
     "task_list": ["BLINK", "TURN_LEFT", "SMILE", "GO_FORWARD"],
     "task_instructions": {
       "BLINK": "Blink",
       "TURN_LEFT": "Turn head LEFT",
       "SMILE": "Smile wide",
       "GO_FORWARD": "Move your head forward"
     }
   }
   ```

2. **Stream frames** (e.g. every 100-200ms from the browser webcam) as base64:

   ```bash
   curl -X POST http://localhost:8000/sessions/$SESSION_ID/frame \
     -H "Content-Type: application/json" \
     -d "{\"image_base64\": \"$(base64 -w0 frame.jpg)\"}"
   ```

   Or as a multipart upload:

   ```bash
   curl -X POST http://localhost:8000/sessions/$SESSION_ID/frame/upload \
     -F "file=@frame.jpg"
   ```

   Response:

   ```json
   {
     "state": "RUNNING",
     "message": "Next: Turn head LEFT",
     "current_task": "TURN_LEFT",
     "current_task_instruction": "Turn head LEFT",
     "task_index": 1,
     "total_tasks": 4,
     "completed_tasks": 1,
     "time_remaining": 6.8,
     "task_list": ["BLINK", "TURN_LEFT", "SMILE", "GO_FORWARD"],
     "is_live": false
   }
   ```

3. **Terminal states**: `state` becomes `VERIFIED` (`is_live: true`) once all
   4 tasks pass, or `FAILED` if a task times out or the face is lost while
   running. At that point, discard the session (or call `/reset` for a
   fresh challenge).

4. **Cleanup**: `DELETE /sessions/{id}` ends a session early. Idle sessions
   are automatically reaped after 2 minutes.

## Notes / production considerations

- **State**: sessions are held in-memory per process. Scale horizontally
  with sticky sessions, or swap the store for Redis + a stateless detector
  call (would need to make `LivenessSession` serializable or move state to
  the client and re-derive per request — non-trivial given MediaPipe's
  FaceMesh is a heavy native object).
- **Concurrency**: each session owns its own `FaceMesh` instance; processing
  is CPU-bound and synchronous. For high concurrency, run multiple Uvicorn
  workers behind a load balancer, or offload frame processing to a process
  pool.
- **Security**: this is liveness-check-only, not a substitute for full
  identity verification or anti-spoofing (no depth camera / IR check —
  determined attackers can spoof with video replay or 3D masks). Pair with
  document verification and server-side rate limiting per session_id/IP.
- **CORS**: locked open (`*`) for dev — restrict `allow_origins` in
  production.
- **Frame size**: capped at 5MB per request; tune `MAX_FRAME_BYTES` as needed.
