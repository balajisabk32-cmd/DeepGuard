"""FastAPI service — Role 3's T+0 deliverable (plan §2).

MOCK-FIRST BY DESIGN. This returns a hard-coded but contract-valid FusionResult so
Role 4 can build the dashboard against a real shape from T+0:30 without waiting for
any ML. Swap in real modules behind a feature flag at T+4 (plan §5, Phase 1).

CRITICAL when the real modules land (plan §1 Role 3, risk row 6):
CPU-bound work must NOT run on the event loop. MediaPipe + scipy inside an
`async def` freezes the WebSocket progress updates — the one failure a judge sees.
Use a ProcessPoolExecutor with one FaceLandmarker per worker.
"""

from __future__ import annotations

import os
import uuid

from fastapi import FastAPI, HTTPException

from src.common.contracts import FusionResult
from src.fusion import score

app = FastAPI(title="DeepGuard", version="2.1")

# Feature flag: flip to "1" once real modules are wired (plan §5, 4:00–5:30).
USE_REAL_MODULES = os.getenv("DEEPGUARD_REAL_MODULES", "0") == "1"

_SESSIONS: dict[str, FusionResult] = {}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "real_modules": USE_REAL_MODULES}


@app.post("/upload")
def upload() -> dict:
    """Session IDs are generated SERVER-SIDE (plan §8.4).

    Never accept a client-supplied path component: it flows into
    data/sessions/{id}/ and is a path-traversal vector.
    """
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


@app.post("/analyze/{session_id}")
def analyze(session_id: str) -> FusionResult:
    if USE_REAL_MODULES:  # pragma: no cover - wired at T+4
        raise NotImplementedError("Real pipeline lands in Phase 1, 4:00–5:30")

    result = score(
        modality_scores={"rppg": 0.12, "lipsync": 0.18},
        modality_quality={"rppg": 0.90, "lipsync": 0.80},
        session_id=session_id,
        total_processing_time_ms=1250,
    )
    _SESSIONS[session_id] = result
    return result


@app.get("/result/{session_id}")
def result(session_id: str) -> FusionResult:
    if session_id not in _SESSIONS:
        raise HTTPException(status_code=404, detail="Unknown session")
    return _SESSIONS[session_id]
