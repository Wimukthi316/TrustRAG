"""TrustRAG FastAPI backend.

Run from the repo root (so `src` is importable):
    .venv\\Scripts\\python.exe -m uvicorn backend.app.main:app --reload --port 8000

Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root on sys.path so `from src.common.schema import ...` resolves whether
# uvicorn is launched from the root or from backend/.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from backend.app.services.detector import get_detector  # noqa: E402
from src.common.schema import (  # noqa: E402
    SCHEMA_VERSION,
    AnalysisResult,
    AnalyzeRequest,
    HealthResponse,
)

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the detector before accepting traffic.

    The first forward pass costs about 7 seconds on the RTX 3050 -- weights,
    CUDA context, kernel autotuning -- and every one after it costs about 44 ms.
    Paying that on startup instead of on the first click is the difference
    between a demo that feels instant and one the panel thinks is broken.

    Never fatal: a server that refuses to start because warmup failed is worse
    than a server whose first request is slow.
    """
    detector = get_detector()
    warmup = getattr(detector, "warmup", None)
    if callable(warmup):
        try:
            print(f"warming {detector.model_version} ...", flush=True)
            # Through the threadpool rather than on the event loop, because
            # warmup blocks for several seconds and the loop should not.
            #
            # Measured, so that nobody repeats the investigation: warmup takes
            # the first request from about 9,300 ms down to about 350 ms, and
            # every request after it runs at 41-47 ms on the RTX 3050. A
            # stubborn ~300 ms remains on the very first request and its cause
            # is NOT identified. Two hypotheses were tested and both were
            # wrong: warming more input lengths did not help, and neither did
            # warming inside this threadpool on the theory that CUDA
            # initialises per thread. Left as is -- 350 ms once is not worth
            # more time when the deadline is the binding constraint.
            from starlette.concurrency import run_in_threadpool

            elapsed = await run_in_threadpool(warmup)
            print(f"detector warm in {elapsed:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"warmup failed ({exc}); the first request will be slow")
    yield


app = FastAPI(
    lifespan=lifespan,
    title="TrustRAG API",
    description=(
        "Span-level calibrated hallucination detection for RAG. "
        "Which detector serves is set by TRUSTRAG_DETECTOR; unset means the "
        "placeholder, whose scores are NOT model output. Check "
        "/api/health -- detector_loaded is false whenever the stub is serving."
    ),
    version="0.1.0",
)

# Vite dev server ports. Tighten this before any public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    det = get_detector()
    return HealthResponse(
        status="ok",
        schema_version=SCHEMA_VERSION,
        model_version=det.model_version,
        detector_loaded=det.model_version != "stub-v0",
    )


@app.post("/api/analyze", response_model=AnalysisResult)
def analyze(req: AnalyzeRequest) -> AnalysisResult:
    """Analyse one (context, question, answer) triple.

    Returns every span the detector considered non-trivial, each tagged with a
    conformal decision at the requested alpha.
    """
    try:
        return get_detector().analyze(req)
    except ValueError as exc:
        # Contract violations (bad offsets, text mismatch) surface here.
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/example", response_model=AnalyzeRequest)
def example() -> AnalyzeRequest:
    """A canned request so the frontend has a 'Load example' button.

    Hand-written illustration, not a RAGTruth record.
    """
    return AnalyzeRequest(
        question="When was the Sigiriya rock fortress built and who built it?",
        context=(
            "Sigiriya is an ancient rock fortress located in the Matale District "
            "of Sri Lanka. According to the Culavamsa, the site was selected by "
            "King Kashyapa as his new capital. He built his palace on top of the "
            "rock and decorated its sides with colourful frescoes."
        ),
        answer=(
            "Sigiriya was built by King Kashyapa in 477 CE as his capital. The "
            "fortress cost approximately 40 million gold coins and employed over "
            "25,000 workers during its construction."
        ),
        alpha=0.1,
    )
