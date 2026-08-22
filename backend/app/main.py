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

from fastapi.responses import FileResponse  # noqa: E402

from backend.app.services.detector import get_detector  # noqa: E402
from backend.app.services.metrics import (  # noqa: E402
    FIGURE_DIR,
    FIGURE_ORDER,
    build_metrics,
)
from src.common.schema import (  # noqa: E402
    SCHEMA_VERSION,
    AnalysisResult,
    AnalyzeRequest,
    HealthResponse,
    MetricsResponse,
    TaskType,
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


@app.get("/api/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """C2's offline evidence, so the demo can show its working.

    Read-only, computed nowhere, cached for the life of the process. When no
    results are on disk it returns `available: false` and a note saying which
    command produces them -- a metrics tab that says "nothing measured yet" is
    honest, and a 500 is not.
    """
    return build_metrics()


@app.get("/api/figures/{name}.png")
def figure(name: str) -> FileResponse:
    """One report figure, by name.

    The name is checked against a fixed list rather than joined onto a path.
    Interpolating a user-supplied string into a filesystem path is how a demo
    server ends up serving `.env`, and the fact that this one only ever runs on
    a laptop is not a reason to write it the other way.
    """
    if name not in FIGURE_ORDER:
        raise HTTPException(
            status_code=404, detail=f"unknown figure {name!r}; try {list(FIGURE_ORDER)}"
        )
    path = FIGURE_DIR / f"{name}.png"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"{name}.png has not been generated. Run "
                "`python -m src.c2_calibration.figures`."
            ),
        )
    return FileResponse(path, media_type="image/png")


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


# RAGTruth test record 16121 (MIT licence), question-answering task, response
# generated by llama-2-70b-chat. Kept verbatim, offsets and all, so what the demo
# shows can be checked against the corpus: the annotators marked [216:310] and
# [311:379] as hallucinated.
#
# Why a real record is the default. The detector is trained on RAGTruth, and a
# hand-written toy is out of distribution for it -- measured on this repository,
# 2026-08-13: over 146 test responses C1 marks a median of 0.4% of the answer and
# never once marked the whole thing, but on the hand-written example below it
# marks all 173 characters. The demo should show the model on the input it was
# built for. The hand-written example is kept, and named honestly, because that
# failure is worth showing too.
RAGTRUTH_QA = AnalyzeRequest(
    question="how to quickly get rid of mice",
    context=(
        "passage 1:Whatever your situation is, the best way to get rid of the "
        "mice is do it fast. You need to take immediate action. This is a "
        "critical step to getting rid of mice. Mice can very quickly take up "
        "residence in your home and start nesting.\n\n"
        "passage 2:1 Mice can use these entry points as highways to get into "
        "your house. 2  Consider putting steel wool inside the pipe or vent "
        "before you seal it off. 3  Steel wool pads will prevent mice from "
        "using the hole to get back into your house. 4  It will add an extra "
        "element of protection if your mice are persistent and resilient.\n\n"
        "passage 3:Get Rid of Rats and Mice. Rats and mice will quickly take "
        "advantage of any source of food or shelter. These rodents will consume "
        "any food discarded by humans, and they can fit through the tiniest of "
        "openings to access your home. Learn how to eliminate these pests from "
        "your home, and prevent them from returning.\n\n"
    ),
    answer=(
        "To quickly get rid of mice, take immediate action and seal all entry "
        "points, including pipes and vents, using steel wool pads to prevent "
        "mice from getting back in. Eliminate sources of food and shelter, and "
        "consider using humane mouse traps or natural deterrents like "
        "peppermint oil to encourage mice to leave. Prevent their return by "
        "keeping your home clean and well-maintained."
    ),
    alpha=0.1,
    task_type=TaskType.QA,
)

HANDWRITTEN = AnalyzeRequest(
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
    task_type=TaskType.QA,
)

EXAMPLES = {"ragtruth_qa": RAGTRUTH_QA, "handwritten": HANDWRITTEN}


@app.get("/api/example", response_model=AnalyzeRequest)
def example(name: str = "ragtruth_qa") -> AnalyzeRequest:
    """A canned request so the frontend has a 'Load example' button.

    `ragtruth_qa` is a real corpus record and is the default. `handwritten` is a
    short invented one, kept deliberately: it is far shorter than anything in
    RAGTruth and the detector over-flags on it, which is a limitation the demo
    should be able to show rather than hide.
    """
    if name not in EXAMPLES:
        raise HTTPException(
            status_code=404,
            detail=f"unknown example {name!r}; try {sorted(EXAMPLES)}",
        )
    return EXAMPLES[name]
