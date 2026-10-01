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

from dotenv import load_dotenv  # noqa: E402

# Read .env from the repo root rather than the process working directory. The
# server is launched from several places -- scripts\run_demo.ps1, uvicorn by
# hand, the test client -- and services/history.py reads os.environ and nothing
# else, so without this MONGODB_URI is silently invisible and the history store
# reports itself unconfigured while a perfectly good cluster sits behind it.
# override=False so a variable already set in the shell still wins.
load_dotenv(REPO_ROOT / ".env", override=False)

from contextlib import asynccontextmanager  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from typing import Optional  # noqa: E402

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from backend.app.services.detector import get_detector  # noqa: E402
from backend.app.services.extract import (  # noqa: E402
    ExtractError,
    extract_text,
)
from backend.app.services.history import get_store  # noqa: E402
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
    HistoryDetail,
    HistoryEntry,
    HistoryPage,
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


class ExtractResponse(BaseModel):
    """Plain text pulled from an uploaded file.

    Kept local to the backend rather than in schema.py: it is not part of the
    analysis contract the four components speak, only the plumbing behind the
    "attach a file" button, so it stays out of the parity check.
    """

    filename: str
    chars: int
    pages: Optional[int] = None
    text: str
    note: Optional[str] = None


@app.post("/api/extract", response_model=ExtractResponse)
async def extract(file: UploadFile = File(...)) -> ExtractResponse:
    """Turn an attached PDF, Word .docx or text file into plain text.

    The interface drops the result into the document box; the model never sees
    the file, only the text a person can read and edit first. Nothing is written
    to disk -- the bytes live only for the length of this request. This runs no
    model and makes no claim about the text.
    """
    data = await file.read()
    try:
        result = extract_text(file.filename or "", data)
    except ExtractError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ExtractResponse(
        filename=file.filename or "file",
        chars=len(result.text),
        pages=result.pages,
        text=result.text,
        note=result.note,
    )


# --- history ----------------------------------------------------------------
# Saved checks, so a person can open one again.
#
# Every route here treats an absent or unreachable store as a normal answer
# rather than as an error. A history page that loads and says "the store is
# unreachable" is worth more than one that fails to load, and the interface has
# a real fallback for it -- it keeps the checks in the browser instead.


@app.get("/api/history", response_model=HistoryPage)
def history(
    q: Optional[str] = None,
    task: Optional[str] = None,
    only_flagged: bool = False,
    limit: Optional[int] = None,
    offset: int = 0,
) -> HistoryPage:
    """Saved checks, newest first, with the filter applied.

    `total` counts everything matching the filter, not the page. A reader who
    narrowed two hundred checks down to three needs to be told both numbers.
    """
    store = get_store()
    if not store.available():
        return HistoryPage(available=False, backend=store.backend, note=store.note())
    try:
        entries, total = store.list(q, task, only_flagged, limit, offset)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return HistoryPage(
        available=True,
        backend=store.backend,
        total=total,
        entries=entries,
    )


@app.post("/api/history", response_model=HistoryEntry, status_code=201)
def history_save(result: AnalysisResult) -> HistoryEntry:
    """Keep one result.

    The analyse route does not write here. Analysing and remembering are
    separate things a caller might want separately, and a POST that quietly
    persisted its input would be the wrong default for a tool whose whole
    subject is what a system does without telling you.
    """
    store = get_store()
    if not store.available():
        raise HTTPException(status_code=503, detail=store.note())
    return store.save(result)


@app.get("/api/history/export")
def history_export() -> dict:
    """Everything, as one JSON document, for a person who wants to keep it.

    Above the /{entry_id} route on purpose: FastAPI matches in declaration
    order, and declared the other way round this path would be read as an id.
    """
    store = get_store()
    if not store.available():
        raise HTTPException(status_code=503, detail=store.note())
    checks = store.export()
    return {
        "schema_version": SCHEMA_VERSION,
        "exported_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "backend": store.backend,
        "count": len(checks),
        "checks": checks,
    }


@app.get("/api/history/{entry_id}", response_model=HistoryDetail)
def history_get(entry_id: str) -> HistoryDetail:
    """One saved check, in full."""
    store = get_store()
    if not store.available():
        raise HTTPException(status_code=503, detail=store.note())
    detail = store.get(entry_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"no saved check {entry_id!r}")
    return detail


@app.delete("/api/history/{entry_id}", status_code=204)
def history_delete(entry_id: str) -> None:
    """Forget one check.

    A person who pasted a document into this must be able to take it back out.
    """
    store = get_store()
    if not store.available():
        raise HTTPException(status_code=503, detail=store.note())
    if not store.delete(entry_id):
        raise HTTPException(status_code=404, detail=f"no saved check {entry_id!r}")


@app.delete("/api/history", status_code=200)
def history_clear() -> dict:
    """Forget all of them."""
    store = get_store()
    if not store.available():
        raise HTTPException(status_code=503, detail=store.note())
    return {"deleted": store.clear()}


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
