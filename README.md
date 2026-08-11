# TrustRAG

**Span-Level Calibrated Hallucination Detection for Retrieval-Augmented Generation**

SLIIT IT4010 Research Project (CDAP) · Wimukthi (IT22244970) · Supervisor: Mr. Samadhi Rathnayaka · CoEAI

RAG systems hallucinate — RAGTruth reports 43.1% of responses from six leading
LLMs contained hallucinated content even with relevant context supplied. Existing
evaluation tools return one aggregate score per answer: a reviewer learns an
answer is "0.62 faithful" but not *which words* are unsupported. Span-level
detectors localise, but emit uncalibrated probabilities with no statistical
guarantee.

TrustRAG attaches to any RAG pipeline and returns, per span: where the
hallucination is, a calibrated confidence, and a conformal decision
(flag / abstain / pass) carrying a distribution-free coverage guarantee.

> ⚠️ **Current state: scaffold only.** The backend runs a `StubDetector` whose
> scores are keyword-matching placeholders. No model has been trained. No number
> produced by this repo today is real. See `notes/STATUS.md`.

## Components

| | What | Status |
|---|---|---|
| **C1** | Span-level hallucination detector (ModernBERT token classification) | in scope |
| **C2** | Calibration + conformal abstention — **the novelty** | in scope |
| C3 | Error taxonomy, explanation, bounded agent | deferred |
| C4 | Retrieval-attribution alignment | deferred |

## Quick start

Requires Python 3.11 and Node 20+ (Node 24 LTS installed).

```powershell
# one-time setup
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
powershell -ExecutionPolicy Bypass -File scripts\setup_frontend.ps1

# terminal 1 — API on :8000
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000

# terminal 2 — UI on :5173
cd frontend
npm run dev
```

Open http://localhost:5173 and click **Load example**. API docs are at
http://127.0.0.1:8000/docs.

## Layout

```
src/common/schema.py       ⭐ the frozen JSON contract — every component speaks this
src/c1_detector/           preprocessing, training, evaluation
src/c2_calibration/        temperature scaling, ECE, split conformal
backend/app/main.py        FastAPI: /api/health, /api/analyze, /api/example
backend/app/services/      detector implementations (stub → LettuceDetect → C1)
frontend/src/types.ts      TypeScript mirror of schema.py — keep them in sync
notebooks/                 thin Kaggle notebooks: clone, install, call src/
notes/STATUS.md            ⭐ current state, updated every session
notes/ACCOUNTS.md          the manual signup steps
CLAUDE.md                  standing context for Claude Code
```

## The contract

`src/common/schema.py` defines `Span` and `AnalysisResult`. C1 fills the detection
fields, C2 fills calibration and conformal fields, C3 and C4 fill theirs if they
ever ship. Unset fields are `null` and the UI degrades gracefully.

`frontend/src/types.ts` mirrors it by hand. **Change both in the same commit.**

## Push to GitHub

Create an **empty** private repo named `trustrag` (no README, no .gitignore),
then:

```powershell
git remote add origin https://github.com/<your-username>/trustrag.git
git push -u origin main
```

## Data

Neither dataset is committed — `data/` is gitignored.

- **RAGTruth** (Niu et al., ACL 2024, arXiv:2401.00396, MIT) — span supervision
- **RAGBench** (arXiv:2407.11005, CC-BY-4.0, `rungalileo/ragbench`) — OOD test only

## Baseline

`KRLabsOrg/lettucedect-large-modernbert-en-v1` — 79.22% example-level F1 as
reported by its authors. The spelling "lettucedect" is their own typo; copy it
exactly.
