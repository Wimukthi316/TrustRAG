# TrustRAG

**Span-Level Calibrated Hallucination Detection for Retrieval-Augmented Generation**

SLIIT IT4010 Research Project (CDAP) · Wimukthi Gunarathna (IT22244970) · Supervisor: Mr. Samadhi Rathnayaka · CoEAI

RAG systems hallucinate — RAGTruth reports 43.1% of responses from six leading
LLMs contained hallucinated content even with relevant context supplied. Existing
evaluation tools return one aggregate score per answer: a reviewer learns an
answer is "0.62 faithful" but not *which words* are unsupported. Span-level
detectors localise, but emit uncalibrated probabilities with no statistical
guarantee.

TrustRAG attaches to any RAG pipeline and returns, per span: where the
hallucination is, a calibrated confidence, and a conformal decision
(flag / abstain / pass) carrying a distribution-free coverage guarantee.

> **Current state: scaffold.** The backend runs a placeholder detector whose
> scores are keyword-matching heuristics. No model has been trained yet, so no
> number produced by this repository is a real result.

## Components

| | What | Status |
|---|---|---|
| **C1** | Span-level hallucination detector (ModernBERT token classification) | in progress |
| **C2** | Calibration + conformal abstention | in progress |
| C3 | Error taxonomy, explanation, bounded verification agent | planned |
| C4 | Retrieval-attribution alignment | planned |

## Requirements

- Python 3.11
- Node.js 20 or later
- A CUDA GPU is optional locally; training targets Kaggle (P100 / T4)

## Setup

```powershell
git clone https://github.com/Wimukthi316/TrustRAG.git
cd TrustRAG

powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
powershell -ExecutionPolicy Bypass -File scripts\setup_frontend.ps1

copy .env.example .env    # then fill in HF_TOKEN and WANDB_API_KEY
```

## Running

```powershell
# terminal 1 — API on :8000
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000

# terminal 2 — UI on :5173
cd frontend
npm run dev
```

Open <http://localhost:5173> and click **Load example**.
Interactive API docs: <http://127.0.0.1:8000/docs>.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Layout

```
src/common/schema.py       the shared JSON contract — every component speaks this
src/c1_detector/           preprocessing, training, evaluation
src/c2_calibration/        temperature scaling, ECE, split conformal
src/c3_explanation/        error taxonomy and explanation
src/c4_attribution/        evidence alignment
backend/app/main.py        FastAPI: /api/health, /api/analyze, /api/example
backend/app/services/      detector implementations
frontend/src/types.ts      TypeScript mirror of schema.py — keep in sync
notebooks/                 thin Kaggle notebooks: clone, install, call src/
eval/                      evaluation entry points
results/                   metric dumps (gitignored)
configs/                   training configs
```

## The data contract

`src/common/schema.py` defines `Span` and `AnalysisResult`. C1 fills the detection
fields, C2 the calibration and conformal fields, C3 and C4 theirs. Unset fields are
`null` and the UI degrades gracefully, so components can ship independently.

`frontend/src/types.ts` mirrors it by hand — **change both in the same commit.**

Validators enforce that `answer[start:end] == span.text` and that no span runs past
the end of the answer, which catches offset-mapping errors early.

## Data

Neither dataset is committed; `data/` is gitignored.

- **RAGTruth** — Niu et al., ACL 2024, arXiv:2401.00396, MIT licence. 2,965
  instances, 17,790 responses, 14,289 human-annotated hallucination spans,
  450-instance test split, three task types, four error categories. Used for span
  supervision.
- **RAGBench** — arXiv:2407.11005, CC-BY-4.0, `rungalileo/ragbench`. Used only as
  an out-of-distribution test set.

## Baseline

`KRLabsOrg/lettucedect-large-modernbert-en-v1` — 79.22% example-level F1 as
reported by its authors. Note the model ID is spelled "lettucedect".

## Licence

Academic project. Datasets and pretrained models retain their own licences.
