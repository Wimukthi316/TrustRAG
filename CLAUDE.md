# CLAUDE.md — TrustRAG repo context

Read this before touching anything. It is the standing context for Claude Code
in this repository.

## Who / what

Wimukthi (IT22244970), 4th-year BSc Data Science, SLIIT, module IT4010 Research
Project (CDAP, 16 credits, 4-member group). Supervisor: Mr. Samadhi Rathnayaka,
research group CoEAI.

**TrustRAG** — Span-Level Calibrated Hallucination Detection for
Retrieval-Augmented Generation. A model-agnostic trust layer that attaches to any
RAG pipeline: it localises hallucinated spans, attaches calibrated confidence with
a distribution-free conformal coverage guarantee, aligns spans to supporting
context, and explains the error type.

**The gap:** span-detection literature and conformal-prediction literature have
not been joined. Conformal factuality work (Mohri & Hashimoto, ICML 2024;
Conformal-RAG, Feng et al., SIGIR 2025) operates at claim/sub-claim level.
Span-level detectors (LettuceDetect) emit uncalibrated probabilities with no
guarantee. **C2 is the novelty.**

## Components

| | Owner | Status |
|---|---|---|
| C1 span detector | Sadini (IT22239648) — **Wimukthi is building it** | in scope |
| **C2 calibration + conformal** | **Wimukthi — the core novelty** | in scope |
| C3 explanation / taxonomy / agent | Ranya (IT22129062) | deferred, only if C1+C2 land early |
| C4 retrieval-attribution | Sakindu (IT22047724) | deferred, only if C1+C2 land early |

## Deadline

PP2 on **2026-08-30**. Repo started 2026-08-11. TAF, Proposal Report, Proposal
Presentation and PP1 are already done. The PP2 rubric has NOT been supplied —
deliverables below are inferred from the PP1 rubric and may be wrong.

## Hard constraints

- **Kaggle is the primary training environment** (P100 16GB or T4×2, roughly
  30 GPU-hr/week, 12-hour sessions). Verify the quota at run time; it changes.
- **Local GPU: RTX 3050 Laptop, 6GB VRAM.** Useful for smoke tests and inference,
  NOT for full fine-tuning runs. Earlier planning docs said "no local GPU" — that
  was wrong, but 6GB is still too small for ModernBERT-large training.
- **No paid compute. No paid APIs.** Free tiers only (Gemini / Groq / OpenRouter
  `:free`), rate-limited, cache aggressively.
- Claude writes the code; Wimukthi runs it on Kaggle and pastes back errors and
  results. Claude cannot run training, download datasets, or produce real numbers.

## Stack (locked — do not re-litigate)

Python 3.11 · PyTorch · HuggingFace transformers/datasets/evaluate/seqeval/accelerate ·
netcal + torchmetrics + scikit-learn for calibration · **FastAPI backend** ·
**React 19 + Vite + Tailwind v4 frontend** (Node 24 LTS).

Note: an earlier plan specified Streamlit. Wimukthi overrode that on 2026-08-11 in
favour of React + Vite + Tailwind + FastAPI. That is now the decision.

## Data and baselines

- **RAGTruth** (Niu et al., ACL 2024, arXiv:2401.00396, MIT): 2,965 instances ×
  6 LLMs = 17,790 responses, 14,289 human spans, 43.1% hallucination density,
  450-instance test split (150/task), 3 tasks, 4 error categories. Span supervision.
- **RAGBench** (arXiv:2407.11005, CC-BY-4.0, `rungalileo/ragbench`): OOD test only.
- Baseline to beat: `KRLabsOrg/lettucedect-large-modernbert-en-v1`, 79.22%
  example-level F1. **The spelling "lettucedect" is the authors' own typo — copy
  it exactly or you get a 404.**
- Backbone: `answerdotai/ModernBERT-base` (149M) first; `-large` (395M) only if
  base succeeds early. `microsoft/deberta-v3-large` as ablation control.

## The JSON contract

`src/common/schema.py` is **frozen**. C1, C2, the FastAPI backend and the React
frontend all speak those exact shapes. `frontend/src/types.ts` mirrors it — change
both in the same commit or the demo breaks.

## Repo layout

```
configs/     training configs (yaml)
data/        gitignored — RAGTruth/RAGBench live here
src/common/  schema.py = the JSON contract
src/c1_detector/   preprocessing, training, eval
src/c2_calibration/ temperature scaling, ECE, split conformal
backend/     FastAPI app (app/main.py, app/services/detector.py)
frontend/    React + Vite + Tailwind
notebooks/   thin Kaggle notebooks: clone + install + run, no logic
eval/        evaluation entry points
results/     gitignored except .gitkeep — JSON metric dumps
notes/       STATUS.md = the daily save-game
```

## Rules for Claude

1. Reply in Singlish (Sinhala in Latin script, English technical terms). Concise,
   no preamble.
2. **Never fabricate a benchmark number.** Write `TODO` where a real number belongs.
   Every number in the report must come from a run Wimukthi actually executed.
3. Never invent paper titles, arXiv IDs, DOIs or URLs.
4. Free-tier quotas and prices change — say "verify this", do not assert.
5. Ask before adding a dependency, changing scope, or refactoring untouched code.
6. Notebooks stay thin. Logic lives in `src/`, imported by the notebook.
7. When time is at risk, recommend the working fallback over the ambitious path.

## Key acceleration rule

**Do not wait for C1 training to finish before starting C2.** While C1 fine-tunes
on Kaggle in the background, build the C2 conformal harness against the public
LettuceDetect checkpoint's output probabilities, then swap the probability source.
This is worth 3–4 days and one person owns both components.

## Trigger points

- **Aug 14** — C1 example-F1 < 0.70: stop tuning, wrap public LettuceDetect. The
  novelty is C2, not C1.
- **Aug 17** — conformal empirical coverage ≠ target: stop everything and debug
  the maths. There is no project without this result.
- **Aug 24** — results freeze. No new experiments after this date.

## Current state

See `notes/STATUS.md`. If it looks stale, say so.
